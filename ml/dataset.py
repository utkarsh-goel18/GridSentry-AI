"""
GridSentry AI — Dataset and Preprocessing Pipeline
Partitions recordings by independent physical simulation scenarios,
extracts 6-channel waveform windows, normalizes using training data only,
and provides PyTorch DataLoaders.
"""

import io
import json
import hashlib
import re
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import scipy.io
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from ml.config import (
    DATASET_1_ZIP,
    DATASET_2_ZIP,
    ARTIFACTS_DIR,
    SCALER_PATH,
    CLASS_MAPPING_PATH,
    CONFIG_PATH,
    STANDARD_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    RAW_TO_STANDARD_CLASS,
    WINDOW_LENGTH,
    D1_FAULT_INDEX,
    D2_FAULT_INDEX,
    PRE_FAULT_SAMPLES,
    RANDOM_SEED,
    BATCH_SIZE
)


class GridSentryDataset(Dataset):
    """PyTorch Dataset for 6-channel transmission-line waveform windows."""

    def __init__(self, waveforms: np.ndarray, labels: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None):
        """
        waveforms: shape (N, 6, window_length)
        labels: shape (N,)
        metadata: optional list of dicts with scenario info
        """
        self.waveforms = torch.from_numpy(waveforms).float()
        self.labels = torch.from_numpy(labels).long()
        self.metadata = metadata or []

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.waveforms[idx], self.labels[idx]


def get_scenario_splits(
    random_seed: int = RANDOM_SEED,
    split_strategy: str = "scenario",
    holdout_value: float = 20
) -> Tuple[set, set, set]:
    """
    Build leakage-aware physical-scenario partitions.

    scenario: random angle holdout within every distance/resistance pair.
    distance: hold out every scenario at one complete distance.
    resistance: hold out every scenario at one complete fault resistance.

    For condition holdouts, validation scenarios are selected only from the
    remaining conditions; the held-out condition is used exclusively for test.
    """
    if split_strategy not in {"scenario", "distance", "resistance"}:
        raise ValueError("split_strategy must be scenario, distance, or resistance")

    rng = np.random.default_rng(random_seed)
    dists = [20, 150, 280]
    rfs = [1.0, 50.0, 100.0]
    angles = list(range(0, 190, 10))

    if split_strategy == "distance" and holdout_value not in dists:
        raise ValueError(f"holdout distance must be one of {dists}")
    if split_strategy == "resistance" and float(holdout_value) not in rfs:
        raise ValueError(f"holdout resistance must be one of {rfs}")

    train_scenarios, val_scenarios, test_scenarios = set(), set(), set()

    for d in dists:
        for r in rfs:
            if split_strategy == "distance" and d == holdout_value:
                test_scenarios.update((d, r, a) for a in angles)
                continue
            if split_strategy == "resistance" and r == float(holdout_value):
                test_scenarios.update((d, r, a) for a in angles)
                continue

            shuffled_angles = rng.permutation(angles).tolist()
            if split_strategy == "scenario":
                tr_a, va_a, te_a = shuffled_angles[:13], shuffled_angles[13:16], shuffled_angles[16:19]
                train_scenarios.update((d, r, a) for a in tr_a)
                val_scenarios.update((d, r, a) for a in va_a)
                test_scenarios.update((d, r, a) for a in te_a)
            else:
                # Keep a validation subset from non-held-out conditions.
                va_a = shuffled_angles[:3]
                tr_a = shuffled_angles[3:]
                val_scenarios.update((d, r, a) for a in va_a)
                train_scenarios.update((d, r, a) for a in tr_a)

    all_scenarios = train_scenarios | val_scenarios | test_scenarios
    if len(all_scenarios) != 171:
        raise RuntimeError(f"Expected 171 unique scenarios, found {len(all_scenarios)}")
    if (train_scenarios & val_scenarios) or (train_scenarios & test_scenarios) or (val_scenarios & test_scenarios):
        raise RuntimeError("Scenario split overlap detected")
    if split_strategy == "distance" and any(s[0] != holdout_value for s in test_scenarios):
        raise RuntimeError("Distance holdout contains an unexpected test scenario")
    if split_strategy == "resistance" and any(s[1] != float(holdout_value) for s in test_scenarios):
        raise RuntimeError("Resistance holdout contains an unexpected test scenario")

    return train_scenarios, val_scenarios, test_scenarios

def _cache_fingerprint(use_jitter: bool, split_strategy: str, holdout_value: float) -> str:
    """Fingerprint source archive metadata and preprocessing settings for cache invalidation."""
    sources = []
    for archive_path in (DATASET_1_ZIP, DATASET_2_ZIP):
        if archive_path.exists():
            stat = archive_path.stat()
            sources.append({
                "path": archive_path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns
            })
        else:
            sources.append({"path": archive_path.name, "missing": True})
    payload = {
        "cache_schema": 3,
        "sources": sources,
        "use_jitter": bool(use_jitter),
        "split_strategy": split_strategy,
        "holdout_value": holdout_value,
        "random_seed": RANDOM_SEED,
        "window_length": WINDOW_LENGTH,
        "d1_fault_index": D1_FAULT_INDEX,
        "d2_fault_index": D2_FAULT_INDEX,
        "pre_fault_samples": PRE_FAULT_SAMPLES,
        "classes": STANDARD_CLASSES
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_preprocess_all(
    cache_dir: Optional[Path] = None,
    use_jitter: bool = True,
    split_strategy: str = "scenario",
    holdout_value: float = 20
) -> Tuple[GridSentryDataset, GridSentryDataset, GridSentryDataset, Dict[str, Any]]:
    """
    Loads both Dataset 1 and Dataset 2, extracts 6-channel windows,
    splits by scenario, calculates training scaler, and returns Datasets.
    Uses cached preprocessed .npz if available.
    """
    if cache_dir is None:
        cache_dir = ARTIFACTS_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "preprocessed_dataset.npz"
    summary_file = ARTIFACTS_DIR / "dataset_split_summary.json"

    train_scenarios, val_scenarios, test_scenarios = get_scenario_splits(
        split_strategy=split_strategy, holdout_value=holdout_value
    )

    fingerprint = _cache_fingerprint(use_jitter, split_strategy, holdout_value)
    cache_valid = False
    if cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as cached:
                cache_valid = (
                    "cache_fingerprint" in cached.files
                    and str(cached["cache_fingerprint"].item()) == fingerprint
                )
                if cache_valid:
                    x_train, y_train = cached['x_train'], cached['y_train']
                    x_val, y_val = cached['x_val'], cached['y_val']
                    x_test, y_test = cached['x_test'], cached['y_test']
                    scaler = json.loads(str(cached['scaler_json'].item()))
        except (OSError, ValueError, KeyError, EOFError):
            cache_valid = False

    if cache_valid:
        print(f"Loading validated preprocessed dataset cache: {cache_file}")
    else:
        if cache_file.exists():
            print("Preprocessed cache is stale or unreadable; rebuilding it.")
        print("Building preprocessed dataset from raw ZIP archives...")
        # Containers for splits
        raw_splits = {'train': [], 'val': [], 'test': []}
        labels_splits = {'train': [], 'val': [], 'test': []}
        meta_splits = {'train': [], 'val': [], 'test': []}

        # 1. Process Dataset 2 (Series-Compensated)
        print("Extracting Dataset 2 recordings...")
        angles_list = list(range(0, 190, 10))
        dists_list = [20, 150, 280]
        rfs_list = [1.0, 50.0, 100.0]
        comps_list = [0.4, 0.5, 0.6, 0.7]

        with zipfile.ZipFile(DATASET_2_ZIP, 'r') as z2:
            for cls_name in STANDARD_CLASSES:
                cls_idx = CLASS_TO_IDX[cls_name]
                raw_bytes = z2.read(f"signals_with_compensation (dataset 2)/falta_{cls_name}.mat")
                with h5py.File(io.BytesIO(raw_bytes), 'r') as hf:
                    v_all = hf[f"V_{cls_name}"][:]  # (19, 4, 3, 3, 3, 1537)
                    i_all = hf[f"I_{cls_name}"][:]  # (19, 4, 3, 3, 3, 1537)

                    for a_idx, ang in enumerate(angles_list):
                        for r_idx, rf in enumerate(rfs_list):
                            for d_idx, dist in enumerate(dists_list):
                                scenario_key = (dist, rf, ang)
                                if scenario_key in train_scenarios:
                                    target_split = 'train'
                                elif scenario_key in val_scenarios:
                                    target_split = 'val'
                                else:
                                    target_split = 'test'

                                # Iterate over all 4 compensation factors
                                for c_idx, comp in enumerate(comps_list):
                                    v_trace = v_all[a_idx, c_idx, r_idx, d_idx, :, :]  # (3, 1537)
                                    i_trace = i_all[a_idx, c_idx, r_idx, d_idx, :, :]  # (3, 1537)
                                    sig_6ch = np.vstack([v_trace, i_trace]).astype(np.float32)  # (6, 1537)

                                    # Extract window(s)
                                    offsets = [-64]
                                    if target_split == 'train' and use_jitter:
                                        # Data augmentation for training: slight jitter shifts
                                        offsets = [-80, -64, -48]

                                    for off in offsets:
                                        start = D2_FAULT_INDEX + off
                                        end = start + WINDOW_LENGTH
                                        if start >= 0 and end <= sig_6ch.shape[1]:
                                            win = sig_6ch[:, start:end]
                                            raw_splits[target_split].append(win)
                                            labels_splits[target_split].append(cls_idx)
                                            meta_splits[target_split].append({
                                                'fault': cls_name,
                                                'distance_km': dist,
                                                'resistance_ohm': rf,
                                                'angle_deg': ang,
                                                'compensation': comp,
                                                'dataset': 'Dataset 2'
                                            })

        # 2. Process Dataset 1 (Uncompensated)
        print("Extracting Dataset 1 recordings...")
        d1_pattern = re.compile(r'sim_Chesf_500_Classifier_LOC3_([A-Z]+)_(\d+)_(\d+)_([0-9.]+)\.mat')
        with zipfile.ZipFile(DATASET_1_ZIP, 'r') as z1:
            for fname in z1.namelist():
                if not fname.endswith('.mat'):
                    continue
                basename = Path(fname).name
                m = d1_pattern.search(basename)
                if not m:
                    continue
                raw_flt, dist_s, ang_s, rf_s = m.groups()
                std_flt = RAW_TO_STANDARD_CLASS.get(raw_flt, raw_flt)
                cls_idx = CLASS_TO_IDX[std_flt]
                dist = int(dist_s)
                ang = int(ang_s)
                rf = float(rf_s)

                scenario_key = (dist, rf, ang)
                if scenario_key in train_scenarios:
                    target_split = 'train'
                elif scenario_key in val_scenarios:
                    target_split = 'val'
                else:
                    target_split = 'test'

                mat = scipy.io.loadmat(io.BytesIO(z1.read(fname)))
                vl = mat['V_L']  # (7681, 3)
                il = mat['I_L']  # (7681, 3)
                sig_6ch = np.vstack([vl.T, il.T]).astype(np.float32)  # (6, 7681)

                offsets = [-64]
                if target_split == 'train' and use_jitter:
                    offsets = [-80, -64, -48]

                for off in offsets:
                    start = D1_FAULT_INDEX + off
                    end = start + WINDOW_LENGTH
                    if start >= 0 and end <= sig_6ch.shape[1]:
                        win = sig_6ch[:, start:end]
                        raw_splits[target_split].append(win)
                        labels_splits[target_split].append(cls_idx)
                        meta_splits[target_split].append({
                            'fault': std_flt,
                            'distance_km': dist,
                            'resistance_ohm': rf,
                            'angle_deg': ang,
                            'compensation': 0.0,
                            'dataset': 'Dataset 1'
                        })

        # Convert to numpy arrays
        x_train_raw = np.array(raw_splits['train'], dtype=np.float32)
        y_train = np.array(labels_splits['train'], dtype=np.int64)
        x_val_raw = np.array(raw_splits['val'], dtype=np.float32)
        y_val = np.array(labels_splits['val'], dtype=np.int64)
        x_test_raw = np.array(raw_splits['test'], dtype=np.float32)
        y_test = np.array(labels_splits['test'], dtype=np.int64)

        print(f"Extracted windows — Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        # 3. Calculate normalization parameters STRICTLY on training set
        print("Computing training set scaler parameters...")
        # x_train_raw shape: (N, 6, L) -> compute per channel across all samples and time steps
        # axes: (0, 2)
        ch_means = np.mean(x_train_raw, axis=(0, 2), dtype=np.float64)  # (6,)
        ch_stds = np.std(x_train_raw, axis=(0, 2), dtype=np.float64)    # (6,)

        # Protect against potential zero std
        ch_stds = np.where(ch_stds < 1e-6, 1.0, ch_stds)

        scaler = {
            'mean': [float(m) for m in ch_means],
            'std': [float(s) for s in ch_stds],
            'channels': ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]
        }

        # Apply normalization
        mean_b = ch_means[None, :, None].astype(np.float32)
        std_b = ch_stds[None, :, None].astype(np.float32)

        x_train = (x_train_raw - mean_b) / std_b
        x_val = (x_val_raw - mean_b) / std_b
        x_test = (x_test_raw - mean_b) / std_b

        # Save cache and scaler
        np.savez_compressed(
            cache_file,
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            scaler_json=json.dumps(scaler),
            cache_fingerprint=np.asarray(fingerprint)
        )
        print(f"Saved dataset cache to: {cache_file}")

        # Save scaler params
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SCALER_PATH, 'w') as f:
            json.dump(scaler, f, indent=2)

        # Save class mapping
        class_mapping_data = {
            'standard_classes': STANDARD_CLASSES,
            'class_to_idx': CLASS_TO_IDX,
            'idx_to_class': IDX_TO_CLASS,
            'num_classes': len(STANDARD_CLASSES)
        }
        with open(CLASS_MAPPING_PATH, 'w') as f:
            json.dump(class_mapping_data, f, indent=2)

        # Save dataset split summary
        split_summary = {
            'total_physical_scenarios': 171,
            'split_strategy': split_strategy,
            'holdout_value': holdout_value if split_strategy != 'scenario' else None,
            'train_scenarios_count': len(train_scenarios),
            'val_scenarios_count': len(val_scenarios),
            'test_scenarios_count': len(test_scenarios),
            'train_windows': int(len(y_train)),
            'val_windows': int(len(y_val)),
            'test_windows': int(len(y_test)),
            'window_length': WINDOW_LENGTH,
            'num_channels': 6,
            'channels': scaler['channels'],
            'scaler': scaler
        }
        with open(summary_file, 'w') as f:
            json.dump(split_summary, f, indent=2)
        print(f"Saved split summary to: {summary_file}")

    train_ds = GridSentryDataset(x_train, y_train)
    val_ds = GridSentryDataset(x_val, y_val)
    test_ds = GridSentryDataset(x_test, y_test)

    return train_ds, val_ds, test_ds, scaler


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    split_strategy: str = "scenario",
    holdout_value: float = 20
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Returns PyTorch DataLoaders for train, val, and test splits."""
    train_ds, val_ds, test_ds, scaler = load_and_preprocess_all(
        split_strategy=split_strategy, holdout_value=holdout_value
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    train_ds, val_ds, test_ds, scaler = load_and_preprocess_all()
    print("Dataset verification:")
    print(f"  Train samples: {len(train_ds)}, shape: {train_ds[0][0].shape}")
    print(f"  Val samples:   {len(val_ds)}, shape: {val_ds[0][0].shape}")
    print(f"  Test samples:  {len(test_ds)}, shape: {test_ds[0][0].shape}")
    print("  Scaler Means:", [f"{m:.2f}" for m in scaler['mean']])
    print("  Scaler Stds: ", [f"{s:.2f}" for s in scaler['std']])

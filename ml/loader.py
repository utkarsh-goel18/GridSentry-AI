"""
GridSentry AI — Reusable MATLAB Signal Loader
Loads and extracts 6-channel waveforms (Va, Vb, Vc, Ia, Ib, Ic)
from both MATLAB v5/v7 (.mat via scipy.io) and MATLAB v7.3 (.mat via h5py).
"""

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import scipy.io
import h5py

from ml.config import (
    DATASET_1_ZIP,
    DATASET_2_ZIP,
    RAW_TO_STANDARD_CLASS,
    SAMPLING_RATE_HZ,
    WINDOW_LENGTH,
    D1_FAULT_INDEX,
    D2_FAULT_INDEX,
    PRE_FAULT_SAMPLES
)


class MatlabSignalLoader:
    """
    Reusable loader for electrical transmission-line transient recordings.
    Extracts 6-channel signals [Va, Vb, Vc, Ia, Ib, Ic] and standardizes fault labels.
    """

    def __init__(self, d1_zip_path: Optional[Path] = None, d2_zip_path: Optional[Path] = None):
        self.d1_zip = Path(d1_zip_path or DATASET_1_ZIP)
        self.d2_zip = Path(d2_zip_path or DATASET_2_ZIP)

    @staticmethod
    def _as_three_phase_time(array: np.ndarray, variable_name: str) -> np.ndarray:
        """Return a finite float32 matrix in (3 phases, time) order."""
        values = np.asarray(array)
        if values.ndim != 2:
            raise ValueError(f"{variable_name} must be 2-D; received shape {values.shape}.")
        if values.shape[0] == 3:
            oriented = values
        elif values.shape[1] == 3:
            oriented = values.T
        else:
            raise ValueError(
                f"{variable_name} must have exactly 3 phase columns or rows; received {values.shape}."
            )
        oriented = np.asarray(oriented, dtype=np.float32)
        if not np.isfinite(oriented).all():
            raise ValueError(f"{variable_name} contains NaN or infinite values.")
        return oriented

    @staticmethod
    def parse_filename_metadata(filename: str) -> Dict[str, Any]:
        """
        Parses simulation parameters from filename format:
        sim_Chesf_500_Classifier_LOC3_{FAULT}_{DISTANCE}_{ANGLE}_{RESISTANCE}.mat
        """
        basename = Path(filename).name
        pattern = re.compile(r'sim_Chesf_500_Classifier_LOC3_([A-Z]+)_(\d+)_(\d+)_([0-9.]+)\.mat')
        m = pattern.search(basename)
        if m:
            raw_flt, dist, ang, rf = m.groups()
            return {
                'raw_fault': raw_flt,
                'standard_fault': RAW_TO_STANDARD_CLASS.get(raw_flt, raw_flt),
                'distance_km': int(dist),
                'angle_deg': int(ang),
                'resistance_ohm': float(rf),
                'compensation': 0.0,
                'dataset': 'Dataset 1 (Uncompensated)'
            }
        return {'raw_filename': basename}

    @staticmethod
    def load_mat_stream(stream_or_path: Union[str, Path, io.BytesIO, bytes]) -> Tuple[Dict[str, Any], str]:
        """
        Loads MATLAB data automatically detecting v5/v7 vs v7.3 format.
        Returns (dict_or_hdf_group, format_string).
        """
        if isinstance(stream_or_path, (str, Path)):
            stream = open(stream_or_path, 'rb')
        elif isinstance(stream_or_path, bytes):
            stream = io.BytesIO(stream_or_path)
        else:
            stream = stream_or_path

        # Try scipy.io first (v5/v7)
        try:
            stream.seek(0)
            data = scipy.io.loadmat(stream)
            return data, "v5_v7"
        except Exception:
            # Fall back to h5py (v7.3)
            stream.seek(0)
            hf = h5py.File(stream, 'r')
            return hf, "v7_3"

    @classmethod
    def extract_6channel_waveform(
        cls,
        mat_obj: Union[Dict[str, Any], h5py.File],
        mat_format: str,
        fault_type_hint: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extracts 6 channels [Va, Vb, Vc, Ia, Ib, Ic] of shape (6, N).
        """
        meta: Dict[str, Any] = {}

        if mat_format == "v5_v7":
            # Dataset 1 format
            # Candidate variables: 'V_L' and 'I_L' (preferred 7681x3)
            if 'V_L' in mat_obj and 'I_L' in mat_obj:
                vl = mat_obj['V_L']  # (N, 3)
                il = mat_obj['I_L']  # (N, 3)
            elif 'V' in mat_obj and 'I' in mat_obj:
                vl = mat_obj['V']
                il = mat_obj['I']
            else:
                raise KeyError("Could not find voltage and current variables (V_L/I_L or V/I) in mat file")

            # Extract metadata
            if 'Fault' in mat_obj:
                raw_fault = str(mat_obj['Fault'][0])
                meta['fault_label'] = RAW_TO_STANDARD_CLASS.get(raw_fault, raw_fault)
            if 'fs' in mat_obj:
                meta['sampling_rate'] = int(mat_obj['fs'][0, 0])
            if 'RealFaultTime' in mat_obj:
                meta['fault_time'] = float(mat_obj['RealFaultTime'][0, 0])

            voltage = cls._as_three_phase_time(vl, "voltage")
            current = cls._as_three_phase_time(il, "current")
            if voltage.shape != current.shape:
                raise ValueError(
                    f"Voltage/current dimensions disagree: {voltage.shape} vs {current.shape}."
                )
            signal_6ch = np.vstack([voltage, current]).astype(np.float32, copy=False)
            return signal_6ch, meta

        elif mat_format == "v7_3":
            # Dataset 2 format: arrays like V_AG, I_AG of shape (19, 4, 3, 3, 3, 1537)
            # Find matching keys
            vk, ik = None, None
            for k in mat_obj.keys():
                if k.startswith("V_"):
                    vk = k
                elif k.startswith("I_"):
                    ik = k
            if not vk or not ik:
                raise KeyError(f"Could not find V_ and I_ variables in HDF5 keys: {list(mat_obj.keys())}")

            flt_raw = vk.replace("V_", "")
            meta['fault_label'] = RAW_TO_STANDARD_CLASS.get(flt_raw, flt_raw)
            meta['sampling_rate'] = SAMPLING_RATE_HZ

            v_ds = mat_obj[vk]
            i_ds = mat_obj[ik]
            if v_ds.shape != i_ds.shape:
                raise ValueError(f"{vk}/{ik} dimensions disagree: {v_ds.shape} vs {i_ds.shape}.")
            if v_ds.ndim == 6:
                v_slice = v_ds[0, 0, 0, 0, :, :]
                i_slice = i_ds[0, 0, 0, 0, :, :]
            elif v_ds.ndim == 2:
                v_slice = v_ds[:]
                i_slice = i_ds[:]
            else:
                raise ValueError(f"Unexpected dimensionality for {vk}: {v_ds.shape}")

            voltage = cls._as_three_phase_time(v_slice, vk)
            current = cls._as_three_phase_time(i_slice, ik)
            if voltage.shape != current.shape:
                raise ValueError(f"Voltage/current slice dimensions disagree: {voltage.shape} vs {current.shape}.")
            signal_6ch = np.vstack([voltage, current]).astype(np.float32, copy=False)
            return signal_6ch, meta

        else:
            raise ValueError(f"Unsupported mat format: {mat_format}")

    def load_dataset_1_sample(self, filename: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads a single recording from signals (dataset 1).zip by filename.
        e.g. 'sim_Chesf_500_Classifier_LOC3_AB_20_0_1.0000.mat'
        """
        if not self.d1_zip.exists():
            raise FileNotFoundError(f"Archive not found: {self.d1_zip}")

        target_base = Path(filename).name
        with zipfile.ZipFile(self.d1_zip, 'r') as z:
            matching = [f for f in z.namelist() if Path(f).name == target_base]
            if not matching:
                raise FileNotFoundError(f"Recording {target_base} not found in {self.d1_zip.name}")
            raw_bytes = z.read(matching[0])

        mat_obj, fmt = self.load_mat_stream(raw_bytes)
        signal_6ch, meta = self.extract_6channel_waveform(mat_obj, fmt)
        file_meta = self.parse_filename_metadata(target_base)
        meta.update(file_meta)
        meta['source_file'] = target_base
        return signal_6ch, meta

    def load_dataset_2_sample(
        self,
        fault_type: str,
        angle_idx: int = 9,      # 90 deg default
        comp_idx: int = 1,       # 50% default
        rf_idx: int = 0,         # 1 ohm default
        dist_idx: int = 0        # 20 km default
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads a single scenario from signals_with_compensation (dataset 2).zip.
        """
        std_fault = RAW_TO_STANDARD_CLASS.get(fault_type, fault_type)
        fname = f"falta_{std_fault}.mat"
        with zipfile.ZipFile(self.d2_zip, 'r') as z:
            matching = [f for f in z.namelist() if Path(f).name == fname]
            if not matching:
                raise FileNotFoundError(f"Fault file {fname} not found in {self.d2_zip.name}")
            raw_bytes = z.read(matching[0])

        with h5py.File(io.BytesIO(raw_bytes), 'r') as hf:
            vk = f"V_{std_fault}"
            ik = f"I_{std_fault}"
            if hf[vk].shape != hf[ik].shape:
                raise ValueError(f"{vk}/{ik} dimensions disagree.")
            v_slice = hf[vk][angle_idx, comp_idx, rf_idx, dist_idx, :, :]
            i_slice = hf[ik][angle_idx, comp_idx, rf_idx, dist_idx, :, :]
            voltage = self._as_three_phase_time(v_slice, vk)
            current = self._as_three_phase_time(i_slice, ik)
            if voltage.shape != current.shape:
                raise ValueError("Voltage/current sample dimensions disagree.")
            signal_6ch = np.vstack([voltage, current]).astype(np.float32, copy=False)

        angles = list(range(0, 190, 10))
        comps = [0.4, 0.5, 0.6, 0.7]
        rfs = [1.0, 50.0, 100.0]
        dists = [20, 150, 280]

        meta = {
            'fault_label': std_fault,
            'sampling_rate': SAMPLING_RATE_HZ,
            'distance_km': dists[dist_idx],
            'resistance_ohm': rfs[rf_idx],
            'angle_deg': angles[angle_idx],
            'compensation': comps[comp_idx],
            'dataset': 'Dataset 2 (Series-Compensated)',
            'source_file': fname
        }
        return signal_6ch, meta

    @classmethod
    def extract_window(
        cls,
        signal_6ch: np.ndarray,
        fault_index: int,
        pre_fault: int = PRE_FAULT_SAMPLES,
        window_length: int = WINDOW_LENGTH
    ) -> np.ndarray:
        """
        Extracts a consistent window of shape (6, window_length) around fault inception.
        Pads with edge values if window exceeds boundaries.
        """
        signal_6ch = np.asarray(signal_6ch)
        if signal_6ch.ndim != 2 or signal_6ch.shape[0] != 6:
            raise ValueError("signal_6ch must have shape (6, N).")
        if signal_6ch.shape[1] == 0 or window_length <= 0 or pre_fault < 0:
            raise ValueError("Signal and window dimensions must be positive.")
        if not np.isfinite(signal_6ch).all():
            raise ValueError("signal_6ch contains NaN or infinite values.")
        start = fault_index - pre_fault
        end = start + window_length
        total_len = signal_6ch.shape[1]

        if start >= 0 and end <= total_len:
            return signal_6ch[:, start:end]

        # Handle boundary padding if needed
        pad_left = max(0, -start)
        pad_right = max(0, end - total_len)
        valid_start = max(0, start)
        valid_end = min(total_len, end)

        chunk = signal_6ch[:, valid_start:valid_end]
        if chunk.shape[1] == 0:
            raise ValueError("Requested window does not overlap the supplied signal.")
        if pad_left > 0 or pad_right > 0:
            chunk = np.pad(chunk, ((0, 0), (pad_left, pad_right)), mode='edge')
        return chunk

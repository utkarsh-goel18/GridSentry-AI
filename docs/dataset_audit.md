# GridSentry AI — Dataset Audit Report

## 1. Overview and Archive Verification

| Archive | File Size | Format | Total Scenarios | Classes | Sampling Rate | Duration |
|---|---|---|---|---|---|---|
| `signals (dataset 1).zip` | ~1.44 GB | MATLAB v5/v7 (`scipy.io`) | 940 files | 10 (94/class) | 15,360 Hz | 0.50 s (7,681 samples) |
| `signals_with_compensation (dataset 2).zip` | ~217 MB | MATLAB v7.3 (`h5py`) | 6,840 scenarios | 10 (684/class) | 15,360 Hz | 0.10 s (1,537 samples) |

## 2. Fault Classes and Portuguese-to-English Mapping

In the raw recordings, the ground phase is designated as **T** (*Terra* in Portuguese, from the Chesf 500 kV Brazilian transmission grid benchmark). This maps directly to standard IEEE/IEC power systems fault nomenclature (**G** for Ground):

| Raw Label (D1) | Standard Label (D2 / Unified) | Description | Class Count (D1) | Class Count (D2) |
|---|---|---|---|---|
| `AT` | `AG` | Single Line-to-Ground: Phase A | 94 | 684 |
| `BT` | `BG` | Single Line-to-Ground: Phase B | 94 | 684 |
| `CT` | `CG` | Single Line-to-Ground: Phase C | 94 | 684 |
| `AB` | `AB` | Line-to-Line: Phases A and B | 94 | 684 |
| `BC` | `BC` | Line-to-Line: Phases B and C | 94 | 684 |
| `CA` | `CA` | Line-to-Line: Phases C and A | 94 | 684 |
| `ABT` | `ABG` | Double Line-to-Ground: Phases A, B, Ground | 94 | 684 |
| `BCT` | `BCG` | Double Line-to-Ground: Phases B, C, Ground | 94 | 684 |
| `CAT` | `CAG` | Double Line-to-Ground: Phases C, A, Ground | 94 | 684 |
| `ABC` | `ABC` | Three-Phase Symmetrical Fault | 94 | 684 |

**Verification Confirmation**:
- Neither dataset contains separate `Normal` or `Overload` class files.
- Class distribution within each dataset is balanced (exactly 94 files per class in D1; exactly 684 scenarios per class in D2).

## 3. Physical Simulation Grid and Pairing Analysis

Both datasets were simulated using the Chesf 500 kV transmission line model with shared operating parameters:
- **Fault Distances**: 20 km (190 D1 files), 150 km (560 D1 files), 280 km (190 D1 files).
- **Inception Angles**: 0° to 180° in 10° increments (19 angles).
- **Fault Resistances**: 1.0 Ohm (570 D1 files), 50.0 Ohm (180 D1 files), 100.0 Ohm (190 D1 files).
- **Series Compensation**:
  - Dataset 1: Uncompensated line (0% compensation).
  - Dataset 2: Series-compensated line with 4 compensation factors (kc in [0.4, 0.5, 0.6, 0.7]).

**Leakage Prevention Strategy**:
Because recordings share physical parameter pairs `(distance, fault_resistance)`, data splits must partition across independent physical simulation scenarios rather than random window slices. Paired compensated and uncompensated recordings for a given scenario stay strictly within the same partition.

## 4. Signal Integrity and Dimensions

- **Missing Values**: 0 NaNs and 0 Infs detected across all inspected recordings.
- **Candidate 6-Channel Input**: Three-phase voltage ($V_a, V_b, V_c$ from `V_L`) and three-phase current ($I_a, I_b, I_c$ from `I_L`).
- **Sampling Frequency**: $f_s = 15,360$ Hz ($256$ samples per 60 Hz electrical cycle).
- **Generated Figures**:
  - `artifacts/figures/waveform_overview_all_classes.png`
  - `artifacts/figures/waveform_d1_vs_d2_comparison.png`

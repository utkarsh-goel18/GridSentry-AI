# GridSentry AI — Engineering Methodology & Run Instructions

## 1. Problem Statement Overview

**GridSentry AI** implements an intelligent transmission-line protection system for high-voltage power grids. The project addresses **Problem Statement 2: Electrical Transmission-Line Fault Classification using a 1D Convolutional Neural Network (1D CNN)**.

High-voltage electrical transmission systems are subject to transient line faults caused by lightning strikes, insulation degradation, equipment failures, and physical contacts. Rapid and accurate identification of the fault category (single line-to-ground, line-to-line, double line-to-ground, or three-phase symmetrical) within 1 to 2 electrical cycles (16.6 to 33.3 ms) is critical for digital protective relays to initiate targeted selective tripping and prevent cascading blackouts.

---

## 2. Dataset Audit & Physics Context

### 2.1 The Transmission Line Benchmark
The recordings are simulated on a **500 kV transmission line model** based on the Chesf (*Companhia Hidro Elétrica do São Francisco*, Brazil) power grid benchmark operating at **60 Hz nominal frequency** with a sampling frequency of:
$$f_s = 15,360\text{ Hz} \quad (256 \text{ samples per electrical cycle})$$

Two local archives provide complementary operating regimes:
1. **`signals (dataset 1).zip`**: Uncompensated 500 kV transmission line (940 `.mat` files in MATLAB v5/v7 format).
2. **`signals_with_compensation (dataset 2).zip`**: Series-compensated 500 kV transmission line with series capacitor banks at compensation factors $k_c \in \{0.4, 0.5, 0.6, 0.7\}$ (10 multi-dimensional `.mat` files in MATLAB v7.3 HDF5 format, totaling 6,840 scenarios).

### 2.2 Fault Taxonomy and Portuguese-to-English Ground Standardization
In the raw Chesf simulation recordings, ground faults use the letter **T** (*Terra* in Portuguese). These map to the universal IEEE/IEC power systems standard **G** (Ground):

| Raw Code (D1) | Standard Label | Power Engineering Description | Dataset 1 Count | Dataset 2 Count |
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

**Audit Findings**:
- **Class Balance**: In Dataset 1, each of the 10 classes has exactly 94 recordings (total 940). In Dataset 2, each class has exactly 684 scenarios (total 6,840).
- **Data Integrity**: Zero missing values (0 NaNs, 0 Infs) detected across both archives.
- **Normal Operation & Overload**: Neither dataset contains separate recordings labeled 'Normal' or 'Overload'. All recordings model transmission line faults, with an initial steady-state pre-fault duration ($t < 0.425$ s in D1; $t < 0.05$ s in D2) prior to fault inception.

---

## 3. Preprocessing & Leakage Prevention Strategy

### 3.1 Candidate 6-Channel Input
The model accepts three-phase instantaneous line voltages and currents:
$$\mathbf{X}(t) = \begin{bmatrix} V_a(t) & V_b(t) & V_c(t) & I_a(t) & I_b(t) & I_c(t) \end{bmatrix}^T$$
- Channels 0–2: Voltages $V_a, V_b, V_c$ (nominal steady-state peak $\approx \pm 450$ kV).
- Channels 3–5: Currents $I_a, I_b, I_c$ (nominal steady-state peak $\approx \pm 600$–$1000$ A; transient peaks $\approx 10$–$20$ kA).

### 3.2 Window Length
A consistent window length of **$L = 512$ samples** is extracted:
$$\tau = \frac{512 \text{ samples}}{15,360 \text{ Hz}} = 33.33 \text{ ms} \quad (2.0 \text{ electrical cycles at } 60\text{ Hz})$$
The window captures 64 samples ($\approx 0.25$ cycle) of pre-fault steady state and 448 samples ($\approx 1.75$ cycles) of transient fault behavior, matching the tripping window of ultra-high-speed protective relays.

### 3.3 Leakage-Free Physical Scenario Partitioning
Both datasets share a common grid of 171 physical simulation scenarios defined by:
- **Fault Distances**: 20 km, 150 km, 280 km (3 distances)
- **Fault Resistances**: 1.0 $\Omega$, 50.0 $\Omega$, 100.0 $\Omega$ (3 resistances)
- **Inception Angles**: 0° to 180° in 10° steps (19 angles)
$$3 \times 3 \times 19 = 171 \text{ physical scenarios}$$

**Partitioning Rule**:
Splits are performed strictly at the physical scenario level before generating overlapping windows:
- **Train Split**: 117 physical scenarios ($\approx 68.4\%$), yielding 15,960 window samples (with $\pm 16$-sample trigger jitter).
- **Validation Split**: 27 physical scenarios ($\approx 15.8\%$), yielding 1,230 window samples.
- **Test Split**: 27 physical scenarios ($\approx 15.8\%$), yielding 1,230 window samples.

All recordings for a given scenario—both uncompensated (Dataset 1) and series-compensated across all 4 compensation factors (Dataset 2)—are kept strictly within the same split partition.

### 3.4 Training-Only Normalization
Channel-wise Z-score normalization parameters are computed exclusively on the training partition:
$$\mu_c = \frac{1}{N_{train} \cdot L} \sum_{i, t} X_{train}[i, c, t], \quad \sigma_c = \sqrt{\frac{1}{N_{train} \cdot L} \sum_{i, t} (X_{train}[i, c, t] - \mu_c)^2}$$

Computed values:
- $V_a$: $\mu = 2.44\text{ kV}, \sigma = 273.86\text{ kV}$
- $V_b$: $\mu = -3.71\text{ kV}, \sigma = 273.74\text{ kV}$
- $V_c$: $\mu = 1.27\text{ kV}, \sigma = 275.56\text{ kV}$
- $I_a$: $\mu = 107.36\text{ A}, \sigma = 2,541.87\text{ A}$
- $I_b$: $\mu = 282.51\text{ A}, \sigma = 2,444.89\text{ A}$
- $I_c$: $\mu = -389.87\text{ A}, \sigma = 2,498.87\text{ A}$

Parameters are stored in `artifacts/scaler_params.json` and reused during inference.

---

## 4. Model Architecture & Evaluation Results

### 4.1 PyTorch 1D CNN Architecture (`GridSentry1DCNN`)
- **Input**: $(B, 6, 512)$
- **Conv1D Stage 1**: Conv1d(6 $\to$ 32, $k=7$, $s=1$, $p=3$) $\to$ BatchNorm1d $\to$ ReLU $\to$ MaxPool1d(2) $\to (B, 32, 256)$
- **Conv1D Stage 2**: Conv1d(32 $\to$ 64, $k=5$, $s=1$, $p=2$) $\to$ BatchNorm1d $\to$ ReLU $\to$ MaxPool1d(2) $\to (B, 64, 128)$
- **Conv1D Stage 3**: Conv1d(64 $\to$ 128, $k=5$, $s=1$, $p=2$) $\to$ BatchNorm1d $\to$ ReLU $\to$ MaxPool1d(2) $\to (B, 128, 64)$
- **Conv1D Stage 4**: Conv1d(128 $\to$ 128, $k=3$, $s=1$, $p=1$) $\to$ BatchNorm1d $\to$ ReLU $\to$ AdaptiveAvgPool1d(1) $\to (B, 128, 1)$
- **Classifier Head**: Flatten $\to$ Dropout(0.3) $\to$ Linear(128 $\to$ 64) $\to$ ReLU $\to$ Linear(64 $\to$ 10)
- **Total Trainable Parameters**: 111,306

### 4.2 Comparative Baseline (`BaselineMLP`)
- Flatten(3072) $\to$ Linear(3072 $\to$ 128) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(128 $\to$ 64) $\to$ Linear(64 $\to$ 10)
- **Total Trainable Parameters**: 402,506 (3.6x larger than 1D CNN)

### 4.3 Held-Out Test Split Metrics (1,230 Unseen Test Samples)

| Metric | 1D CNN (Primary) | Baseline MLP |
|---|---|---|
| **Parameters** | **111,306** | 402,506 |
| **Validation Loss** | **0.0000** | 0.0317 |
| **Test Accuracy** | **100.00%** | **100.00%** |
| **Macro-Precision** | **100.00%** | **100.00%** |
| **Macro-Recall** | **100.00%** | **100.00%** |
| **Macro-F1 Score** | **100.00%** | **100.00%** |
| **Weighted-F1 Score** | **100.00%** | **100.00%** |

#### Per-Class Performance on Held-Out Test Set (1D CNN):
| Class | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| `AG` (Phase A-Ground) | 100.00% | 100.00% | 100.00% | 123 |
| `BG` (Phase B-Ground) | 100.00% | 100.00% | 100.00% | 123 |
| `CG` (Phase C-Ground) | 100.00% | 100.00% | 100.00% | 123 |
| `AB` (Phase A-Phase B) | 100.00% | 100.00% | 100.00% | 123 |
| `BC` (Phase B-Phase C) | 100.00% | 100.00% | 100.00% | 123 |
| `CA` (Phase C-Phase A) | 100.00% | 100.00% | 100.00% | 123 |
| `ABG` (Phase AB-Ground) | 100.00% | 100.00% | 100.00% | 123 |
| `BCG` (Phase BC-Ground) | 100.00% | 100.00% | 100.00% | 123 |
| `CAG` (Phase CA-Ground) | 100.00% | 100.00% | 100.00% | 123 |
| `ABC` (Three-Phase) | 100.00% | 100.00% | 100.00% | 123 |

Figures saved:
- `artifacts/figures/confusion_matrix_1d_cnn.png`
- `artifacts/figures/training_history_1d_cnn.png`
- `artifacts/figures/waveform_overview_all_classes.png`
- `artifacts/figures/waveform_d1_vs_d2_comparison.png`

---

## 5. Execution Guide & Commands

### 5.1 Environment Setup
```powershell
# Create and activate virtual environment (optional)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 5.2 Dataset Audit
```powershell
python -m ml.audit
```
Audits both archives, verifies shapes, checks missing values, and generates waveform figures in `artifacts/figures/`.

### 5.3 Training the 1D CNN
```powershell
python -m ml.train --model 1d_cnn --epochs 15 --batch-size 64
```
Trains the primary 1D CNN, applies ReduceLROnPlateau, triggers early stopping, and saves the best model to `artifacts/best_model.pt`.

To train the comparative baseline MLP:
```powershell
python -m ml.train --model baseline --epochs 10 --batch-size 64
```

### 5.4 Evaluating Held-Out Performance
```powershell
python -m ml.evaluate --checkpoint artifacts/best_model.pt
```

### 5.5 Running the Automated Test Suite
```powershell
python -m pytest -v
```

### 5.6 Starting the FastAPI Backend
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Interactive API documentation will be accessible at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 6. Known Limitations & Next Steps

1. **Absence of Separate Normal / Overload Recordings**:
   The provided dataset files exclusively contain simulated electrical fault events. Pre-fault steady state is present in the initial cycles of each recording, but continuous normal operation and high-load power transfer (overload without fault) are not represented as separate discrete categories. Future work can incorporate non-fault load switching and capacitor energization transients.
2. **Fixed Benchmark Topology**:
   Simulations represent a single 500 kV transmission line system (Chesf). While distance (20–280 km), inception angle (0°–180°), fault resistance (1–100 $\Omega$), and series compensation (0%–70%) vary extensively, topology transfer (e.g. 230 kV lines or multi-terminal lines) should be evaluated in subsequent benchmarks.
3. **Recommended Frontend Integration**:
   - Connect the React frontend to `GET /samples` to populate an interactive fault selector dropdown.
   - Use `POST /predict` with `{ "sample_id": selectedSample }` to display the predicted fault type, confidence gauge, and 10-class probability distribution bar chart.
   - For waveform visualization, render the 3-phase voltage and 3-phase current traces side-by-side using Chart.js or Recharts.

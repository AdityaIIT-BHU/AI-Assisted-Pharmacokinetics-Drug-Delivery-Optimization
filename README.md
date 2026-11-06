# 🧬 PBPK Drug Delivery Optimizer

**4-Compartment Physiologically-Based Pharmacokinetic Model with CBF-Dreamer AI**

An AI-powered tool for optimizing oncology drug dosing schedules using a physics-informed neural network with Control Barrier Function (CBF) safety constraints.

## Architecture

### 4-Compartment PBPK Model
```
  Infusion (IV)
       │
       ▼
  ┌─────────┐    k₁₂    ┌─────────┐
  │  Plasma  │ ───────► │  Tumor   │
  │  (V₁)   │ ◄─────── │  (V₂)   │
  └─────────┘    k₂₁    └─────────┘
       │  ▲
   k₁₃│  │k₃₁
       ▼  │
  ┌─────────┐           ┌─────────┐
  │ Healthy  │    k_el   │  Elim.  │
  │ Tissue   │ ────────► │  (V₄)  │
  │  (V₃)   │           └─────────┘
  └─────────┘
```

### CBF-Dreamer Neural Network
- **Input**: Weight (kg), Height (cm), Blood Pressure (mmHg), BSA (m²)
- **Architecture**: 4→128→128→64→32→5 with GELU activation, BatchNorm, Dropout
- **Output**: Optimal dose rate, AUC tumor, AUC healthy, infusion duration, cycle interval
- **Safety**: Control Barrier Function loss (λ=10⁴) prevents dose overshooting

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (takes ~2-3 minutes)
python model.py

# 3. Launch the web app
python app.py

# 4. Open http://localhost:5000 in your browser
```

## Web App Features

| Feature | Description |
|---------|-------------|
| **Patient Input** | Weight, Height, Blood Pressure |
| **Optimal Dose Rate** | mg/hr infusion rate (CBF-constrained) |
| **Dosing Schedule** | Full cycle plan with recommended cycles |
| **Toxicity Graph** | Concentration vs Time for all compartments |
| **AUC Analysis** | Drug exposure on tumor vs healthy tissues |
| **Safety Margin** | Visual indicator of how close to toxic limit |

## Files

| File | Description |
|------|-------------|
| `model.py` | PBPK ODE model + CBF neural network + training |
| `app.py` | Flask web server with REST API |
| `static/index.html` | Frontend UI with Chart.js visualizations |
| `dataset_100000.csv` | Training dataset (12,002 patients) |
| `cbf_pbpk_model.pt` | Trained model checkpoint (generated) |

## API

### `POST /api/predict`
```json
{
  "weight_kg": 75.0,
  "height_cm": 170.0,
  "sbp_mmhg": 120.0
}
```

Returns: dosing schedule, exposure metrics, toxicity level, simulation curves.

---
⚠️ **Research prototype – not for clinical use without regulatory approval**

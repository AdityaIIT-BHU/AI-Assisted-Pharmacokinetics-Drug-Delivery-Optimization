"""
4-Compartment PBPK Model with CBF-Dreamer AI for Drug Delivery Optimization
=============================================================================

Architecture:
  - 4-compartment PBPK ODE solver (Plasma → Tumor → Healthy Tissue → Elimination)
  - Physics-Informed CBF Neural Network (Control Barrier Function safety layer)
  - Multi-output predictions: Optimal dose rate, AUC tumor, AUC healthy,
    infusion schedule, cycle timing, and full concentration-time profiles

Inputs:  Weight (kg), Height (cm), Systolic Blood Pressure (mmHg)
Outputs: Optimal dose rate, infusion schedule, cycle timing, toxicity curves, AUC
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp

# ============================================================================
# 1. FOUR-COMPARTMENT PBPK ODE MODEL
# ============================================================================

# Default PK rate constants (1/hr) - derived from literature for a generic
# cytotoxic oncology drug. These are scaled per-patient by BSA and clearance.
DEFAULT_PK_PARAMS = {
    'k_abs':   1.2,    # Absorption into plasma (for IV this is ~instant)
    'k_12':    0.45,   # Plasma → Tumor distribution
    'k_21':    0.30,   # Tumor → Plasma redistribution
    'k_13':    0.35,   # Plasma → Healthy Tissue distribution
    'k_31':    0.25,   # Healthy Tissue → Plasma redistribution
    'k_el':    0.18,   # Plasma → Elimination (renal/hepatic clearance)
    'V_plasma':  5.0,  # Volume of distribution - plasma (L)
    'V_tumor':   2.0,  # Volume of distribution - tumor (L)
    'V_healthy': 15.0, # Volume of distribution - healthy tissue (L)
    'V_elim':    3.0,  # Volume of distribution - elimination compartment (L)
}


def patient_pk_params(weight_kg, height_cm, sbp_mmhg):
    """
    Scale population PK parameters for an individual patient.

    Body Surface Area (BSA) by Mosteller formula.
    Clearance scaling by allometric weight^0.75.
    Blood-pressure factor modulates tissue perfusion rates.
    """
    bsa = np.sqrt((height_cm * weight_kg) / 3600.0)

    # Allometric clearance scaling (normalised to 70 kg reference)
    cl_scale = (weight_kg / 70.0) ** 0.75

    # Blood pressure perfusion factor (normalised to 120 mmHg reference)
    bp_factor = (sbp_mmhg / 120.0) ** 0.3

    params = dict(DEFAULT_PK_PARAMS)
    params['k_el'] *= cl_scale
    params['k_12'] *= bsa * bp_factor
    params['k_21'] *= bsa
    params['k_13'] *= bsa * bp_factor
    params['k_31'] *= bsa
    # Volume scaling by weight
    params['V_plasma']  *= (weight_kg / 70.0)
    params['V_tumor']   *= bsa
    params['V_healthy'] *= (weight_kg / 70.0)
    return params, bsa


def pbpk_odes(t, y, dose_rate, params):
    """
    4-compartment ODE system.

    Compartments:
      y[0] = A_plasma   (mg)  – central / plasma
      y[1] = A_tumor    (mg)  – tumor tissue
      y[2] = A_healthy  (mg)  – healthy / peripheral tissue
      y[3] = A_elim     (mg)  – elimination sink

    dose_rate is continuous IV infusion (mg/hr).
    """
    Ap, At, Ah, Ae = y
    k12, k21 = params['k_12'], params['k_21']
    k13, k31 = params['k_13'], params['k_31']
    k_el      = params['k_el']

    dAp = dose_rate - (k12 + k13 + k_el) * Ap + k21 * At + k31 * Ah
    dAt = k12 * Ap - k21 * At
    dAh = k13 * Ap - k31 * Ah
    dAe = k_el * Ap

    return [dAp, dAt, dAh, dAe]


def simulate_pbpk(dose_rate, params, t_span=(0, 72), n_points=500):
    """
    Run a single-cycle PBPK simulation and return time-course data.

    Returns dict with arrays: t, C_plasma, C_tumor, C_healthy,
    auc_tumor, auc_healthy, max_healthy_conc, max_tumor_conc.
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        pbpk_odes, t_span, y0=[0, 0, 0, 0],
        args=(dose_rate, params),
        t_eval=t_eval, method='RK45', max_step=0.1
    )

    C_plasma  = sol.y[0] / params['V_plasma']
    C_tumor   = sol.y[1] / params['V_tumor']
    C_healthy = sol.y[2] / params['V_healthy']

    auc_tumor   = float(np.trapz(C_tumor,   sol.t))
    auc_healthy = float(np.trapz(C_healthy, sol.t))

    return {
        't':               sol.t.tolist(),
        'C_plasma':        C_plasma.tolist(),
        'C_tumor':         C_tumor.tolist(),
        'C_healthy':       C_healthy.tolist(),
        'auc_tumor':       auc_tumor,
        'auc_healthy':     auc_healthy,
        'max_healthy_conc': float(np.max(C_healthy)),
        'max_tumor_conc':  float(np.max(C_tumor)),
    }


# ============================================================================
# 2. CBF-DREAMER NEURAL NETWORK
# ============================================================================

class FourCompartmentCBFNet(nn.Module):
    """
    Physics-Informed Control-Barrier-Function Neural Network.

    Inputs  (4): weight_kg, height_cm, sbp_mmhg, bsa
    Outputs (5): optimal_dose_rate, auc_tumor, auc_healthy,
                 infusion_duration_hr, cycle_interval_hr
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.head = nn.Linear(32, 5)

    def forward(self, x):
        features = self.net(x)
        return self.head(features)


def cbf_safety_loss(predictions, targets, lambda_barrier=1e4, safe_healthy_limit=2.5):
    """
    Composite loss = MSE + CBF barrier penalty.

    The barrier enforces:
      • Predicted dose_rate ≤ safe target dose_rate  (prevents overdosing)
      • Predicted AUC_healthy stays bounded           (limits toxicity exposure)

    The penalty is quadratic with a large multiplier (λ = 10⁴) so that the
    network is *heavily* discouraged from crossing the safety boundary,
    implementing the CBF invariance condition ḣ(x) + α·h(x) ≥ 0 in loss-space.
    """
    mse = nn.MSELoss()(predictions, targets)

    # Dose over-prediction penalty
    dose_violation = torch.relu(predictions[:, 0] - targets[:, 0])
    dose_penalty   = torch.mean(dose_violation ** 2) * lambda_barrier

    # AUC healthy tissue over-exposure penalty
    auc_healthy_pred = predictions[:, 2]
    auc_healthy_tgt  = targets[:, 2]
    auc_violation    = torch.relu(auc_healthy_pred - auc_healthy_tgt * 1.05)
    auc_penalty      = torch.mean(auc_violation ** 2) * (lambda_barrier * 0.5)

    return mse + dose_penalty + auc_penalty


# ============================================================================
# 3. TRAINING
# ============================================================================

def prepare_dataset(csv_path='dataset_100000.csv'):
    """Load CSV, engineer features and targets."""
    df = pd.read_csv(csv_path)

    # Body Surface Area (Mosteller)
    df['bsa'] = np.sqrt((df['height_cm'] * df['weight_kg']) / 3600.0)

    # Synthetic systolic blood pressure (correlated with weight, with noise)
    # The dataset doesn't have BP, so we generate a physiologically plausible
    # surrogate:  SBP ≈ 100 + 0.3·weight + noise
    rng = np.random.RandomState(42)
    df['sbp_mmhg'] = 100.0 + 0.3 * df['weight_kg'] + rng.normal(0, 5, len(df))
    df['sbp_mmhg'] = df['sbp_mmhg'].clip(90, 180)

    # --- Target engineering ---
    # Safe optimal dose: scale so max_healthy_conc hits exactly 2.5 mg/L
    SAFE_HEALTHY_LIMIT = 2.5
    df['optimal_dose_rate'] = SAFE_HEALTHY_LIMIT * (
        df['dose_rate_mg_hr'] / df['max_healthy_conc']
    )
    df['optimal_auc_tumor'] = df['auc_tumor'] * (
        df['optimal_dose_rate'] / df['dose_rate_mg_hr']
    )
    df['optimal_auc_healthy'] = df['auc_healthy'] * (
        df['optimal_dose_rate'] / df['dose_rate_mg_hr']
    )

    # Infusion duration (hr): heuristic – lower dose → longer infusion
    df['infusion_duration_hr'] = (
        2.0 + (100.0 - df['optimal_dose_rate'].clip(upper=100)) * 0.04
        + df['bsa'] * 0.5
    )
    df['infusion_duration_hr'] = df['infusion_duration_hr'].clip(1.0, 8.0)

    # Cycle interval (days → hours): faster clearance (higher k_el proxy) → shorter cycle
    # We use weight as a proxy for clearance scaling
    cl_proxy = (df['weight_kg'] / 70.0) ** 0.75
    df['cycle_interval_hr'] = (21.0 * 24.0) / cl_proxy  # ~21-day base cycle
    df['cycle_interval_hr'] = df['cycle_interval_hr'].clip(14*24, 28*24)

    features = df[['weight_kg', 'height_cm', 'sbp_mmhg', 'bsa']].values
    targets  = df[[
        'optimal_dose_rate',
        'optimal_auc_tumor',
        'optimal_auc_healthy',
        'infusion_duration_hr',
        'cycle_interval_hr',
    ]].values

    return features, targets, df


def train_model(csv_path='dataset_100000.csv', epochs=300, lr=0.003,
                save_path='cbf_pbpk_model.pt'):
    """Full training loop with CBF safety loss."""
    print("=" * 60)
    print("  4-Compartment PBPK + CBF-Dreamer Model Training")
    print("=" * 60)

    X, Y, df = prepare_dataset(csv_path)

    # Scalers
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=0.15, random_state=42
    )

    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_test_t  = torch.FloatTensor(X_test)
    Y_test_t  = torch.FloatTensor(Y_test)

    model     = FourCompartmentCBFNet()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train_t)
        loss  = cbf_safety_loss(preds, Y_train_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test_t)
                val_loss  = nn.MSELoss()(val_preds, Y_test_t).item()
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:>4d}/{epochs}  "
                  f"Train CBF Loss: {loss.item():.6f}  "
                  f"Val MSE: {val_loss:.6f}  "
                  f"LR: {lr_now:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler_X_mean':    scaler_X.mean_.tolist(),
                    'scaler_X_scale':   scaler_X.scale_.tolist(),
                    'scaler_Y_mean':    scaler_Y.mean_.tolist(),
                    'scaler_Y_scale':   scaler_Y.scale_.tolist(),
                }, save_path)

    print(f"\n[OK] Training complete. Best validation MSE: {best_loss:.6f}")
    print(f"[OK] Model checkpoint saved to: {save_path}")
    return model, scaler_X, scaler_Y


# ============================================================================
# 4. INFERENCE
# ============================================================================

def load_model(checkpoint_path='cbf_pbpk_model.pt'):
    """Load trained model and scalers from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    model = FourCompartmentCBFNet()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    scaler_X = StandardScaler()
    scaler_X.mean_  = np.array(ckpt['scaler_X_mean'])
    scaler_X.scale_ = np.array(ckpt['scaler_X_scale'])
    scaler_X.var_   = scaler_X.scale_ ** 2
    scaler_X.n_features_in_ = len(scaler_X.mean_)

    scaler_Y = StandardScaler()
    scaler_Y.mean_  = np.array(ckpt['scaler_Y_mean'])
    scaler_Y.scale_ = np.array(ckpt['scaler_Y_scale'])
    scaler_Y.var_   = scaler_Y.scale_ ** 2
    scaler_Y.n_features_in_ = len(scaler_Y.mean_)

    return model, scaler_X, scaler_Y


def predict_optimized_dosing(model, scaler_X, scaler_Y,
                              weight_kg, height_cm, sbp_mmhg):
    """
    Given patient parameters, predict full optimized dosing schedule
    and run PBPK simulation with the predicted optimal dose.
    """
    bsa = np.sqrt((height_cm * weight_kg) / 3600.0)

    # ── Neural network prediction ──
    x_raw = np.array([[weight_kg, height_cm, sbp_mmhg, bsa]])
    x_sc  = scaler_X.transform(x_raw)
    x_t   = torch.FloatTensor(x_sc)

    with torch.no_grad():
        y_sc = model(x_t).numpy()

    y_raw = scaler_Y.inverse_transform(y_sc)[0]

    optimal_dose_rate     = float(max(y_raw[0], 0.5))
    predicted_auc_tumor   = float(max(y_raw[1], 0.0))
    predicted_auc_healthy = float(max(y_raw[2], 0.0))
    infusion_duration_hr  = float(np.clip(y_raw[3], 1.0, 8.0))
    cycle_interval_hr     = float(np.clip(y_raw[4], 14*24, 28*24))

    # ── PBPK simulation with predicted optimal dose ──
    pk_params, _ = patient_pk_params(weight_kg, height_cm, sbp_mmhg)
    sim = simulate_pbpk(optimal_dose_rate, pk_params, t_span=(0, 72), n_points=500)

    # ── Safety check: if simulated max healthy > 2.5, scale down ──
    SAFE_LIMIT = 2.5
    if sim['max_healthy_conc'] > SAFE_LIMIT:
        scale = SAFE_LIMIT / sim['max_healthy_conc'] * 0.95
        optimal_dose_rate *= scale
        sim = simulate_pbpk(optimal_dose_rate, pk_params,
                            t_span=(0, 72), n_points=500)

    # Toxicity level classification
    max_h = sim['max_healthy_conc']
    if max_h < 1.5:
        toxicity_level = "Low"
    elif max_h < 2.0:
        toxicity_level = "Moderate"
    elif max_h < 2.5:
        toxicity_level = "High (within safe limit)"
    else:
        toxicity_level = "CRITICAL – exceeds safe limit"

    cycle_days = cycle_interval_hr / 24.0

    result = {
        'patient': {
            'weight_kg':  weight_kg,
            'height_cm':  height_cm,
            'sbp_mmhg':   sbp_mmhg,
            'bsa':        round(bsa, 4),
        },
        'dosing_schedule': {
            'optimal_dose_rate_mg_hr':   round(optimal_dose_rate, 2),
            'total_dose_per_cycle_mg':   round(optimal_dose_rate * infusion_duration_hr, 2),
            'infusion_duration_hr':      round(infusion_duration_hr, 2),
            'cycle_interval_days':       round(cycle_days, 1),
            'cycles_recommended':        max(1, int(round(180 / cycle_days))),  # ~6-month plan
        },
        'exposure': {
            'auc_tumor':           round(sim['auc_tumor'], 4),
            'auc_healthy':         round(sim['auc_healthy'], 4),
            'auc_ratio':           round(sim['auc_tumor'] / max(sim['auc_healthy'], 0.001), 4),
            'max_tumor_conc':      round(sim['max_tumor_conc'], 4),
            'max_healthy_conc':    round(sim['max_healthy_conc'], 4),
        },
        'toxicity': {
            'level':               toxicity_level,
            'max_healthy_conc':    round(max_h, 4),
            'safe_limit':          SAFE_LIMIT,
            'margin':              round(SAFE_LIMIT - max_h, 4),
        },
        'simulation': {
            't':          sim['t'],
            'C_plasma':   sim['C_plasma'],
            'C_tumor':    sim['C_tumor'],
            'C_healthy':  sim['C_healthy'],
        }
    }

    return result


# ============================================================================
# 5. MAIN – Train if run directly
# ============================================================================

if __name__ == '__main__':
    import sys

    csv_path = 'dataset_100000.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at '{csv_path}'")
        sys.exit(1)

    model, scaler_X, scaler_Y = train_model(csv_path, epochs=300)

    # Quick sanity-check inference
    print("\n" + "=" * 60)
    print("  Sanity-Check Inference")
    print("=" * 60)
    result = predict_optimized_dosing(
        model, scaler_X, scaler_Y,
        weight_kg=75.0, height_cm=170.0, sbp_mmhg=120.0
    )
    print(f"  Optimal dose rate : {result['dosing_schedule']['optimal_dose_rate_mg_hr']} mg/hr")
    print(f"  Infusion duration : {result['dosing_schedule']['infusion_duration_hr']} hr")
    print(f"  Cycle interval    : {result['dosing_schedule']['cycle_interval_days']} days")
    print(f"  AUC tumor         : {result['exposure']['auc_tumor']}")
    print(f"  AUC healthy       : {result['exposure']['auc_healthy']}")
    print(f"  Toxicity level    : {result['toxicity']['level']}")
    print(f"  Safety margin     : {result['toxicity']['margin']} mg/L")

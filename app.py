"""
Flask Web Application for PBPK Drug Delivery Optimizer
======================================================

REST API + Static HTML/JS frontend.
Endpoints:
  GET  /              → Web UI
  POST /api/predict   → Run AI prediction + PBPK simulation
  GET  /api/health    → Health check
"""

import os
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model import load_model, predict_optimized_dosing, train_model

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

MODEL_PATH = 'cbf_pbpk_model.pt'

# Global model references – loaded once at startup
_model = None
_scaler_X = None
_scaler_Y = None


def get_model():
    """Lazy-load the model (train if checkpoint missing)."""
    global _model, _scaler_X, _scaler_Y
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            print("[!] No checkpoint found - training model from scratch...")
            _model, _scaler_X, _scaler_Y = train_model(
                'dataset_100000.csv', epochs=300, save_path=MODEL_PATH
            )
        else:
            print("[OK] Loading pre-trained model...")
            _model, _scaler_X, _scaler_Y = load_model(MODEL_PATH)
    return _model, _scaler_X, _scaler_Y


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': _model is not None})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Expects JSON body:
    {
        "weight_kg": 75.0,
        "height_cm": 170.0,
        "sbp_mmhg": 120.0
    }
    """
    try:
        data = request.get_json(force=True)

        weight = float(data.get('weight_kg', 0))
        height = float(data.get('height_cm', 0))
        sbp    = float(data.get('sbp_mmhg', 120))

        # Validation
        errors = []
        if not (20 <= weight <= 200):
            errors.append('Weight must be between 20 and 200 kg')
        if not (100 <= height <= 250):
            errors.append('Height must be between 100 and 250 cm')
        if not (60 <= sbp <= 220):
            errors.append('Systolic BP must be between 60 and 220 mmHg')

        if errors:
            return jsonify({'error': '; '.join(errors)}), 400

        model, scaler_X, scaler_Y = get_model()
        result = predict_optimized_dosing(
            model, scaler_X, scaler_Y,
            weight_kg=weight, height_cm=height, sbp_mmhg=sbp
        )

        # Downsample simulation arrays for JSON response (every 5th point)
        sim = result['simulation']
        step = 5
        result['simulation'] = {
            't':         sim['t'][::step],
            'C_plasma':  sim['C_plasma'][::step],
            'C_tumor':   sim['C_tumor'][::step],
            'C_healthy': sim['C_healthy'][::step],
        }

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Startup ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Pre-load model at startup
    get_model()
    print("\n[SERVER] PBPK Drug Delivery Optimizer is running!")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)

import os
import sys
import torch
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root (e.g., mpp-spandit-npz/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.avit import build_avit
from utils.YParams import YParams

NPZ_DIR = '/lustre/scratch5/exempt/artimis/data/lsc240420/'

SELECTED_FIELDS = [
    'Uvelocity', 'Wvelocity', 'density_case', 'density_cushion', 'density_maincharge', 'density_outerair', 'density_striker', 'density_throw'
]
# CLI Arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Run MPP prediction for a given timestep")
parser.add_argument('--predict_timestep', type=int, required=True, help='Timestep to predict (e.g., 94)')
parser.add_argument('--n_steps', type=int, required=True, help='Number of prior timesteps to use (e.g., 2)')
parser.add_argument('--dry_run', action='store_true', help='Only validate file existence; do not run inference')
args = parser.parse_args()

PREDICT_IDX = args.predict_timestep
N_STEPS = args.n_steps
DRY_RUN = args.dry_run

# -------------------------------
# Dynamic Paths
# ------------------------------
CONFIG_PATH = f'config/mpp_lsc_avit_ti_config_nsteps_{N_STEPS}.yaml'
CKPT_PATH = f'/users/spandit/proj/runs/mpp/basic_config/lsc240420_nsteps_{N_STEPS}/training_checkpoints/best_ckpt.tar'
LOG_PATH = "prediction_rmse_log_nsteps{N_STEPS}.csv"

# -------------------------------

# -------------------------------
# Check file existence
# -------------------------------
def ensure_file_exists(path, label="file"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Required {label} not found: {path}")
    else:
        print (f"✅Required {label} found: {path}")

def main():
    # Config & checkpoint
    ensure_file_exists(CONFIG_PATH, "config")
    ensure_file_exists(CKPT_PATH, "checkpoint")

    print(f"Here 1111", flush=True)
    print(f"Here 1112", flush=True)

    # Input .npz files
    input_tensors = []
    for offset in range(PREDICT_IDX - N_STEPS, PREDICT_IDX):
        fname = f"lsc240420_id05300_pvi_idx{offset:05d}.npz"
        fpath = os.path.join(NPZ_DIR, fname)
        print(f"📥 Validating input: {fname}")
        ensure_file_exists(fpath, "input .npz")
        if not DRY_RUN:
            with np.load(fpath) as data:
                arrays = [np.nan_to_num(data[key].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                          for key in SELECTED_FIELDS]
                tensor = torch.tensor(np.stack(arrays, axis=0), dtype=torch.float32)
                input_tensors.append(tensor)

    # Ground truth
    target_fname = f"lsc240420_id05300_pvi_idx{PREDICT_IDX:05d}.npz"
    target_path = os.path.join(NPZ_DIR, target_fname)
    print(f"📥 Validating ground truth: {target_fname}")
    ensure_file_exists(target_path, "ground truth .npz")

    if DRY_RUN:
        print("✅ DRY RUN complete. All files exist.")
        exit(0)

    # -------------------------------
    # Load Model
    # -------------------------------
    params = YParams(CONFIG_PATH, 'basic_config')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_avit(params).to(device)
    model.eval()

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"✅ Model loaded from: {CKPT_PATH}")

    # -------------------------------
    # Prepare inputs
    # -------------------------------
    x = torch.stack(input_tensors, dim=0).unsqueeze(1).to(device)
    labels = torch.arange(x.shape[2]).unsqueeze(0).to(device)
    bcs = torch.zeros(1, 2).to(device)

    # Load ground truth
    with np.load(target_path) as data:
        arrays = [np.nan_to_num(data[key].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                  for key in SELECTED_FIELDS]
        y_true = torch.tensor(np.stack(arrays, axis=0), dtype=torch.float32).to(device)

    # -------------------------------
    # Run prediction
    # -------------------------------
    with torch.no_grad():
        y_pred = model(x, labels, bcs)[0]

    # -------------------------------
    # Save .npz output
    # -------------------------------
    output_fname = f"lsc240420_id05300_pvi_idx{PREDICT_IDX:05d}_PRED.npz"
    output_path = os.path.join(NPZ_DIR, output_fname)
    np.savez(output_path, **{key: y_pred[i].cpu().numpy() for i, key in enumerate(SELECTED_FIELDS)})
    print(f"✅ Saved prediction to: {output_path}")

    # -------------------------------
    # Compute and log RMSE
    # -------------------------------
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    print(f"📊 RMSE vs ground truth: {rmse:.4f}")

    # Append to log CSV
    log_entry = pd.DataFrame([{
        'predict_timestep': PREDICT_IDX,
        'n_steps': N_STEPS,
        'rmse': rmse,
        'checkpoint': CKPT_PATH,
        'config': CONFIG_PATH,
        'output_file': output_path
    }])

    if os.path.exists(LOG_PATH):
        log_entry.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(LOG_PATH, index=False)

    print(f"📝 Logged RMSE to {LOG_PATH}")

    # -------------------------------
    # Plot GT vs Prediction
    # -------------------------------
    fig, axes = plt.subplots(3, 8, figsize=(24, 9))
    for i in range(len(SELECTED_FIELDS)):
        axes[0, i].imshow(y_true[i].cpu(), cmap='viridis')
        axes[0, i].set_title(SELECTED_FIELDS[i], fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(y_pred[i].cpu(), cmap='viridis')
        axes[1, i].axis('off')

        diff = torch.abs(y_pred[i] - y_true[i])
        axes[2, i].imshow(diff.cpu(), cmap='hot')
        axes[2, i].axis('off')

    row_labels = ["Ground Truth", "Predicted", "Absolute Error"]
    for row_idx, label in enumerate(row_labels):
        fig.text(0.01, 0.92 - 0.31 * row_idx, label, va='top', ha='left', fontsize=14, rotation=90)

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plot_path = f"lsc240420_id05300_pvi_idx{PREDICT_IDX:05d}_nsteps{N_STEPS}_PRED_vs_GT.png"
    plt.savefig(plot_path, dpi=300)
    print(f"📷 Saved visualization to: {plot_path}")


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
import argparse
from utils.YParams import YParams
from models.avit import build_avit
import matplotlib.pyplot as plt

# -------------------------------
# Command-line arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Run MPP prediction for a given timestep")
parser.add_argument('--predict_timestep', type=int, required=True, help='Timestep to predict (e.g., 94)')
parser.add_argument('--n_steps', type=int, required=True, help='Number of prior timesteps to use (e.g., 2)')
args = parser.parse_args()

PREDICT_IDX = args.predict_timestep
N_STEPS = args.n_steps

# -------------------------------
# Dynamic paths
# -------------------------------
CONFIG_PATH = f'config/mpp_lsc_avit_ti_config_nsteps_{N_STEPS}.yaml'
CKPT_PATH = f'/users/spandit/proj/runs/mpp/basic_config/lsc240420_nsteps_{N_STEPS}/best_ckpt.tar'
NPZ_DIR = '/lustre/scratch5/exempt/artimis/data/lsc240420/'

# -------------------------------
# File existence checks
# -------------------------------
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"❌ Checkpoint file not found: {CKPT_PATH}")

# -------------------------------
# Model & data field definitions
# -------------------------------
CONFIG_NAME = 'basic_config'
SELECTED_FIELDS = [
    'pressure_throw', 'density_throw', 'temperature_throw',
    'density_case', 'pressure_case', 'temperature_case',
    'Uvelocity', 'Wvelocity'
]

# -------------------------------
# Load model
# -------------------------------
params = YParams(CONFIG_PATH, CONFIG_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_avit(params).to(device)
model.eval()

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt['model_state'])
print(f"✅ Model loaded from: {CKPT_PATH}")

# -------------------------------
# Helper to load .npz file
# -------------------------------
def load_npz_tensor(fpath):
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"❌ Required file not found: {fpath}")
    with np.load(fpath) as data:
        arrays = []
        for key in SELECTED_FIELDS:
            arr = data[key].astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            assert arr.ndim == 2, f"{key} in {fpath} is not 2D"
            arrays.append(arr)
        return torch.tensor(np.stack(arrays, axis=0), dtype=torch.float32)

# -------------------------------
# Load input timesteps
# -------------------------------
input_tensors = []
for offset in range(PREDICT_IDX - N_STEPS, PREDICT_IDX):
    fname = f"lsc240420_id05300_pvi_idx{offset:05d}.npz"
    fpath = os.path.join(NPZ_DIR, fname)
    print(f"📥 Loading input: {fname}")
    input_tensors.append(load_npz_tensor(fpath))

x = torch.stack(input_tensors, dim=0).unsqueeze(1).to(device)  # [T, 1, C, H, W]

# -------------------------------
# Load ground truth
# -------------------------------
target_fname = f"lsc240420_id05300_pvi_idx{PREDICT_IDX:05d}.npz"
target_path = os.path.join(NPZ_DIR, target_fname)
print(f"📥 Loading ground truth: {target_fname}")
y_true = load_npz_tensor(target_path).to(device)  # [C, H, W]

# -------------------------------
# Prepare inputs and run prediction
# -------------------------------
labels = torch.arange(x.shape[2]).unsqueeze(0).to(device)  # [1, C]
bcs = torch.zeros(1, 2).to(device)

with torch.no_grad():
    y_pred = model(x, labels, bcs)[0]  # remove batch dim → [C, H, W]

# -------------------------------
# Save predicted output
# -------------------------------
output_fname = f"lsc240420_id05300_pvi_idx{PREDICT_IDX:05d}_PRED.npz"
output_path = os.path.join(NPZ_DIR, output_fname)
np.savez(output_path, **{key: y_pred[i].cpu().numpy() for i, key in enumerate(SELECTED_FIELDS)})
print(f"✅ Saved prediction to: {output_path}")

# -------------------------------
# Compute RMSE
# -------------------------------
rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
print(f"📊 RMSE vs ground truth: {rmse:.4f}")

# -------------------------------
# Plot GT, prediction, and error
# -------------------------------
fig, axes = plt.subplots(3, 8, figsize=(24, 9))

for i in range(len(SELECTED_FIELDS)):
    axes[0, i].imshow(y_true[i].cpu(), cmap='viridis')
    axes[0, i].set_title(SELECTED_FIELDS[i], fontsize=10)
    axes[0, i].axis('off')

    axes[1, i].imshow(y_pred[i].cpu(), cmap='viridis')
    axes[1, i].axis('off')

    error = torch.abs(y_pred[i] - y_true[i])
    axes[2, i].imshow(error.cpu(), cmap='hot')
    axes[2, i].axis('off')

row_labels = ["Ground Truth", "Predicted", "Absolute Error"]
for row_idx, label in enumerate(row_labels):
    fig.text(0.01, 0.92 - 0.31 * row_idx, label, va='top', ha='left', fontsize=14, rotation=90)

plt.tight_layout(rect=[0.05, 0, 1, 1])
plot_path = f"lsc240420_id05300_pvi_idx{PREDICT_IDX:05d}_PRED_vs_GT.png"
plt.savefig(plot_path, dpi=300)
print(f"📷 Saved visualization to: {plot_path}")


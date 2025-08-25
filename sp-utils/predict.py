import os
import torch
import numpy as np
from utils.YParams import YParams
from models.avit import build_avit
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
CONFIG_PATH = 'config/mpp_lsc_avit_ti_config.yaml'
CONFIG_NAME = 'basic_config'
NPZ_DIR = '/lustre/scratch5/exempt/artimis/data/lsc240420/'  # <-- Update path if needed
CKPT_PATH = '/users/spandit/proj/runs/mpp/basic_config/lsc240420_scratch5__2025_08_11__12_46_29/best_ckpt.tar'
TARGET_IDX = 94
N_STEPS = 2

SELECTED_FIELDS = [
    'pressure_throw', 'density_throw', 'temperature_throw',
    'density_case', 'pressure_case', 'temperature_case',
    'Uvelocity', 'Wvelocity'
]

# -------------------------------
# Load config and model
# -------------------------------
params = YParams(CONFIG_PATH, CONFIG_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_avit(params).to(device)
model.eval()

ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt['model_state'])
print("✅ Model loaded.")

# -------------------------------
# Helper to load .npz file
# -------------------------------
def load_npz_tensor(fpath):
    with np.load(fpath) as data:
        arrays = []
        for key in SELECTED_FIELDS:
            arr = data[key].astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            assert arr.ndim == 2, f"{key} in {fpath} is not 2D"
            arrays.append(arr)
        stacked = np.stack(arrays, axis=0)  # [C, H, W]
        return torch.tensor(stacked, dtype=torch.float32)

# -------------------------------
# Load input and target
# -------------------------------
input_tensors = []
for offset in range(TARGET_IDX - N_STEPS, TARGET_IDX):
    fname = f"lsc240420_id05300_pvi_idx{offset:05d}.npz"
    fpath = os.path.join(NPZ_DIR, fname)
    print(f"Loading input: {fname}")
    input_tensors.append(load_npz_tensor(fpath))

x = torch.stack(input_tensors, dim=0).unsqueeze(1).to(device)  # [T, 1, C, H, W]

# Load ground truth
target_fname = f"lsc240420_id05300_pvi_idx{TARGET_IDX:05d}.npz"
y_true = load_npz_tensor(os.path.join(NPZ_DIR, target_fname)).to(device)  # [C, H, W]

# Prepare labels and bcs
labels = torch.arange(x.shape[2]).unsqueeze(0).to(device)  # [1, C]
bcs = torch.zeros(1, 2).to(device)

# -------------------------------
# Run prediction
# -------------------------------
with torch.no_grad():
    y_pred = model(x, labels, bcs)  # [B, C, H, W]

y_pred = y_pred[0]  # Remove batch dimension → [C, H, W]

# -------------------------------
# Save prediction as .npz
# -------------------------------
output_fname = f"lsc240420_id05300_pvi_idx{TARGET_IDX:05d}_PRED.npz"
output_path = os.path.join(NPZ_DIR, output_fname)
pred_dict = {key: y_pred[i].cpu().numpy() for i, key in enumerate(SELECTED_FIELDS)}
np.savez(output_path, **pred_dict)
print(f"✅ Saved prediction to {output_path}")

# -------------------------------
# Compute and print RMSE
# -------------------------------
rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
print(f"📊 RMSE vs ground truth: {rmse:.4f}")

# -------------------------------
# Plot side-by-side GT vs. PRED per channel
# -------------------------------
fig, axes = plt.subplots(3, 8, figsize=(24, 9))  # 3 rows: GT / Pred / Error

fig, axes = plt.subplots(3, 8, figsize=(24, 9))  # 3 rows x 8 fields

for i in range(len(SELECTED_FIELDS)):
    # Ground Truth
    axes[0, i].imshow(y_true[i].cpu(), cmap='viridis')
    axes[0, i].set_title(SELECTED_FIELDS[i], fontsize=10)
    axes[0, i].axis('off')

    # Prediction
    axes[1, i].imshow(y_pred[i].cpu(), cmap='viridis')
    axes[1, i].axis('off')

    # Absolute Error
    diff = torch.abs(y_pred[i] - y_true[i])
    axes[2, i].imshow(diff.cpu(), cmap='hot')
    axes[2, i].axis('off')

# Add global row labels (outside the subplots)
row_labels = ["Ground Truth", "Predicted", "Absolute Error"]
for row_idx, label in enumerate(row_labels):
    # y = from top of figure: 0.92, 0.61, 0.30 (spacing tuned for 3 rows)
    fig.text(0.01, 0.92 - 0.31 * row_idx, label, va='top', ha='left', fontsize=14, rotation=90)

# Adjust layout to leave space on the left for row labels
plt.tight_layout(rect=[0.05, 0, 1, 1])

# Save the figure
save_name = f"lsc240420_id05300_pvi_idx{TARGET_IDX:05d}_PRED_vs_GT.png"
plt.savefig(save_name, dpi=300)
print(f"📷 Saved visualization to {save_name}")
#plt.show()

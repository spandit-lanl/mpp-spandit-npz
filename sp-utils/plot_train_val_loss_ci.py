import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Load and clean CSV ===
csv_file = "out_train_val_loss.csv"
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()
df.columns = ['train', 'val']
df['Epoch'] = df.index + 1

# === Rolling statistics for confidence intervals ===
window = 10
df['train_mean'] = df['train'].rolling(window, min_periods=1).mean()
df['train_std'] = df['train'].rolling(window, min_periods=1).std()
df['val_mean'] = df['val'].rolling(window, min_periods=1).mean()
df['val_std'] = df['val'].rolling(window, min_periods=1).std()

# === Min loss markers ===
train_min = df['train_mean'].min()
train_min_epoch = df['train_mean'].idxmin()
val_min = df['val_mean'].min()
val_min_epoch = df['val_mean'].idxmin()

# === Early stopping threshold (95% CI upper bound at final epoch) ===
early_stop_threshold = df['val_mean'].iloc[-1] + 1.96 * df['val_std'].iloc[-1]

# === Zoomed-in view: last 100 epochs ===
zoom_df = df[-100:]

# === Create stacked subplots ===
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

# -------- Top: Full view --------
axes[0].plot(df['Epoch'], df['train_mean'], color='red', label='Train Loss')
axes[0].fill_between(df['Epoch'],
                     df['train_mean'] - 1.96 * df['train_std'],
                     df['train_mean'] + 1.96 * df['train_std'],
                     color='red', alpha=0.3)

axes[0].plot(df['Epoch'], df['val_mean'], color='blue', label='Validation Loss')
axes[0].fill_between(df['Epoch'],
                     df['val_mean'] - 1.96 * df['val_std'],
                     df['val_mean'] + 1.96 * df['val_std'],
                     color='blue', alpha=0.3)

axes[0].scatter(df['Epoch'][train_min_epoch], train_min, color='darkred', marker='o',
                label=f'Train Min (Epoch {train_min_epoch})')
axes[0].scatter(df['Epoch'][val_min_epoch], val_min, color='darkblue', marker='o',
                label=f'Val Min (Epoch {val_min_epoch})')

axes[0].axhline(y=early_stop_threshold, color='gray', linestyle='--',
                label=f'Early Stop Threshold ≈ {early_stop_threshold:.4f}')

axes[0].set_title("Training and Validation Loss Over 500 Epochs")
axes[0].set_ylabel("RMSE")
axes[0].legend()
axes[0].grid(True)

# -------- Bottom: Zoomed-in view --------
axes[1].plot(zoom_df['Epoch'], zoom_df['train_mean'], color='red', label='Train Loss')
axes[1].fill_between(zoom_df['Epoch'],
                     zoom_df['train_mean'] - 1.96 * zoom_df['train_std'],
                     zoom_df['train_mean'] + 1.96 * zoom_df['train_std'],
                     color='red', alpha=0.3)

axes[1].plot(zoom_df['Epoch'], zoom_df['val_mean'], color='blue', label='Validation Loss')
axes[1].fill_between(zoom_df['Epoch'],
                     zoom_df['val_mean'] - 1.96 * zoom_df['val_std'],
                     zoom_df['val_mean'] + 1.96 * zoom_df['val_std'],
                     color='blue', alpha=0.3)

axes[1].axhline(y=early_stop_threshold, color='gray', linestyle='--',
                label=f'Early Stop Threshold ≈ {early_stop_threshold:.4f}')

axes[1].scatter(df['Epoch'][train_min_epoch], train_min, color='darkred', marker='o',
                label=f'Train Min (Epoch {train_min_epoch})')
axes[1].scatter(df['Epoch'][val_min_epoch], val_min, color='darkblue', marker='o',
                label=f'Val Min (Epoch {val_min_epoch})')

axes[1].set_title("Zoomed-in View: Last 100 Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("RMSE")
axes[1].legend()
axes[1].grid(True)

# -------- Save and show --------
plt.tight_layout()
plt.savefig("mpp_lsc_train_val_loss_full_and_zoomed.png", dpi=300)
plt.show()


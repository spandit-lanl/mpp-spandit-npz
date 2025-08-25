import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

# File path (adjust if necessary)
csv_file = "out_train_val_loss.csv"

# Read the CSV using pandas
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()
df.columns = ['train', 'val']
epochs = np.arange(1, len(df) + 1)

# Compute rolling stats
window = 10
df['train_mean'] = df['train'].rolling(window, min_periods=1).mean()
df['train_std'] = df['train'].rolling(window, min_periods=1).std()
df['val_mean'] = df['val'].rolling(window, min_periods=1).mean()
df['val_std'] = df['val'].rolling(window, min_periods=1).std()

# 95% Confidence Interval: ±1.96 * std
ci_scale = 1.96

# Plotting
plt.figure(figsize=(6, 4))

# Training Loss
plt.plot(epochs, df['train_mean'], 'r-', label='Train Loss')
plt.fill_between(epochs,
                 df['train_mean'] - ci_scale * df['train_std'],
                 df['train_mean'] + ci_scale * df['train_std'],
                 color='red', alpha=0.3)

# Validation Loss
plt.plot(epochs, df['val_mean'], 'b-', label='Validation Loss')
plt.fill_between(epochs,
                 df['val_mean'] - ci_scale * df['val_std'],
                 df['val_mean'] + ci_scale * df['val_std'],
                 color='blue', alpha=0.3)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with 95% Confidence Intervals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mpp_lsc_train_val_loss_with_95ci.png", dpi=300)
plt.show()


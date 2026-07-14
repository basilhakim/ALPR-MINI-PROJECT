import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. Generate Synthetic Vibration Data
# ==========================================
t = np.linspace(0, 0.1, 1000) # 0.1 seconds
# New Tool: Clean sine wave + low noise
signal_new = 0.5 * np.sin(2 * np.pi * 100 * t) + np.random.normal(0, 0.1, 1000)
# Worn Tool: Higher amplitude + high noise + spikes
signal_worn = 1.5 * np.sin(2 * np.pi * 100 * t) + np.random.normal(0, 0.4, 1000)
signal_worn[::50] += 2.0 # Add random impacts

plt.figure(figsize=(10, 4))
plt.plot(t, signal_worn, 'r', alpha=0.7, label='Worn Tool')
plt.plot(t, signal_new, 'g', label='New Tool')
plt.title("Figure 4-1: Time Domain Vibration Signal (New vs Worn)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (g)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 2. Generate RMS Trend (Tool Life Curve)
# ==========================================
runs = np.arange(1, 11)
# Create an exponential curve to mimic wear
rms_values = 0.2 * np.exp(0.3 * runs) + np.random.normal(0, 0.05, 10)

plt.figure(figsize=(8, 5))
plt.plot(runs, rms_values, 'b-o', linewidth=2)
plt.axhline(y=1.5, color='r', linestyle='--', label='Failure Threshold')
plt.title("Figure 4-2: Vibration RMS Trend vs. Experimental Runs")
plt.xlabel("Run Number")
plt.ylabel("RMS Value (g)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 3. Generate Confusion Matrix
# ==========================================
# Mock predictions
y_true =  ['New']*30 + ['Middle']*30 + ['Worn']*30
y_pred =  ['New']*28 + ['Middle']*2 + ['Middle']*27 + ['Worn']*3 + ['Worn']*30

labels = ['New', 'Middle', 'Worn']
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Figure 4-4: SVM Classification Confusion Matrix")
plt.xlabel("Predicted Condition")
plt.ylabel("Actual Condition")
plt.show()
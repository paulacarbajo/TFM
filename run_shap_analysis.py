"""
SHAP Analysis for OOS Period

Analyzes which features are driving predictions toward -1 (short) in the OOS period.
Uses SHAP values to understand model behavior and identify key drivers of short predictions.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("SHAP ANALYSIS: UNDERSTANDING SHORT PREDICTIONS")
print("="*80)

# ============================================================================
# 1. LOAD TRAINED MODELS
# ============================================================================
print("\n[1] Loading trained models from fold 8...")

with open('data/processed/walk_forward_results.pkl', 'rb') as f:
    wf_results = pickle.load(f)

fold_8 = wf_results['all_fold_results'][-1]
lgbm_model = fold_8['models']['lightgbm']
feature_names = fold_8['feature_names']

print(f"[OK] Loaded LightGBM model from fold {fold_8['fold_number']}")
print(f"[OK] Features: {len(feature_names)}")

# ============================================================================
# 2. LOAD OOS DATA
# ============================================================================
print("\n[2] Loading OOS data (2020-2024)...")

df = pd.read_hdf('data/processed/assets.h5', key='engineered_features')

# Filter by date using the date level of the MultiIndex
dates = df.index.get_level_values('date')
oos_mask = (dates >= '2020-01-01') & (dates <= '2024-12-31')
df_oos = df[oos_mask].copy()

print(f"[OK] Loaded {len(df_oos)} rows")

# ============================================================================
# 3. PREPARE DATA (SAME AS BACKTEST)
# ============================================================================
print("\n[3] Preparing data with ticker_id...")

# Add ticker_id (0=SPY, 1=USO)
ticker_values = df_oos.index.get_level_values('ticker')
df_oos['ticker_id'] = (ticker_values == 'USO').astype(int)

# Extract SPY data only
spy_mask = df_oos.index.get_level_values('ticker') == 'SPY'
df_spy = df_oos[spy_mask].copy()

# Get features
X_spy = df_spy[feature_names].reset_index(drop=True)
y_spy = df_spy['label_binary'].reset_index(drop=True)

print(f"[OK] SPY samples: {len(X_spy)}")
print(f"[OK] Features used: {feature_names}")

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================
print("\n[4] Making predictions...")

y_proba = lgbm_model.predict_proba(X_spy)[:, 1]
y_pred_binary = (y_proba > 0.5).astype(int)
y_pred_signal = np.where(y_pred_binary == 1, 1, -1)

n_long = (y_pred_signal == 1).sum()
n_short = (y_pred_signal == -1).sum()

print(f"[OK] Long predictions:  {n_long:4d} ({n_long/len(y_pred_signal)*100:5.1f}%)")
print(f"[OK] Short predictions: {n_short:4d} ({n_short/len(y_pred_signal)*100:5.1f}%)")

# ============================================================================
# 5. COMPUTE SHAP VALUES
# ============================================================================
print("\n[5] Computing SHAP values (this may take 1-2 minutes)...")

explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_spy)

# For binary classification, get values for positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"[OK] SHAP values computed: {shap_values.shape}")

# ============================================================================
# 6. SUMMARY PLOT - TOP 15 FEATURES
# ============================================================================
print("\n[6] Creating SHAP summary plot...")

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, 
    X_spy, 
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.title('SHAP Summary: Top 15 Features (SPY, OOS 2020-2024)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('notes/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Saved: notes/shap_summary.png")

# ============================================================================
# 7. ANALYZE SHORT PREDICTIONS
# ============================================================================
print("\n[7] Analyzing features driving SHORT predictions...")

# Filter to short predictions only
short_mask = y_pred_signal == -1
X_short = X_spy[short_mask]
shap_short = shap_values[short_mask]

print(f"[OK] Analyzing {len(X_short)} short predictions")

# Calculate mean absolute SHAP value for each feature
mean_abs_shap = np.abs(shap_short).mean(axis=0)

# Create DataFrame for analysis
shap_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap,
    'mean_shap': shap_short.mean(axis=0)
})

# Sort by mean absolute SHAP (importance)
shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)

print("\nTop 15 features driving SHORT predictions:")
print("-" * 80)
print(f"{'Feature':<30} {'Mean |SHAP|':<15} {'Mean SHAP':<15}")
print("-" * 80)

for idx, row in shap_df.head(15).iterrows():
    direction = "→ SHORT" if row['mean_shap'] < 0 else "→ LONG"
    print(f"{row['feature']:<30} {row['mean_abs_shap']:>12.4f}    {row['mean_shap']:>12.4f} {direction}")

# ============================================================================
# 8. PLOT: FEATURES DRIVING SHORT PREDICTIONS
# ============================================================================
print("\n[8] Creating plot of short prediction drivers...")

top_15_short = shap_df.head(15).copy()

fig, ax = plt.subplots(figsize=(12, 8))

# Create horizontal bar plot
colors = ['red' if x < 0 else 'blue' for x in top_15_short['mean_shap']]
y_pos = np.arange(len(top_15_short))

ax.barh(y_pos, top_15_short['mean_shap'], color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_15_short['feature'])
ax.invert_yaxis()
ax.set_xlabel('Mean SHAP Value', fontsize=12)
ax.set_title('Top 15 Features Driving SHORT Predictions (SPY, OOS 2020-2024)', 
             fontsize=14, pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.7, label='Pushes toward SHORT (-1)'),
    Patch(facecolor='blue', alpha=0.7, label='Pushes toward LONG (+1)')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('notes/shap_short_drivers.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Saved: notes/shap_short_drivers.png")

# ============================================================================
# 9. ADDITIONAL INSIGHTS
# ============================================================================
print("\n[9] Additional insights:")
print("-" * 80)

# Feature importance from model
feature_importance = lgbm_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 features by LightGBM importance:")
for idx, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:<30} {row['importance']:>10.0f}")

# Check if ticker_id is important
ticker_id_importance = importance_df[importance_df['feature'] == 'ticker_id']
if not ticker_id_importance.empty:
    rank = (importance_df['feature'] == 'ticker_id').values.nonzero()[0][0] + 1
    print(f"\nticker_id rank: {rank}/{len(feature_names)}")
    print(f"ticker_id importance: {ticker_id_importance['importance'].values[0]:.0f}")
else:
    print("\nticker_id not found in features")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"""
KEY FINDINGS:
- Total OOS predictions (SPY): {len(y_pred_signal)}
- Short predictions: {n_short} ({n_short/len(y_pred_signal)*100:.1f}%)
- Long predictions: {n_long} ({n_long/len(y_pred_signal)*100:.1f}%)

OUTPUTS:
- notes/shap_summary.png: Overall feature importance
- notes/shap_short_drivers.png: Features driving short predictions

INTERPRETATION:
- Red bars = Features pushing predictions toward SHORT (-1)
- Blue bars = Features pushing predictions toward LONG (+1)
- Larger absolute values = Stronger influence

The plots show which features are responsible for the model's bias
toward short predictions in the OOS period.
""")
print("="*80)

"""
SHAP Analysis for OOS Period

Analyzes which features are driving predictions toward -1 (short) in the OOS period.
Uses SHAP values to understand model behavior and identify key drivers of short predictions.
"""

import pickle
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import argparse

parser = argparse.ArgumentParser(description='SHAP analysis for OOS period')
parser.add_argument('--config', default='config/config.yaml',
                    help='Path to config YAML (default: config/config.yaml)')
args, _ = parser.parse_known_args()
config_stem = Path(args.config).stem
suffix = config_stem[len('config'):]

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
wf_cfg = config.get('models', {}).get('walk_forward', {})
oos_start = wf_cfg.get('test_start', '2020-01-01')
oos_end = wf_cfg.get('test_end', '2024-12-31')

print("="*80)
print("SHAP ANALYSIS: UNDERSTANDING SHORT PREDICTIONS")
print("="*80)
print(f"Config: {args.config}  (output suffix: '{suffix}')")
print("="*80)

# ============================================================================
# 1. LOAD TRAINED MODELS
# ============================================================================
print("\n[1] Loading trained models from last fold (most recent training data)...")

with open(f'data/processed/walk_forward_results{suffix}.pkl', 'rb') as f:
    wf_results = pickle.load(f)

last_fold = wf_results['all_fold_results'][-1]
lgbm_model = last_fold['models']['lightgbm']
feature_names = last_fold['feature_names']

print(f"[OK] Loaded LightGBM model from fold {last_fold['fold_number']}")
print(f"[OK] Features: {len(feature_names)}")

# ============================================================================
# 2. LOAD OOS DATA
# ============================================================================
print("\n[2] Loading OOS data (2020-2024)...")

df = pd.read_hdf('data/processed/assets.h5', key='engineered_features')

# Filter by date using the date level of the MultiIndex
dates = df.index.get_level_values('date')
oos_mask = (dates >= oos_start) & (dates <= oos_end)
df_oos = df[oos_mask].copy()

print(f"[OK] Loaded {len(df_oos)} rows")

# ============================================================================
# 3. PREPARE DATA
# ============================================================================
print("\n[3] Preparing SPY data...")

# Extract SPY data only
spy_mask = df_oos.index.get_level_values('ticker') == 'SPY'
df_spy = df_oos[spy_mask].copy()

# Drop rows with NaN in features or label (last 8 rows have NaN label_binary, max_holding_period)
valid_mask = df_spy[feature_names].notna().all(axis=1) & df_spy['label_binary'].notna()
df_spy = df_spy[valid_mask]

# Get features (use same feature_names from trained model)
X_spy = df_spy[feature_names].reset_index(drop=True)
y_spy = df_spy['label_binary'].reset_index(drop=True)

print(f"[OK] SPY samples: {len(X_spy)} (dropped {(~valid_mask).sum()} NaN rows)")
print(f"[OK] Features used ({len(feature_names)}): {feature_names}")

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
legend_elements = [
    mpatches.Patch(facecolor='red', alpha=0.7, label='Pushes toward SHORT (-1)'),
    mpatches.Patch(facecolor='blue', alpha=0.7, label='Pushes toward LONG (+1)')
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

# ============================================================================
# 10. SUMMARY - ITERATION 1
# ============================================================================
print("\n" + "="*80)
print("ITERATION 1 ANALYSIS COMPLETE")
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

# ============================================================================
# ITERATION 2: WITH REGIME FEATURES
# ============================================================================
print("\n\n" + "="*80)
print("ITERATION 2: SHAP ANALYSIS WITH REGIME FEATURES")
print("="*80)

# ============================================================================
# 11. LOAD REGIME MODEL AND DETECTOR
# ============================================================================
print("\n[11] Loading regime model from last fold (most recent training data)...")

with open(f'data/processed/walk_forward_results_regime{suffix}.pkl', 'rb') as f:
    wf_results_regime = pickle.load(f)

last_fold_regime = wf_results_regime['all_fold_results'][-1]
lgbm_model_regime = last_fold_regime['models']['lightgbm']
regime_detector = last_fold_regime['regime_detector']
feature_names_regime = last_fold_regime['feature_names']

print(f"[OK] Loaded LightGBM model from fold {last_fold_regime['fold_number']}")
print(f"[OK] Features: {len(feature_names_regime)}")
print(f"[OK] Loaded RegimeDetector")

# ============================================================================
# 12. ADD REGIME FEATURES TO OOS DATA
# ============================================================================
print("\n[12] Adding regime features to OOS data...")

# Use the same df_oos from iteration 1
# Predict regimes for OOS period
regime_labels, regime_probs = regime_detector.predict(df_oos)

df_oos_regime = df_oos.copy()
df_oos_regime['regime_state'] = regime_labels
df_oos_regime['regime_prob_0'] = regime_probs[:, 0]
df_oos_regime['regime_prob_1'] = regime_probs[:, 1]
df_oos_regime['regime_prob_2'] = regime_probs[:, 2]

print(f"[OK] Added regime features to {len(df_oos_regime)} rows")

# Extract SPY data only
spy_mask_regime = df_oos_regime.index.get_level_values('ticker') == 'SPY'
df_spy_regime = df_oos_regime[spy_mask_regime].copy()

# Drop rows with NaN in features or label (last 8 rows have NaN label_binary, max_holding_period)
valid_mask_regime = df_spy_regime[feature_names_regime].notna().all(axis=1) & df_spy_regime['label_binary'].notna()
df_spy_regime = df_spy_regime[valid_mask_regime]

# Get features (15 features: 11 technical + 4 regime)
X_spy_regime = df_spy_regime[feature_names_regime].reset_index(drop=True)
y_spy_regime = df_spy_regime['label_binary'].reset_index(drop=True)

print(f"[OK] SPY samples: {len(X_spy_regime)} (dropped {(~valid_mask_regime).sum()} NaN rows)")
print(f"[OK] Features used ({len(feature_names_regime)}): {feature_names_regime}")

# ============================================================================
# 13. MAKE PREDICTIONS WITH REGIME MODEL
# ============================================================================
print("\n[13] Making predictions with regime model...")

y_proba_regime = lgbm_model_regime.predict_proba(X_spy_regime)[:, 1]
y_pred_binary_regime = (y_proba_regime > 0.5).astype(int)
y_pred_signal_regime = np.where(y_pred_binary_regime == 1, 1, -1)

n_long_regime = (y_pred_signal_regime == 1).sum()
n_short_regime = (y_pred_signal_regime == -1).sum()

print(f"[OK] Long predictions:  {n_long_regime:4d} ({n_long_regime/len(y_pred_signal_regime)*100:5.1f}%)")
print(f"[OK] Short predictions: {n_short_regime:4d} ({n_short_regime/len(y_pred_signal_regime)*100:5.1f}%)")

# ============================================================================
# 14. COMPUTE SHAP VALUES FOR REGIME MODEL
# ============================================================================
print("\n[14] Computing SHAP values for regime model (this may take 1-2 minutes)...")

explainer_regime = shap.TreeExplainer(lgbm_model_regime)
shap_values_regime = explainer_regime.shap_values(X_spy_regime)

# For binary classification, get values for positive class
if isinstance(shap_values_regime, list):
    shap_values_regime = shap_values_regime[1]

print(f"[OK] SHAP values computed: {shap_values_regime.shape}")

# ============================================================================
# 15. SUMMARY PLOT - TOP 15 FEATURES (REGIME)
# ============================================================================
print("\n[15] Creating SHAP summary plot for regime model...")

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_regime, 
    X_spy_regime, 
    feature_names=feature_names_regime,
    max_display=15,
    show=False
)
plt.title('SHAP Summary: Top 15 Features with Regime (SPY, OOS 2020-2024)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('notes/shap_summary_regime.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Saved: notes/shap_summary_regime.png")

# ============================================================================
# 16. ANALYZE SHORT PREDICTIONS (REGIME)
# ============================================================================
print("\n[16] Analyzing features driving SHORT predictions (regime model)...")

# Filter to short predictions only
short_mask_regime = y_pred_signal_regime == -1
X_short_regime = X_spy_regime[short_mask_regime]
shap_short_regime = shap_values_regime[short_mask_regime]

print(f"[OK] Analyzing {len(X_short_regime)} short predictions")

# Calculate mean absolute SHAP value for each feature
mean_abs_shap_regime = np.abs(shap_short_regime).mean(axis=0)

# Create DataFrame for analysis
shap_df_regime = pd.DataFrame({
    'feature': feature_names_regime,
    'mean_abs_shap': mean_abs_shap_regime,
    'mean_shap': shap_short_regime.mean(axis=0)
})

# Sort by mean absolute SHAP (importance)
shap_df_regime = shap_df_regime.sort_values('mean_abs_shap', ascending=False)

print("\nTop 15 features driving SHORT predictions (with regime):")
print("-" * 80)
print(f"{'Feature':<30} {'Mean |SHAP|':<15} {'Mean SHAP':<15}")
print("-" * 80)

for idx, row in shap_df_regime.head(15).iterrows():
    direction = "→ SHORT" if row['mean_shap'] < 0 else "→ LONG"
    print(f"{row['feature']:<30} {row['mean_abs_shap']:>12.4f}    {row['mean_shap']:>12.4f} {direction}")

# ============================================================================
# 17. PLOT: FEATURES DRIVING SHORT PREDICTIONS (REGIME)
# ============================================================================
print("\n[17] Creating plot of short prediction drivers (regime model)...")

top_15_short_regime = shap_df_regime.head(15).copy()

fig, ax = plt.subplots(figsize=(12, 8))

# Create horizontal bar plot
colors_regime = ['red' if x < 0 else 'blue' for x in top_15_short_regime['mean_shap']]
y_pos_regime = np.arange(len(top_15_short_regime))

ax.barh(y_pos_regime, top_15_short_regime['mean_shap'], color=colors_regime, alpha=0.7)
ax.set_yticks(y_pos_regime)
ax.set_yticklabels(top_15_short_regime['feature'])
ax.invert_yaxis()
ax.set_xlabel('Mean SHAP Value', fontsize=12)
ax.set_title('Top 15 Features Driving SHORT Predictions with Regime (SPY, OOS 2020-2024)', 
             fontsize=14, pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='red', alpha=0.7, label='Pushes toward SHORT (-1)'),
    mpatches.Patch(facecolor='blue', alpha=0.7, label='Pushes toward LONG (+1)')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('notes/shap_short_drivers_regime.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Saved: notes/shap_short_drivers_regime.png")

# ============================================================================
# 18. SUMMARY - ITERATION 2
# ============================================================================
print("\n" + "="*80)
print("ITERATION 2 ANALYSIS COMPLETE")
print("="*80)
print(f"""
KEY FINDINGS:
- Total OOS predictions (SPY): {len(y_pred_signal_regime)}
- Short predictions: {n_short_regime} ({n_short_regime/len(y_pred_signal_regime)*100:.1f}%)
- Long predictions: {n_long_regime} ({n_long_regime/len(y_pred_signal_regime)*100:.1f}%)

OUTPUTS:
- notes/shap_summary_regime.png: Overall feature importance with regime
- notes/shap_short_drivers_regime.png: Features driving short predictions with regime

REGIME FEATURE IMPORTANCE:
Check the plots to see how regime_state, regime_prob_0, regime_prob_1, and
regime_prob_2 rank among the 15 features in driving predictions.
""")
print("="*80)

print("\n" + "="*80)
print("ALL ANALYSES COMPLETE")
print("="*80)

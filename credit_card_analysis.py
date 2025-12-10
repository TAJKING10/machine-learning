"""
Credit Card Default Prediction - Complete Analysis
Dataset: credit_card_default.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*80)
print("CREDIT CARD DEFAULT PREDICTION - COMPLETE ANALYSIS")
print("="*80)

# ============================================================================
# 1️⃣ LOAD THE DATASET
# ============================================================================
print("\n1️⃣ Loading Dataset...")
df = pd.read_csv('credit_card_default.csv')

print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# ============================================================================
# 2️⃣ DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("2️⃣ DATA CLEANING")
print("="*80)

# Drop ID column
print("\nDropping ID column...")
df = df.drop('ID', axis=1)
print(f"New shape: {df.shape}")

# Check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found!")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows")

# Check target distribution
print("\nTarget Variable Distribution:")
print(df['default.payment.next.month'].value_counts())
print("\nTarget Proportions:")
print(df['default.payment.next.month'].value_counts(normalize=True))

# ============================================================================
# 3️⃣ DATA VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("3️⃣ GENERATING VISUALIZATIONS")
print("="*80)

# Create output directory for plots
import os
os.makedirs('plots', exist_ok=True)

# ---- Histogram: AGE ----
print("\nGenerating histogram for AGE...")
plt.figure(figsize=(10, 6))
plt.hist(df['AGE'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Age', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('plots/01_histogram_age.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/01_histogram_age.png")

# ---- Histogram: LIMIT_BAL ----
print("Generating histogram for LIMIT_BAL...")
plt.figure(figsize=(10, 6))
plt.hist(df['LIMIT_BAL'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
plt.title('Distribution of Credit Limit', fontsize=16, fontweight='bold')
plt.xlabel('Credit Limit Balance', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('plots/02_histogram_limit_bal.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/02_histogram_limit_bal.png")

# ---- Histogram: BILL_AMT1 ----
print("Generating histogram for BILL_AMT1...")
plt.figure(figsize=(10, 6))
plt.hist(df['BILL_AMT1'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Distribution of Bill Amount (Month 1)', fontsize=16, fontweight='bold')
plt.xlabel('Bill Amount', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('plots/03_histogram_bill_amt1.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/03_histogram_bill_amt1.png")

# ---- Countplot: SEX vs Default ----
print("Generating countplot for SEX vs Default...")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='SEX', hue='default.payment.next.month', palette='Set2')
plt.title('Gender vs Default Status', fontsize=16, fontweight='bold')
plt.xlabel('Gender (1=Male, 2=Female)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Default', labels=['No Default', 'Default'])
plt.savefig('plots/04_countplot_sex_vs_default.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/04_countplot_sex_vs_default.png")

# ---- Barplot: EDUCATION vs Default Rate ----
print("Generating barplot for EDUCATION vs Default Rate...")
education_default = df.groupby('EDUCATION')['default.payment.next.month'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=education_default, x='EDUCATION', y='default.payment.next.month', palette='viridis')
plt.title('Education Level vs Default Rate', fontsize=16, fontweight='bold')
plt.xlabel('Education (1=Grad School, 2=University, 3=High School)', fontsize=12)
plt.ylabel('Default Rate', fontsize=12)
plt.ylim(0, 0.5)
plt.savefig('plots/05_barplot_education_vs_default.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/05_barplot_education_vs_default.png")

# ---- Boxplot: LIMIT_BAL by Default ----
print("Generating boxplot for LIMIT_BAL by Default...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='default.payment.next.month', y='LIMIT_BAL', palette='Set1')
plt.title('Credit Limit by Default Status', fontsize=16, fontweight='bold')
plt.xlabel('Default (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Credit Limit Balance', fontsize=12)
plt.savefig('plots/06_boxplot_limit_bal_by_default.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/06_boxplot_limit_bal_by_default.png")

# ---- Correlation Heatmap ----
print("Generating correlation heatmap...")
plt.figure(figsize=(16, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of All Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/07_correlation_heatmap.png")

# ---- Top Correlations with Target ----
print("\nTop 10 Features Correlated with Default:")
target_corr = correlation_matrix['default.payment.next.month'].sort_values(ascending=False)
print(target_corr.head(11)[1:])  # Exclude self-correlation

# ============================================================================
# 4️⃣ DATA PREPROCESSING FOR MODELING
# ============================================================================
print("\n" + "="*80)
print("4️⃣ DATA PREPROCESSING FOR MODELING")
print("="*80)

# Split features and target
print("\nSplitting features (X) and target (y)...")
X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split (80-20)
print("\nSplitting into train and test sets (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale numerical features
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Scaling complete!")

# ============================================================================
# 5️⃣ MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n" + "="*80)
print("5️⃣ MODEL TRAINING AND EVALUATION")
print("="*80)

# Dictionary to store results
results = []

# ---- Model 1: Logistic Regression ----
print("\n[1/4] Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_pred_proba)

print(f"✓ Logistic Regression - Accuracy: {lr_accuracy:.4f}, F1: {lr_f1:.4f}, AUC: {lr_auc:.4f}")

results.append({
    'Model': 'Logistic Regression',
    'Accuracy': lr_accuracy,
    'Precision': lr_precision,
    'Recall': lr_recall,
    'F1-Score': lr_f1,
    'AUC-ROC': lr_auc
})

# ---- Model 2: Decision Tree ----
print("[2/4] Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_pred_proba = dt_model.predict_proba(X_test_scaled)[:, 1]

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_pred_proba)

print(f"✓ Decision Tree - Accuracy: {dt_accuracy:.4f}, F1: {dt_f1:.4f}, AUC: {dt_auc:.4f}")

results.append({
    'Model': 'Decision Tree',
    'Accuracy': dt_accuracy,
    'Precision': dt_precision,
    'Recall': dt_recall,
    'F1-Score': dt_f1,
    'AUC-ROC': dt_auc
})

# ---- Model 3: Random Forest ----
print("[3/4] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print(f"✓ Random Forest - Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}")

results.append({
    'Model': 'Random Forest',
    'Accuracy': rf_accuracy,
    'Precision': rf_precision,
    'Recall': rf_recall,
    'F1-Score': rf_f1,
    'AUC-ROC': rf_auc
})

# ---- Model 4: Gradient Boosting ----
print("[4/4] Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_pred_proba = gb_model.predict_proba(X_test_scaled)[:, 1]

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)
gb_auc = roc_auc_score(y_test, gb_pred_proba)

print(f"✓ Gradient Boosting - Accuracy: {gb_accuracy:.4f}, F1: {gb_f1:.4f}, AUC: {gb_auc:.4f}")

results.append({
    'Model': 'Gradient Boosting',
    'Accuracy': gb_accuracy,
    'Precision': gb_precision,
    'Recall': gb_recall,
    'F1-Score': gb_f1,
    'AUC-ROC': gb_auc
})

# ============================================================================
# 6️⃣ RESULTS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("6️⃣ MODEL COMPARISON RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))

# Find best model
best_model_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {results_df.loc[best_model_idx, 'F1-Score']:.4f})")

# Save results to CSV
results_df.to_csv('model_results.csv', index=False)
print("\n✓ Model results saved to: model_results.csv")

# ============================================================================
# 7️⃣ CONFUSION MATRICES AND ROC CURVES
# ============================================================================
print("\n" + "="*80)
print("7️⃣ GENERATING CONFUSION MATRICES AND ROC CURVES")
print("="*80)

models_dict = {
    'Logistic Regression': (lr_pred, lr_pred_proba),
    'Decision Tree': (dt_pred, dt_pred_proba),
    'Random Forest': (rf_pred, rf_pred_proba),
    'Gradient Boosting': (gb_pred, gb_pred_proba)
}

# ---- Confusion Matrices ----
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (model_name, (pred, _)) in enumerate(models_dict.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                cbar=False, square=True)
    axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('plots/08_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/08_confusion_matrices.png")

# ---- ROC Curves ----
plt.figure(figsize=(10, 8))

for model_name, (_, pred_proba) in models_dict.items():
    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    auc = roc_auc_score(y_test, pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('plots/09_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/09_roc_curves.png")

# ---- Model Performance Comparison Bar Chart ----
plt.figure(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.15

plt.bar(x - 2*width, results_df['Accuracy'], width, label='Accuracy', color='skyblue')
plt.bar(x - width, results_df['Precision'], width, label='Precision', color='lightcoral')
plt.bar(x, results_df['Recall'], width, label='Recall', color='lightgreen')
plt.bar(x + width, results_df['F1-Score'], width, label='F1-Score', color='gold')
plt.bar(x + 2*width, results_df['AUC-ROC'], width, label='AUC-ROC', color='plum')

plt.xlabel('Models', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xticks(x, results_df['Model'], rotation=15, ha='right')
plt.legend(loc='lower right')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/10_model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/10_model_performance_comparison.png")

# ============================================================================
# 8️⃣ FEATURE IMPORTANCE (GRADIENT BOOSTING)
# ============================================================================
print("\n" + "="*80)
print("8️⃣ FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
plt.title('Top 15 Feature Importance (Gradient Boosting)', fontsize=16, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('plots/11_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: plots/11_feature_importance.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  📊 Visualizations: plots/ folder (11 PNG files)")
print("  📄 Model Results: model_results.csv")
print("\nAll visualizations and results have been saved successfully!")
print("="*80)

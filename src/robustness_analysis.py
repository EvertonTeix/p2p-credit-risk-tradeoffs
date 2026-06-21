import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# Configuração de Paths
MODELS_DIR = '../models'
OUTPUTS_DIR = '../outputs/Robustness_Outputs'
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 1. CARREGAMENTO DOS MODELOS E DADOS
print("Carregando modelos e dados...")
best_dt = joblib.load(os.path.join(MODELS_DIR, 'best_dt.joblib'))
best_rf = joblib.load(os.path.join(MODELS_DIR, 'best_rf.joblib'))
best_lr = joblib.load(os.path.join(MODELS_DIR, 'best_lr.joblib'))
best_xgb = joblib.load(os.path.join(MODELS_DIR, 'best_xgb.joblib'))

X_test_scaled = joblib.load(os.path.join(MODELS_DIR, 'X_test_final.joblib'))
y_test = joblib.load(os.path.join(MODELS_DIR, 'y_test_final.joblib'))
features_names = joblib.load(os.path.join(MODELS_DIR, 'features_names_final.joblib'))

# 2. FUNÇÕES DE PERTURBAÇÃO
def perturb_numeric(X, col_idx, magnitude, seed=42):
    np.random.seed(seed)
    X_pert = X.copy()
    std = np.std(X_pert[:, col_idx])
    noise = np.random.normal(0, std * magnitude, size=X_pert.shape[0])
    X_pert[:, col_idx] = X_pert[:, col_idx] + noise
    return X_pert

def perturb_categorical(X, col_idx, magnitude, valid_values, seed=42):
    np.random.seed(seed)
    X_pert = X.copy()
    n_samples = X_pert.shape[0]
    n_flip = int(n_samples * magnitude)
    
    flip_indices = np.random.choice(n_samples, n_flip, replace=False)
    for idx in flip_indices:
        current_val = X_pert[idx, col_idx]
        other_values = [v for v in valid_values if v != current_val]
        if other_values:
            X_pert[idx, col_idx] = np.random.choice(other_values)
            
    return X_pert

# Precisamos saber quais colunas na matriz escalonada são binárias/categóricas e quais são numéricas
col_valid_values = {}
col_types = {}

for i, col in enumerate(features_names):
    unique_vals = np.unique(X_test_scaled[:, i])
    # Se só tem valores binários/poucos valores únicos (resultado do get_dummies), consideramos categórica
    if len(unique_vals) <= 2:
        col_types[i] = 'categorical'
        col_valid_values[i] = unique_vals
    else:
        col_types[i] = 'numeric'

# 3. EXTRAÇÃO DAS TOP-5 FEATURES VIA SHAP
print("Calculando SHAP para extrair top-5 features...")
X_test_df = pd.DataFrame(X_test_scaled, columns=features_names)
X_test_sample = X_test_df.sample(n=min(300, len(X_test_df)), random_state=42)

modelos_dict = {
    'Decision Tree': best_dt,
    'Logistic Regression': best_lr,
    'Random Forest': best_rf,
    'XGBoost': best_xgb
}

top_features_per_model = {}
shap_records = []

for name, model in modelos_dict.items():
    print(f"  - Extraindo features de {name}...")
    if name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        explainer = shap.TreeExplainer(model, feature_names=features_names)
        shap_values = explainer(X_test_sample)
        
        # O objeto shap_values pode ter o shape (samples, features, classes) ou (samples, features)
        if len(shap_values.shape) > 2:
            vals = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
        else:
            vals = np.abs(shap_values.values).mean(axis=0)
    else:
        explainer = shap.LinearExplainer(model, X_test_sample)
        shap_values = explainer(X_test_sample)
        vals = np.abs(shap_values.values).mean(axis=0)
        
    top_indices = np.argsort(vals)[::-1][:5]
    top_features = [(features_names[i], i, vals[i]) for i in top_indices]
    
    top_features_per_model[name] = [i for _, i, _ in top_features]
    
    for rank, (feat_name, feat_idx, imp) in enumerate(top_features):
        shap_records.append({
            'Model': name,
            'Rank': rank + 1,
            'Feature': feat_name,
            'Mean_Abs_SHAP': imp
        })

df_shap = pd.DataFrame(shap_records)
df_shap.to_csv(os.path.join(MODELS_DIR, 'shap_top_features.csv'), index=False)
print("Top features salvas em models/shap_top_features.csv")

# 4. FUNÇÃO PRINCIPAL DE ANÁLISE DE ROBUSTEZ
def run_robustness_analysis(model, name, X_test, y_test, top_indices, magnitudes=[0.05, 0.10, 0.20]):
    results = []
    
    # Baseline
    y_pred_base = model.predict(X_test)
    y_proba_base = model.predict_proba(X_test)[:, 1]
    
    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base)
    prec_base = precision_score(y_test, y_pred_base, zero_division=0)
    rec_base = recall_score(y_test, y_pred_base, zero_division=0)
    auc_base = roc_auc_score(y_test, y_proba_base)
    
    results.append({
        'Model': name,
        'Magnitude': 0.0,
        'Accuracy': acc_base,
        'F1 Score': f1_base,
        'Precision': prec_base,
        'Recall': rec_base,
        'AUC': auc_base,
        'Flip Rate': 0.0
    })
    
    # Perturbações
    for mag in magnitudes:
        X_pert = X_test.copy()
        for idx in top_indices:
            if col_types[idx] == 'numeric':
                X_pert = perturb_numeric(X_pert, idx, mag, seed=42)
            else:
                X_pert = perturb_categorical(X_pert, idx, mag, col_valid_values[idx], seed=42)
                
        y_pred_pert = model.predict(X_pert)
        y_proba_pert = model.predict_proba(X_pert)[:, 1]
        
        acc = accuracy_score(y_test, y_pred_pert)
        f1 = f1_score(y_test, y_pred_pert)
        prec = precision_score(y_test, y_pred_pert, zero_division=0)
        rec = recall_score(y_test, y_pred_pert, zero_division=0)
        auc = roc_auc_score(y_test, y_proba_pert)
        
        flip_rate = np.mean(y_pred_base != y_pred_pert)
        
        results.append({
            'Model': name,
            'Magnitude': mag,
            'Accuracy': acc,
            'F1 Score': f1,
            'Precision': prec,
            'Recall': rec,
            'AUC': auc,
            'Flip Rate': flip_rate
        })
        
    return pd.DataFrame(results)

# 5. EXECUTANDO PARA OS 4 MODELOS
print("\nExecutando análise de robustez...")
all_results = []
magnitudes = [0.05, 0.10, 0.20]

for name, model in modelos_dict.items():
    df_res = run_robustness_analysis(model, name, X_test_scaled, y_test, top_features_per_model[name], magnitudes)
    all_results.append(df_res)
    
df_final = pd.concat(all_results, ignore_index=True)

csv_path = os.path.join(OUTPUTS_DIR, 'robustness_analysis.csv')
df_final.to_csv(csv_path, index=False)
print(f"Resultados consolidados salvos em {csv_path}")

# 6. GERAÇÃO DE GRÁFICOS
print("\nGerando gráficos...")
sns.set_theme(style="whitegrid")

# F1 Plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_final, x='Magnitude', y='F1 Score', hue='Model', marker='o', linewidth=2)
plt.title('F1 Score Degradation by Perturbation Magnitude', fontweight='bold')
plt.xlabel('Magnitude')
plt.ylabel('F1 Score')
plt.xticks([0.0, 0.05, 0.10, 0.20])
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'robustness_f1_plot.png'), dpi=300)
plt.close()

# Flip Rate Plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_final, x='Magnitude', y='Flip Rate', hue='Model', marker='o', linewidth=2)
plt.title('Flip Rate by Perturbation Magnitude', fontweight='bold')
plt.xlabel('Magnitude')
# Flip rate is a fraction between 0 and 1, maybe show as percentage
plt.ylabel('Flip Rate')
plt.xticks([0.0, 0.05, 0.10, 0.20])
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'robustness_flip_rate_plot.png'), dpi=300)
plt.close()

print("Análise de Robustez Concluída!")

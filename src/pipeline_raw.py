import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import shap
import os

from lime import lime_tabular
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FUNÇÕES AUXILIARES
# =============================================================================

def analyse_metrics(y_predict, y_proba, y_test):
    return {
        "Accuracy": accuracy_score(y_test, y_predict),
        "F1 Score" : f1_score(y_test, y_predict),
        "Precision": precision_score(y_test, y_predict, zero_division=0),
        "Recall": recall_score(y_test, y_predict, zero_division=0),
        "AUC": roc_auc_score(y_test, y_proba)
    }

def train_model_with_timing(model, X_train, y_train, X_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_predict = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_predict, y_proba, training_time

# =============================================================================
# 2. CARREGAMENTO E PRÉ-PROCESSAMENTO (IDÊNTICO AO PIPELINE COMPLETO)
# =============================================================================

print("Carregando banco de dados...")
database = pd.read_csv('../data/LoanData_Bondora.csv')
database = database[database['Status'] != 'Current'] 

remove_database = [
    'ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC', 'BiddingStartedOn',
    'UserName', 'LoanApplicationStartedDate', 'LoanDate', 'FirstPaymentDate',
    'MaturityDate_Original', 'MaturityDate_Last', 'ApplicationSignedHour',
    'ApplicationSignedWeekday', 'MonthlyPaymentDay',
    'ActiveScheduleFirstPaymentReached', 'LastPaymentOn', 'ExpectedLoss',
    'ExpectedReturn', 'ProbabilityOfDefault', 'RecoveryStage',
    'ModelVersion', 'Rating', 'Restructured', 'CreditScoreEsMicroL', 'DateOfBirth',
    'CurrentDebtDaysPrimary', 'DebtOccuredOn', 'CurrentDebtDaysSecondary',
    'DebtOccuredOnForSecondary', 'DefaultDate', 'EL_V0', 'Rating_V0', 'EL_V1',
    'Rating_V1', 'Rating_V2', 'CreditScoreEsEquifaxRisk',
    'CreditScoreFiAsiakasTietoRiskGrade', 'CreditScoreEeMini', 'GracePeriodStart',
    'GracePeriodEnd', 'NextPaymentDate', 'ReScheduledOn',
    'ActiveLateLastPaymentCategory', 'EmploymentDurationCurrentEmployer',
    'PlannedInterestTillDate', 'PlannedPrincipalTillDate', 'PrincipalOverdueBySchedule',
    'WorseLateCategory', 'PrincipalBalance', 'InterestAndPenaltyBalance',
    'NextPaymentNr', 'ContractEndDate', 'PrincipalPaymentsMade',
    'InterestAndPenaltyPaymentsMade', 'PrincipalWriteOffs', 'InterestAndPenaltyWriteOffs'
]

database = database.drop(columns=remove_database, errors='ignore')

nullFeatures = [col for col in database.columns if database[col].isnull().sum() / database.shape[0] > 0.2]
database = database.drop(columns=nullFeatures)

numerical_features = database.select_dtypes(include=np.number).columns.tolist()
categorical_features = database.select_dtypes(exclude=np.number).columns.tolist()

for col in numerical_features:
    database[col] = database[col].fillna(database[col].median())

database = database[numerical_features + categorical_features]
database['Status'] = database['Status'].map({'Repaid': 0, 'Late': 1})

for col in categorical_features:
    database[col] = database[col].astype(str)
    le = LabelEncoder()
    database[col] = le.fit_transform(database[col])

cols_cat = ["VerificationType", "LanguageCode", "UseOfLoan", "EmploymentStatus", "OccupationArea", "HomeOwnershipType", "NewCreditCustomer", "Country"]
cols_cat_existentes = [c for c in cols_cat if c in database.columns]

database = pd.get_dummies(database, columns=cols_cat_existentes, prefix=cols_cat_existentes, drop_first=True)
for col in database.columns:
    if database[col].dtype == 'bool':
        database[col] = database[col].astype(int)
        
corr_matrix = database.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
colunas_remover = [col for col in upper.columns if any(upper[col] >= 0.7)]
database.drop(columns=colunas_remover, inplace=True, errors='ignore')

y = database['Status']
X = database.drop('Status', axis=1)
features_names_final = X.columns

# =============================================================================
# 3. K-FOLD E GRID SEARCH
# =============================================================================

param_DT = {'max_depth': range(1, 11)}
param_LR = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
param_RF = {'n_estimators': [100], 'max_depth': [5, 10], 'criterion': ['gini']} 
param_XGB = {'n_estimators': [50, 100],'max_depth': [3, 5, 7],'learning_rate': [0.1, 0.2]}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

metrics_per_fold = {
    'Decision Tree': [],
    'Random Forest': [],
    'Logistic Regression': [],
    'XGBoost': []
}

best_dt, best_rf, best_lr, best_xgb = None, None, None, None
X_train_final, X_test_final = None, None
y_test_final = None

fold = 1
for train_index, test_index in kfold.split(X):
    print(f"Rodando KFold {fold}/5 ...")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    X_train_divided, X_val, y_train_divided, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_divided_scaled = scaler.fit_transform(X_train_divided)
    X_val_scaled = scaler.transform(X_val)

    # === DECISION TREE ===
    accuracy_DT, par_DT = [], []
    for params in ParameterGrid(param_DT):
        dt = DecisionTreeClassifier(criterion="gini", max_depth=params['max_depth'], random_state=42)
        dt.fit(X_train_divided_scaled, y_train_divided)
        accuracy_DT.append(accuracy_score(y_val, dt.predict(X_val_scaled)))
        par_DT.append(params)
        
    best_param_dt = par_DT[accuracy_DT.index(max(accuracy_DT))]
    best_dt = DecisionTreeClassifier(criterion="gini", max_depth=best_param_dt['max_depth'], random_state=42)
    y_pred_DT, y_proba_DT, _ = train_model_with_timing(best_dt, X_train_scaled, y_train, X_test_scaled)
    metrics_per_fold['Decision Tree'].append(analyse_metrics(y_pred_DT, y_proba_DT, y_test))

    # === RANDOM FOREST ===
    accuracy_RF, par_RF = [], []
    for params in ParameterGrid(param_RF):
        rf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], criterion=params['criterion'], random_state=42, n_jobs=-1)
        rf.fit(X_train_divided_scaled, y_train_divided)
        accuracy_RF.append(accuracy_score(y_val, rf.predict(X_val_scaled)))
        par_RF.append(params)
        
    best_param_rf = par_RF[accuracy_RF.index(max(accuracy_RF))]
    best_rf = RandomForestClassifier(n_estimators=best_param_rf['n_estimators'], max_depth=best_param_rf['max_depth'], criterion=best_param_rf['criterion'], random_state=42, n_jobs=-1)
    y_pred_RF, y_proba_RF, _ = train_model_with_timing(best_rf, X_train_scaled, y_train, X_test_scaled)
    metrics_per_fold['Random Forest'].append(analyse_metrics(y_pred_RF, y_proba_RF, y_test))
    
    # === LOGISTIC REGRESSION ===
    accuracy_LR, par_LR = [], []
    for params in ParameterGrid(param_LR):
        lr = LogisticRegression(C=params['C'], penalty=params['penalty'], solver=params['solver'], random_state=42, n_jobs=-1)
        lr.fit(X_train_divided_scaled, y_train_divided)
        accuracy_LR.append(accuracy_score(y_val, lr.predict(X_val_scaled)))
        par_LR.append(params)
        
    best_param_lr = par_LR[accuracy_LR.index(max(accuracy_LR))]
    best_lr = LogisticRegression(C=best_param_lr['C'], penalty=best_param_lr['penalty'], solver=best_param_lr['solver'], random_state=42, n_jobs=-1)
    y_pred_LR, y_proba_LR, _ = train_model_with_timing(best_lr, X_train_scaled, y_train, X_test_scaled)
    metrics_per_fold['Logistic Regression'].append(analyse_metrics(y_pred_LR, y_proba_LR, y_test))
    
    # === XGBOOST ===
    accuracy_XGB, par_XGB = [], []
    for params in ParameterGrid(param_XGB):
        xgb = XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], random_state=42, n_jobs=-1)
        xgb.fit(X_train_divided_scaled, y_train_divided)
        accuracy_XGB.append(accuracy_score(y_val, xgb.predict(X_val_scaled)))
        par_XGB.append(params)
        
    best_param_xgb = par_XGB[accuracy_XGB.index(max(accuracy_XGB))]
    best_xgb = XGBClassifier(n_estimators=best_param_xgb['n_estimators'], max_depth=best_param_xgb['max_depth'], learning_rate=best_param_xgb['learning_rate'], random_state=42, n_jobs=-1)
    y_pred_XGB, y_proba_XGB, _ = train_model_with_timing(best_xgb, X_train_scaled, y_train, X_test_scaled)
    metrics_per_fold['XGBoost'].append(analyse_metrics(y_pred_XGB, y_proba_XGB, y_test))

    X_train_final = X_train_scaled
    X_test_final = X_test_scaled
    y_test_final = y_test
    fold += 1

# =============================================================================
# 4. GERAÇÃO DE EXPLICABILIDADE (XAI)
# =============================================================================

print(f"\n-> Gerando explicações XAI (SHAP e LIME) para o cenário RAW...")

base_output_dir = '../outputs/RAW_Outputs'
X_test_df = pd.DataFrame(X_test_final, columns=features_names_final)
X_test_sample = X_test_df.sample(n=min(300, len(X_test_df)), random_state=42)

modelos_obj = {
    'DT': best_dt,
    'RF': best_rf,
    'LR': best_lr,
    'XGB': best_xgb
}

for sigla, modelo in modelos_obj.items():
    dir_shap = f'{base_output_dir}/plots_SHAP/{sigla}'
    dir_lime = f'{base_output_dir}/plots_LIME'
    os.makedirs(dir_shap, exist_ok=True)
    os.makedirs(dir_lime, exist_ok=True)
    
    # --- SHAP ---
    if sigla in ['DT', 'RF', 'XGB']:
        explainer = shap.TreeExplainer(modelo, feature_names=features_names_final)
        shap_values = explainer(X_test_sample)
        
        try:
            if len(shap_values.shape) > 2:
                for c_idx in range(2):
                    shap.plots.waterfall(shap_values[2, :, c_idx], show=False)
                    plt.savefig(f'{dir_shap}/waterfall_class{c_idx}_{sigla}.png', bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    shap.plots.beeswarm(shap_values[:, :, c_idx], show=False)
                    plt.savefig(f'{dir_shap}/beeswarm_class{c_idx}_{sigla}.png', bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    shap.plots.bar(shap_values[:, :, c_idx], show=False)
                    plt.savefig(f'{dir_shap}/bar_class{c_idx}_{sigla}.png', bbox_inches='tight', dpi=300)
                    plt.close()
            else:
                shap.plots.waterfall(shap_values[2], show=False)
                plt.savefig(f'{dir_shap}/waterfall_{sigla}.png', bbox_inches='tight', dpi=300)
                plt.close()

                shap.plots.beeswarm(shap_values, show=False)
                plt.savefig(f'{dir_shap}/beeswarm_{sigla}.png', bbox_inches='tight', dpi=300)
                plt.close()

                shap.plots.bar(shap_values, show=False)
                plt.savefig(f'{dir_shap}/bar_{sigla}.png', bbox_inches='tight', dpi=300)
                plt.close()
        except Exception as e:
            print(f"Erro ao plotar SHAP para {sigla}: {e}")
            
    elif sigla == 'LR':
        explainer = shap.LinearExplainer(modelo, X_train_final)
        shap_values = explainer(X_test_sample)
        try:
            shap.plots.waterfall(shap_values[2], show=False)
            plt.savefig(f'{dir_shap}/waterfall_{sigla}.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            shap.plots.beeswarm(shap_values, show=False)
            plt.savefig(f'{dir_shap}/beeswarm_{sigla}.png', bbox_inches='tight', dpi=300)
            plt.close()
        except:
            pass

    # --- LIME ---
    try:
        lime_explainer = lime_tabular.LimeTabularExplainer(X_train_final, feature_names=list(features_names_final), verbose=False, mode='classification')
        explanation = lime_explainer.explain_instance(
            data_row=X_test_final[0],
            predict_fn=modelo.predict_proba,
            num_features=5
        )
        explanation.save_to_file(f'{dir_lime}/lime_{sigla}_inst0.html')
    except Exception as e:
        print(f"Erro LIME em {sigla}: {e}")

# =============================================================================
# 5. RESULTADOS GLOBAIS DE PERFORMANCE (LATEX)
# =============================================================================

from IPython.display import display

resultados_performance_globais = []
for m_name in metrics_per_fold.keys():
    df_pm = pd.DataFrame(metrics_per_fold[m_name])
    avg_pm = df_pm.mean(skipna=True).to_dict()
    avg_pm['Model'] = m_name
    resultados_performance_globais.append(avg_pm)

df_performance = pd.DataFrame(resultados_performance_globais)
cols_perf = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
df_performance = df_performance[cols_perf]

os.makedirs('../outputs/RAW_Outputs', exist_ok=True)
df_performance.to_csv('../outputs/RAW_Outputs/raw_metrics_global.csv', index=False)

resultados_folds = []
for m_name, folds in metrics_per_fold.items():
    for i, fold_metrics in enumerate(folds):
        row = {"Model": m_name, "Fold": i + 1}
        row.update(fold_metrics)
        resultados_folds.append(row)
        
pd.DataFrame(resultados_folds).to_csv('../outputs/RAW_Outputs/raw_metrics_per_fold.csv', index=False)

print("\n=== TABELA DE PERFORMANCE (CENÁRIO RAW) ===")
display(df_performance.round(4))

print("\n=== CÓDIGO LATEX ===")
print(df_performance.round(4).to_latex(index=False, escape=False, float_format="%.4f"))

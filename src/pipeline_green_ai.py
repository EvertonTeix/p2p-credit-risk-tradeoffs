import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from codecarbon import EmissionsTracker

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
# FUNÇÕES AUXILIARES
# =============================================================================

def analyse_metrics(y_predict, y_proba, y_test):
    return {
        "Accuracy": accuracy_score(y_test, y_predict),
        "F1 Score" : f1_score(y_test, y_predict),
        "Precision": precision_score(y_test, y_predict, zero_division=0),
        "Recall": recall_score(y_test, y_predict, zero_division=0),
        "AUC": roc_auc_score(y_test, y_proba)
    }

# =============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO (IDÊNTICO AOS DEMAIS PILARES)
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

# =============================================================================
# 2. K-FOLD, GRID SEARCH E CODECARBON
# =============================================================================

param_DT = {'max_depth': range(1, 11)}
param_LR = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
param_RF = {'n_estimators': [100], 'max_depth': [5, 10], 'criterion': ['gini']} 
param_XGB = {'n_estimators': [50, 100],'max_depth': [3, 5, 7],'learning_rate': [0.1, 0.2]}

os.makedirs('../outputs/GreenAI_Outputs', exist_ok=True)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

resultados = []

fold = 1
for train_index, test_index in kfold.split(X):
    print(f"\n[{'='*20} Rodando KFold {fold}/5 {'='*20}]")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    X_train_divided, X_val, y_train_divided, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_divided_scaled = scaler.fit_transform(X_train_divided)
    X_val_scaled = scaler.transform(X_val)

    # === DECISION TREE ===
    print("Avaliação de Custo: Decision Tree (GridSearch + Fit)")
    tracker = EmissionsTracker(output_dir="../outputs/GreenAI_Outputs", log_level="error")
    tracker.start()
    
    accuracy_DT, par_DT = [], []
    for params in ParameterGrid(param_DT):
        dt = DecisionTreeClassifier(criterion="gini", max_depth=params['max_depth'], random_state=42)
        dt.fit(X_train_divided_scaled, y_train_divided)
        accuracy_DT.append(accuracy_score(y_val, dt.predict(X_val_scaled)))
        par_DT.append(params)
        
    best_param_dt = par_DT[accuracy_DT.index(max(accuracy_DT))]
    best_dt = DecisionTreeClassifier(criterion="gini", max_depth=best_param_dt['max_depth'], random_state=42)
    best_dt.fit(X_train_scaled, y_train)
    
    tracker.stop()
    y_pred_DT = best_dt.predict(X_test_scaled)
    y_proba_DT = best_dt.predict_proba(X_test_scaled)[:, 1]
    
    row_dt = {
        "Model": "Decision Tree",
        "Fold": fold,
        "Energy Consumption (kWh)": tracker._total_energy.kWh,
        "Carbon Emissions (kg CO2)": tracker.final_emissions,
        "CPU Usage (kWh)": tracker._total_cpu_energy.kWh,
        "Memory Usage (kWh)": tracker._total_ram_energy.kWh
    }
    row_dt.update(analyse_metrics(y_pred_DT, y_proba_DT, y_test))
    resultados.append(row_dt)

    # === RANDOM FOREST ===
    print("Avaliação de Custo: Random Forest (GridSearch + Fit)")
    tracker = EmissionsTracker(output_dir="../outputs/GreenAI_Outputs", log_level="error")
    tracker.start()
    
    accuracy_RF, par_RF = [], []
    for params in ParameterGrid(param_RF):
        rf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], criterion=params['criterion'], random_state=42, n_jobs=-1)
        rf.fit(X_train_divided_scaled, y_train_divided)
        accuracy_RF.append(accuracy_score(y_val, rf.predict(X_val_scaled)))
        par_RF.append(params)
        
    best_param_rf = par_RF[accuracy_RF.index(max(accuracy_RF))]
    best_rf = RandomForestClassifier(n_estimators=best_param_rf['n_estimators'], max_depth=best_param_rf['max_depth'], criterion=best_param_rf['criterion'], random_state=42, n_jobs=-1)
    best_rf.fit(X_train_scaled, y_train)
    
    tracker.stop()
    y_pred_RF = best_rf.predict(X_test_scaled)
    y_proba_RF = best_rf.predict_proba(X_test_scaled)[:, 1]
    
    row_rf = {
        "Model": "Random Forest",
        "Fold": fold,
        "Energy Consumption (kWh)": tracker._total_energy.kWh,
        "Carbon Emissions (kg CO2)": tracker.final_emissions,
        "CPU Usage (kWh)": tracker._total_cpu_energy.kWh,
        "Memory Usage (kWh)": tracker._total_ram_energy.kWh
    }
    row_rf.update(analyse_metrics(y_pred_RF, y_proba_RF, y_test))
    resultados.append(row_rf)
    
    # === LOGISTIC REGRESSION ===
    print("Avaliação de Custo: Logistic Regression (GridSearch + Fit)")
    tracker = EmissionsTracker(output_dir="../outputs/GreenAI_Outputs", log_level="error")
    tracker.start()
    
    accuracy_LR, par_LR = [], []
    for params in ParameterGrid(param_LR):
        lr = LogisticRegression(C=params['C'], penalty=params['penalty'], solver=params['solver'], random_state=42, n_jobs=-1)
        lr.fit(X_train_divided_scaled, y_train_divided)
        accuracy_LR.append(accuracy_score(y_val, lr.predict(X_val_scaled)))
        par_LR.append(params)
        
    best_param_lr = par_LR[accuracy_LR.index(max(accuracy_LR))]
    best_lr = LogisticRegression(C=best_param_lr['C'], penalty=best_param_lr['penalty'], solver=best_param_lr['solver'], random_state=42, n_jobs=-1)
    best_lr.fit(X_train_scaled, y_train)
    
    tracker.stop()
    y_pred_LR = best_lr.predict(X_test_scaled)
    y_proba_LR = best_lr.predict_proba(X_test_scaled)[:, 1]
    
    row_lr = {
        "Model": "Logistic Regression",
        "Fold": fold,
        "Energy Consumption (kWh)": tracker._total_energy.kWh,
        "Carbon Emissions (kg CO2)": tracker.final_emissions,
        "CPU Usage (kWh)": tracker._total_cpu_energy.kWh,
        "Memory Usage (kWh)": tracker._total_ram_energy.kWh
    }
    row_lr.update(analyse_metrics(y_pred_LR, y_proba_LR, y_test))
    resultados.append(row_lr)
    
    # === XGBOOST ===
    print("Avaliação de Custo: XGBoost (GridSearch + Fit)")
    tracker = EmissionsTracker(output_dir="../outputs/GreenAI_Outputs", log_level="error")
    tracker.start()
    
    accuracy_XGB, par_XGB = [], []
    for params in ParameterGrid(param_XGB):
        xgb = XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], random_state=42, n_jobs=-1)
        xgb.fit(X_train_divided_scaled, y_train_divided)
        accuracy_XGB.append(accuracy_score(y_val, xgb.predict(X_val_scaled)))
        par_XGB.append(params)
        
    best_param_xgb = par_XGB[accuracy_XGB.index(max(accuracy_XGB))]
    best_xgb = XGBClassifier(n_estimators=best_param_xgb['n_estimators'], max_depth=best_param_xgb['max_depth'], learning_rate=best_param_xgb['learning_rate'], random_state=42, n_jobs=-1)
    best_xgb.fit(X_train_scaled, y_train)
    
    tracker.stop()
    y_pred_XGB = best_xgb.predict(X_test_scaled)
    y_proba_XGB = best_xgb.predict_proba(X_test_scaled)[:, 1]
    
    row_xgb = {
        "Model": "XGBoost",
        "Fold": fold,
        "Energy Consumption (kWh)": tracker._total_energy.kWh,
        "Carbon Emissions (kg CO2)": tracker.final_emissions,
        "CPU Usage (kWh)": tracker._total_cpu_energy.kWh,
        "Memory Usage (kWh)": tracker._total_ram_energy.kWh
    }
    row_xgb.update(analyse_metrics(y_pred_XGB, y_proba_XGB, y_test))
    resultados.append(row_xgb)

    fold += 1

# =============================================================================
# 3. CONSOLIDAÇÃO E EXPORTAÇÃO (GRÁFICOS E CSV)
# =============================================================================

df_resultados = pd.DataFrame(resultados)

ranking_total = df_resultados.groupby("Model").agg({
    "Energy Consumption (kWh)": "sum",
    "Carbon Emissions (kg CO2)": "sum",
    "Accuracy": "mean",
    "F1 Score": "mean",
    "Precision": "mean",
    "Recall": "mean",
    "AUC": "mean"
}).sort_values(by="Energy Consumption (kWh)", ascending=True)

print("\n" + "="*50)
print("CUSTO COMPUTACIONAL E EMISSÕES TOTAIS POR MODELO")
print("="*50)
print(ranking_total)

df_resultados.to_csv("../outputs/GreenAI_Outputs/resultados_folds.csv", index=False)
ranking_total.to_csv("../outputs/GreenAI_Outputs/resultados_totais.csv")

# Plot: Custo Total (Sum) vs Accuracy (Mean)
models = ranking_total.index
energy_consumption = ranking_total["Energy Consumption (kWh)"]
carbon_emissions = ranking_total["Carbon Emissions (kg CO2)"]
accuracy = ranking_total["Accuracy"]

fig, ax1 = plt.subplots(figsize=(10, 6))

width = 0.35
x = np.arange(len(models))
ax1.bar(x - width/2, energy_consumption, width, label='Total Energy (kWh)', color='#5896bb')
ax1.bar(x + width/2, carbon_emissions, width, label='Total Carbon (kg CO₂)', color='#747474')
ax1.set_ylabel('Total Consumption / Emissions (Sum across Folds)', fontsize=12)
ax1.set_ylim(0, max(max(energy_consumption), max(carbon_emissions)) * 1.2)

ax2 = ax1.twinx()
ax2.plot(x, accuracy, 'o-', color='green', label='Mean Accuracy', markersize=8)
ax2.set_ylabel('Accuracy', fontsize=12)
# Ajusta o eixo de acurácia de forma mais inteligente
ax2.set_ylim(min(accuracy)*0.95, max(accuracy)*1.05)

ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_title('Green AI: Total Computational Cost vs Mean Accuracy', fontweight='bold')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
os.makedirs("../outputs/GreenAI_Outputs", exist_ok=True)
plt.savefig("../outputs/GreenAI_Outputs/Model_Comparison_GreenAI_Total.png", dpi=300, bbox_inches='tight')
print("\nGráfico salvo em '../outputs/GreenAI_Outputs/Model_Comparison_GreenAI_Total.png'")

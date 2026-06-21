import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARREGAMENTO DOS DADOS (DOS 3 PILARES)
# =============================================================================

print("Carregando resultados exportados dos três pilares...")

try:
    df_raw = pd.read_csv('../outputs/RAW_Outputs/raw_metrics_global.csv')
    df_green = pd.read_csv('../outputs/GreenAI_Outputs/resultados_totais.csv')
    df_dir = pd.read_csv('../outputs/DIR_Outputs/dir_fairness_global.csv')
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos CSV: {e}")
    print("Certifique-se de que os três scripts base (raw, green_ai, completo) já rodaram com sucesso.")
    exit()

# =============================================================================
# 2. PRÉ-PROCESSAMENTO: EXTRAÇÃO DE CRITÉRIOS
# =============================================================================

# --- RAW (PERFORMANCE) ---
# Usaremos F1 Score e AUC como Benefícios (quanto maior, melhor)
df_raw = df_raw[['Model', 'F1 Score', 'AUC']]

# --- GREEN AI (SUSTENTABILIDADE) ---
# Usaremos o Consumo de Energia (kWh) e Emissões de Carbono como Custos
df_green = df_green[['Model', 'Energy Consumption (kWh)', 'Carbon Emissions (kg CO2)']]

# --- DIR (FAIRNESS / JUSTIÇA) ---
# Focaremos nas análises do modelo totalmente mitigado (Cenário "Todas")
df_dir_todas = df_dir[df_dir['Feature'].str.startswith('All')].copy()

# O Disparate Impact Ratio (DIR) ideal é 1. Vamos calcular a distância absoluta até 1.
df_dir_todas['Disparate Impact Ratio'] = pd.to_numeric(df_dir_todas['Disparate Impact Ratio'], errors='coerce').fillna(1)
df_dir_todas['DIR_Distance'] = abs(1.0 - df_dir_todas['Disparate Impact Ratio'])

# O Statistical Parity Diff (SPD) ideal é 0. Vamos calcular a distância absoluta.
df_dir_todas['SPD_Distance'] = abs(df_dir_todas['Statistical Parity Diff'])

# Tira a média das distâncias de injustiça sobre as 4 features
df_dir_grouped = df_dir_todas.groupby('Model')[['DIR_Distance', 'SPD_Distance']].mean().reset_index()


# =============================================================================
# 3. CONSTRUÇÃO DA MATRIZ DE DECISÃO
# =============================================================================

df_topsis = df_raw.merge(df_green, on='Model').merge(df_dir_grouped, on='Model')

# Configuração Matemática do TOPSIS
# Pesos: 20% RAW (10% F1, 10% AUC) | 40% Green AI (20% Energia, 20% CO2) | 40% DIR (20% DIR_Dist, 20% SPD_Dist)
criterios = ['F1 Score', 'AUC', 'Energy Consumption (kWh)', 'Carbon Emissions (kg CO2)', 'DIR_Distance', 'SPD_Distance']
pesos = np.array([0.10, 0.10, 0.20, 0.20, 0.20, 0.20])

# Tipos de critério: 'max' (Benefício) ou 'min' (Custo)
tipos = ['max', 'max', 'min', 'min', 'min', 'min']

print("\n--- MATRIZ DE DECISÃO ORIGINAL ---")
print(df_topsis)

# =============================================================================
# 4. APLICAÇÃO DO ALGORITMO TOPSIS
# =============================================================================

matriz = df_topsis[criterios].values

# 4.1 Normalização por Vetorização (Euclidiana)
norm_div = np.sqrt((matriz ** 2).sum(axis=0))
# Evitar divisão por zero caso a coluna seja toda zero
norm_div[norm_div == 0] = 1 
matriz_norm = matriz / norm_div

# 4.2 Matriz Ponderada
matriz_pond = matriz_norm * pesos

# 4.3 Soluções Ideais Positiva (ideal_pos) e Negativa (ideal_neg)
ideal_pos = np.zeros(len(criterios))
ideal_neg = np.zeros(len(criterios))

for i, tipo in enumerate(tipos):
    if tipo == 'max':
        ideal_pos[i] = np.max(matriz_pond[:, i])
        ideal_neg[i] = np.min(matriz_pond[:, i])
    else:
        ideal_pos[i] = np.min(matriz_pond[:, i])
        ideal_neg[i] = np.max(matriz_pond[:, i])

# 4.4 Cálculo das Distâncias Euclidianas aos Ideais
dist_pos = np.sqrt(((matriz_pond - ideal_pos)**2).sum(axis=1))
dist_neg = np.sqrt(((matriz_pond - ideal_neg)**2).sum(axis=1))

# 4.5 Coeficiente de Proximidade Relativa (Closeness / TOPSIS Score)
score = dist_neg / (dist_pos + dist_neg)
df_topsis['TOPSIS Score'] = score

# =============================================================================
# 5. RANQUEAMENTO E EXPORTAÇÃO
# =============================================================================

df_topsis['Ranking'] = df_topsis['TOPSIS Score'].rank(ascending=False).astype(int)
df_topsis = df_topsis.sort_values(by='TOPSIS Score', ascending=False)

print("\n--- RESULTADO FINAL DO TOPSIS ---")
cols_exibicao = ['Ranking', 'Model', 'TOPSIS Score', 'F1 Score', 'Energy Consumption (kWh)', 'Carbon Emissions (kg CO2)', 'DIR_Distance']
print(df_topsis[cols_exibicao].round(4))

os.makedirs('../outputs/TOPSIS_Outputs', exist_ok=True)
df_topsis.to_csv('../outputs/TOPSIS_Outputs/topsis_ranking.csv', index=False)

# --- GRÁFICO DO RANKING ---
plt.figure(figsize=(10, 6))
sns.barplot(x='TOPSIS Score', y='Model', data=df_topsis, palette='viridis')
plt.title('TOPSIS Ranking (Trade-off: Performance vs Sustainability vs Fairness)', fontweight='bold', fontsize=14)
plt.xlabel('TOPSIS Score (Closeness to Ideal)', fontsize=12)
plt.ylabel('Model', fontsize=12)

# Adicionar rótulos numéricos nas barras
for index, value in enumerate(df_topsis['TOPSIS Score']):
    plt.text(value, index, f' {value:.4f}', va='center', fontsize=11, fontweight='bold')

plt.xlim(0, 1.0)
plt.tight_layout()
plt.savefig('../outputs/TOPSIS_Outputs/TOPSIS_Ranking_Barplot.png', dpi=300, bbox_inches='tight')
print("\nGráfico exportado para '../outputs/TOPSIS_Outputs/TOPSIS_Ranking_Barplot.png'")

print("\n=== CÓDIGO LATEX DA TABELA FINAL ===")
print(df_topsis[['Ranking', 'Model', 'TOPSIS Score']].round(4).to_latex(index=False))

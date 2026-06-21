# Trade-off de Pilares em Machine Learning: Performance vs. Sustentabilidade vs. Justiça

Este repositório contém a pesquisa e o código-fonte desenvolvidos para avaliar o *trade-off* multidimensional em algoritmos de Machine Learning aplicados à concessão de crédito. O objetivo principal é desafiar o paradigma de "Performance Preditiva a qualquer custo", introduzindo as restrições éticas de justiça algorítmica (**Fairness/DIR**) e eficiência ambiental (**Green AI**).

Para resolver a escolha do algoritmo ideal dentro dessas três dimensões simultaneamente, empregamos o método de Análise de Decisão Multicritério **TOPSIS**.

---

## 📁 Estrutura do Repositório

O projeto segue um padrão rigoroso de arquitetura de software para Data Science, isolando dados, códigos e relatórios:

```text
trade_off_pilares/
├── data/                       # Arquivos de dados brutos
│   └── LoanData_Bondora.csv    # (Dataset pesado, ignorado no GitHub)
├── src/                        # Códigos-fonte
│   ├── pipeline_raw.py         # Treinamento Base e Performance
│   ├── pipeline_green_ai.py    # Rastreamento Energético
│   ├── pipeline_dir.py         # Mitigação de Viés e Fairness
│   └── topsis_analysis.py      # Cruzamento Multicritério
├── outputs/                    # Resultados gerados automaticamente
│   ├── RAW_Outputs/            # CSVs de métricas e gráficos XAI (SHAP/LIME) 
│   ├── GreenAI_Outputs/        # Consumo de energia e emissões de carbono
│   ├── DIR_Outputs/            # Métricas de Fairness e XAI do modelo mitigado
│   └── TOPSIS_Outputs/         # Ranking de Trade-off matemático definitivo
├── gerar_tabelas.ipynb         # Notebook para geração de gráficos e tabelas LaTeX para o artigo
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🏛️ A Arquitetura dos 3 Pilares

A arquitetura do projeto foi dividida de forma altamente modular em 3 pipelines independentes, mas que compartilham o exato mesmo fluxo de pré-processamento para garantir a integridade comparativa:

1. **Pilar 1: Performance Base (RAW)**
   - Script: `pipeline_raw.py`
   - **Objetivo**: Treinar 4 algoritmos base (Decision Tree, Random Forest, Logistic Regression e XGBoost) usando *5-Fold Cross Validation* e *GridSearch*. Extrai as métricas tradicionais (Acurácia, F1-Score, AUC, Precision, Recall) e gera explicações visuais via *XAI (LIME e SHAP)*.

2. **Pilar 2: Sustentabilidade (Green AI)**
   - Script: `pipeline_green_ai.py`
   - **Objetivo**: Isolar o rastreamento computacional através da biblioteca `codecarbon`. Envelopa o processo de validação e treinamento de ponta a ponta, quantificando o consumo energético total (kWh) e as emissões de $CO_2$.

3. **Pilar 3: Justiça e Mitigação (Fairness / DIR)**
   - Script: `pipeline_dir.py`
   - **Objetivo**: Utilizar o *Disparate Impact Remover (DIR)* da biblioteca `AIF360` para mitigar os dados contra preconceitos históricos (Gênero, Idade, Estado Civil e Educação). Calcula as métricas de justiça (*Disparate Impact Ratio*, *Statistical Parity Difference*, etc.) avaliando cenários individuais e um cenário robusto ("Todas" as features mitigadas simultaneamente).

---

## 🔍 Transparência Algorítmica (XAI)

Um diferencial crítico desta pesquisa é a aplicação de técnicas de *eXplainable AI (XAI)* para quebrar a natureza "caixa-preta" dos algoritmos de crédito. O sistema gera automaticamente as explicações visuais comparando os modelos ANTES (Pilar 1) e DEPOIS da mitigação de viés (Pilar 3):
- **SHAP (SHapley Additive exPlanations)**: Análises globais e locais das distribuições de decisão, gerando gráficos de *Waterfall*, *Beeswarm* e *Bar* para avaliar como cada feature protegida puxou a predição.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Explicações a nível de instância, mostrando os limiares exatos que aprovaram ou reprovaram o empréstimo para clientes específicos.

Os artefatos gráficos gerados auxiliam não apenas na compreensão de *por que* o algoritmo tomou uma decisão, mas provam visualmente os efeitos da mitigação do DIR sobre os padrões de decisão do banco.

---

## ⚖️ A Resolução: TOPSIS Analysis

Após a execução dos três pipelines, o script `topsis_analysis.py` coleta os resultados sumarizados gerados em formato CSV e aplica o método matemático TOPSIS. 

As dimensões são normalizadas e cruzadas de acordo com a prioridade (pesos):
*   **40% Green AI** (Custos: Minimizar Consumo de Energia - 20% e Emissões de $CO_2$ - 20%)
*   **40% Justiça** (Custos: Minimizar a distância absoluta de injustiça para as métricas DIR - 20% e SPD - 20%)
*   **20% Performance** (Benefícios: Maximizar F1-Score - 10% e AUC - 10%)

O resultado final é exportado como um ranking (`topsis_ranking.csv`) demonstrando qual algoritmo possui o melhor equilíbrio real para ir para produção.

---

## 🚀 Instalação e Uso

### Pré-requisitos
Certifique-se de ter o Python (versão 3.8 a 3.10 recomendada) instalado na sua máquina. O dataset base (`LoanData_Bondora.csv`) deve ser baixado do Kaggle e colocado na raiz do projeto:
- 🔗 **Link do Dataset:** [Bondora Peer-to-Peer Lending Loan Data](https://www.kaggle.com/datasets/sid321axn/bondora-peer-to-peer-lending-loan-data)

*(Aviso: devido ao seu tamanho de 149MB, o dataset é ignorado no repositório GitHub via `.gitignore`).*

### 1. Criar e Ativar Ambiente Virtual
Recomendamos o uso do `venv` para não conflitar bibliotecas pesadas como AIF360 e Shap.
```bash
python -m venv .venv

# Windows (Powershell)
.venv\Scripts\Activate.ps1

# Linux / Mac
source .venv/bin/activate
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 3. Ordem de Execução
Para garantir que o TOPSIS capte os resultados mais recentes, execute os scripts a partir da raiz do projeto usando o prefixo `src/`:

```bash
# 1. Gera o Baseline de Performance
python src/pipeline_raw.py

# 2. Rastreia o custo Energético
python src/pipeline_green_ai.py

# 3. Mitiga e calcula as métricas de Justiça
python src/pipeline_dir.py

# 4. Compila os 3 Pilares e calcula o Vencedor
python src/topsis_analysis.py
```

Os resultados analíticos finais, bem como tabelas para LaTeX e gráficos cruzados, serão salvos automaticamente na pasta `outputs/`, subdivididos em `RAW_Outputs/`, `GreenAI_Outputs/`, `DIR_Outputs/` e `TOPSIS_Outputs/`.

---

## ✒️ Autoria e Contato

**Everton Teixeira**  
Bacharelado em Ciência da Computação - Universidade Federal do Ceará (UFC)  
📧 Contato: [tteverton75@gmail.com](mailto:tteverton75@gmail.com)

---

## 📄 Licença

Este projeto está licenciado sob a **Licença MIT** - veja o arquivo [LICENSE](LICENSE) para mais detalhes. Isso significa que o código é de código aberto (Open Source) e pode ser livremente utilizado, modificado e distribuído para fins acadêmicos e comerciais, desde que a autoria seja mantida.

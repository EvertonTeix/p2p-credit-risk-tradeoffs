# 🌍 P2P Credit Risk Trade-offs  
### Balancing Accuracy, Fairness, Explainability and Sustainability in P2P Lending

---

## 📌 Overview

This repository accompanies the research study:

> **"Balancing Accuracy, Fairness, Explainability, and Sustainability in Peer-to-Peer Credit Risk Assessment"**

In peer-to-peer (P2P) lending platforms, machine learning models are widely used to predict credit default. However, model selection is not purely a matter of predictive accuracy. Real-world deployment involves **multi-dimensional trade-offs** between:

- 🎯 Predictive Performance  
- ⚖️ Fairness (Bias Mitigation)  
- 🔍 Explainability  
- 🌱 Environmental Sustainability  

This project provides a **reproducible and governance-oriented evaluation protocol** to make these trade-offs explicit and comparable.

---

## 🎯 Research Objective

Rather than proposing a new algorithm, this study aims to:

- Benchmark widely used ML classifiers in P2P credit default prediction  
- Quantify fairness disparities across sensitive attributes  
- Measure explainability quality using stability and sparsity indicators  
- Estimate computational footprint and CO₂ emissions during training  
- Make the **accuracy–fairness–explainability–sustainability trade-off explicit**

---

## 🧠 Models Evaluated

We benchmark four commonly used classifiers:

- Decision Tree (DT)  
- Logistic Regression (LR)  
- Random Forest (RF)  
- XGBoost  

All models are evaluated under consistent preprocessing and cross-validation settings.

---

## 📊 Evaluation Dimensions

### 1️⃣ Predictive Performance
- AUC
- Accuracy
- Precision
- Recall
- F1-score

---

### 2️⃣ Fairness Analysis

Group-level evaluation using:

- Statistical Parity Difference  
- Disparate Impact  
- Equal Opportunity Gap  
- Error-rate Differences  

Sensitive attributes are evaluated independently to assess group disparities.

---

### 3️⃣ Explainability

Post-hoc interpretability methods:

- SHAP  
- LIME  

Quantitative indicators:

- Stability across folds  
- Sparsity of explanations  
- Feature importance consistency  

---

### 4️⃣ Sustainability

Environmental impact measured during training using:

- CodeCarbon  

Reported metrics:

- Energy consumption (kWh)  
- CO₂ emissions (g)  
- Emissions per fold  
- Emissions per unit of predictive gain  

This enables evaluation of the **marginal environmental cost per performance improvement**.

---

## 🌱 Why This Matters

Financial ML systems directly impact:

- Borrower access to credit  
- Platform financial stability  
- Environmental footprint of AI systems  
- Regulatory compliance  

This repository supports **responsible and sustainable ML deployment** in fintech.

---

## 📂 Dataset

This study uses the **Bondora Peer-to-Peer Lending Loan Data**, publicly available on Kaggle:

🔗 https://www.kaggle.com/datasets/sid321axn/bondora-peer-to-peer-lending-loan-data

Due to GitHub file size limitations, the dataset is **not included** in this repository.


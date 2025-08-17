# STAB - Predição de Churn com Redes Neurais
## Disciplina: Redes Neurais 2025.1  

---

## 📌 Descrição do Projeto
Este projeto tem como objetivo prever o **Churn de clientes** (saída ou permanência em uma base de clientes) utilizando **redes neurais baseadas em STab (Stochastic Attention for Tabular Data)**.  

A abordagem combina **atenção estocástica** para variáveis tabulares, técnicas de **balanceamento de classes** e **busca de hiperparâmetros com Optuna**.  

O modelo é avaliado com métricas tradicionais de classificação binária, como:
- **KS-Statistic**  
- **AUC-ROC**  
- **AUC-PR**  
- **F1-Score**  
- **Matriz de Confusão**

---
## ⚙️ Principais Funcionalidades

### 🔹 Pré-processamento dos dados
- Normalização de variáveis numéricas  
- Codificação de variáveis categóricas com **LabelEncoder** e **CatMap**  
- Balanceamento da base com **oversampling**

### 🔹 Treinamento do modelo STAB
- Configuração customizada (**dim, depth, heads, dropout, U, etc.**)  
- **Wrapper Num_Cat** para integração numérico/categórica  
- Treinamento com **keras4torch**

### 🔹 Avaliação do modelo
- Estatística **KS** e curva cumulativa  
- **AUC-ROC**, **AUC-PR** e **F1-Score**  
- Matriz de confusão

### 🔹 Hiperparâmetros com Optuna
- Busca automática para maximizar a estatística **KS**  
- Testa combinações de **dimensão, profundidade, dropout, taxa de aprendizado, etc.**

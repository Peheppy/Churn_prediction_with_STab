# Customer Churn Prediction with STab

## Neural Networks 2025.1  

This project performs prediction of customer churn using **STab (Stochastic Attention for Tabular Data)** implemented in **PyTorch** and **keras4torch**, with hyperparameter tuning using **Optuna**. The goal is to evaluate and optimize STab architectures for tabular classification tasks.

---

## 📂 Project Structure

- `customer_churn_dataset_clean(2).csv` : Preprocessed dataset with churn labels (`Churn`) and train/validation/test splits.  
- `saved/` : Folder containing saved model checkpoints (`savefileAD`)  
- `notebook.ipynb` : Main notebook with preprocessing, model definition, training, evaluation, and hyperparameter tuning.  

---

## 🛠 Technologies Used

- Python 3.x  
- Pandas, Numpy  
- Matplotlib, Seaborn  
- Scikit-learn (preprocessing, metrics, resampling)  
- PyTorch (STab model implementation, training, and evaluation)  
- keras4torch (Keras-like API wrapper for PyTorch models)  
- Optuna (hyperparameter optimization)  
- tqdm (progress bar)  
- einops, tab-transformer-pytorch  

---

## 🔍 Project Pipeline

1. **Data Preprocessing**  
   - Handling missing values in numerical features  
   - Normalization of numerical columns (`StandardScaler`)  
   - Encoding categorical columns (`LabelEncoder` + `CatMap`)  

2. **Dataset Splitting**  
   - Predefined splits: `train`, `validation`, `test`  

3. **Data Balancing**  
   - Oversampling minority class to balance the dataset  

4. **Model Definition**  
   - STab architecture with categorical embeddings and continuous features  
   - Wrapper `Num_Cat` for mixed numerical/categorical inputs  
   - Hyperparameters: embedding dimension, depth, number of attention heads, dropout rates, U and cases  

5. **Training**  
   - Optimizer: AdamW  
   - Loss function: `BCEWithLogitsLoss`  
   - Learning rate scheduler: `LinearLR`  
   - Callbacks: ModelCheckpoint, LRScheduler  

6. **Evaluation**  
   - Metrics: KS-statistic, AUC-ROC, F1 Score, AUC-PR, Confusion Matrix  
   - Visualization of cumulative distributions for KS  
   - Model evaluation on test set  

7. **Hyperparameter Optimization**  
   - Using Optuna to tune STab parameters (`dim`, `depth`, `heads`, `attn_dp`, `ff_dp`, `U`, `lr`, `weight_decay`)  
   - Objective: maximize KS-statistic on validation set  

---

## ⚙ Example STab Configuration

```python
stab_config = {
    "optim": optim.AdamW,
    "optim_lr": 1e-4,
    "optim_wd": 1e-5,
    "loss": nn.BCEWithLogitsLoss(),
    "dim": 32,
    "depth": 9,
    "heads": 8,
    "attn_dp": 0.1,
    "ff_dp": 0.1,
    "U": 4,
    "cases": 8,
    "verbose": 2,
    "epochs": 60,
    "bs": 64,
    "metrics": ["accuracy"]
}

# Medical Cost Prediction using Machine Learning

## 📌 Project Overview
This project predicts individual medical insurance costs using real-world data and machine learning techniques.  
It demonstrates a **complete end-to-end machine learning workflow**, following industry best practices such as:

- Proper data preprocessing
- Baseline and ensemble models
- Hyperparameter tuning with cross-validation
- Model explainability
- Model persistence and inference

.
---

## 🎯 Problem Statement
Medical insurance costs depend on several demographic and lifestyle factors such as age, BMI, and smoking status.  
The objective of this project is to build a regression model that accurately predicts medical charges while ensuring:

- No data leakage
- Fair model comparison
- Explainable predictions
- Reproducible results

---

## 📊 Dataset
- **Type**: Tabular data
- **Target Variable**: `charges` (medical cost)
- **Features**:
  - `age`
  - `sex`
  - `bmi`
  - `children`
  - `smoker`
  - `region`

The dataset is publicly available and commonly used for regression benchmarking.

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Loading
- Load raw CSV data
- Validate schema and structure

### 2️⃣ Preprocessing
- Numerical features: Standard scaling
- Categorical features: One-hot encoding
- Train/Test split (80/20)

### 3️⃣ Baseline Model
- **Linear Regression**
- Evaluation Metrics:
  - RMSE
  - R²

### 4️⃣ Ensemble Model
- **Random Forest Regressor**
- Captures:
  - Non-linear relationships
  - Feature interactions

### 5️⃣ Hyperparameter Tuning
- **Optuna** for hyperparameter optimization
- 5-fold cross-validation on training data
- Test set used only once (no circular analysis)

### 6️⃣ Model Explainability
- **SHAP (SHapley Additive exPlanations)**
- Identifies:
  - Feature importance
  - Directional impact of features on predictions

### 7️⃣ Model Persistence & Inference
- Trained pipeline saved using `joblib`
- Separate inference script for predictions on new data

### 8️⃣ REST API (FastAPI)
- `/predict` endpoint
- JSON input validation using Pydantic
- Interactive Swagger documentation
---

### 9️⃣ Input Validation
- Field constraints (age, BMI ranges)
- Categorical validation
- Automatic request rejection for invalid input

---

### 🔟 Unit Testing
- pytest-based model tests
- Ensures:
  - Model loads correctly
  - Predictions return valid output
  - No negative predictions

---

## 📈 Results (Final Tuned Model)

| Metric | Value |
|------|------|
| RMSE | ~4600 |
| R² | ~0.86 |

### Key Insights
- Smoking status is the strongest predictor of medical cost
- BMI and age have significant positive impact
- Random Forest outperforms Linear Regression by capturing non-linear patterns

---

## 🛠️ Tech Stack
- **Language**: Python
- **Libraries**:
  - pandas, numpy
  - scikit-learn
  - Optuna
  - SHAP
  - matplotlib
  - joblib
  - FastAPI
  - Pydantic
  - pytest

---

## 📂 Project Structure

medical-cost-prediction/
│
├── data/
│ └── raw/
│ └── insurance.csv
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── train.py
│ ├── tune.py
│ ├── evaluate.py
│ ├── explain.py
│ └── save_load.py
│
├── scripts/
│ ├── quick_test.py
│ └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate


### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 3️⃣ Train Model & Evaluate
```python
python -m scripts.quick_test

### 4️⃣ Run Inference on New Data
```python
python -m scripts.predict

### 5️⃣ Run API
```python
uvicorn api:app --reload
Open:  http://127.0.0.1:8000/docs

### 6️⃣ Run unit tests
```python
python -m pytest



📌 Author

Rigal Patel
Aspiring Machine Learning / AI Engineer




#  Bike Sharing Demand Forecast

A machine learning project that predicts the daily number of bike rentals using the Bike Sharing Dataset from the UCI Machine Learning Repository.

The project follows a complete machine learning workflow, including exploratory data analysis (EDA), baseline modeling, feature engineering, model training, time-series validation, and model evaluation.

---

##  Project Objectives

- Analyze factors affecting daily bike rental demand.
- Build a baseline forecasting model.
- Engineer time-series features.
- Compare multiple regression models.
- Evaluate models using proper time-series validation.

---

##  Dataset

**Dataset:** Bike Sharing Dataset

- Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Daily records from 2011 to 2012
- 731 observations

### Target variable

- `cnt` — Total daily bike rentals

### Main features

- Calendar information
  - season
  - year
  - month
  - weekday
  - holiday
  - workingday

- Weather information
  - temperature
  - feeling temperature
  - humidity
  - windspeed
  - weather situation

---

##  Project Structure

```
bike_sharing_forecast/
│
├── data/
│   ├── raw/
│   └── processed/
│   └── Data_README.txt
├── notebook/
│   ├── EDA.ipynb
│   ├── baseline.ipynb
│   ├── feature_en.ipynb
│   └── models.ipynb
│
│
├── README.md
└── requirements.txt
```

---

##  Exploratory Data Analysis

The exploratory analysis includes:

- Missing value inspection
- Descriptive statistics
- Distribution visualization
- Correlation heatmap
- Seasonal and temporal analysis
- Relationship between weather variables and bike rentals

Key findings:

- Temperature has a strong positive correlation with bike rentals.
- Humidity and windspeed show weak negative correlations.
- Bike demand exhibits clear yearly and seasonal trends.
- Weekly seasonality is relatively weak.

---

##  Baseline Models

Two simple forecasting baselines were implemented.

| Baseline | MAE | RMSE | R2 |
|----------|------:|------:|------:|
| Persistence (Yesterday) | **878.39** | **1282.32** | **0.53** |
| Seasonal (7 days ago) | 1194.73 | 1759.66 | 0.12 |

Persistence forecasting achieved the lower error and was selected as the baseline.

---

##  Feature Engineering

Additional time-series features were created:

- `cnt_lag1`
- `cnt_lag7`
- `cnt_roll7_mean`

Categorical variables were encoded using **One-Hot Encoding**.

---

##  Models

Three regression models were evaluated.

- Linear Regression
- Random Forest Regressor
- HistGradientBoosting Regressor

Time-based train/test split was used instead of random splitting.

---

##  Results

| Model | MAE | RMSE | R2 |
|------|------:|------:|------:|
| Persistence Baseline | 878.39 | 1282.32 | 0.53 |
| Linear Regression | **659.12** | **947.21** | **0.75** |
| Random Forest | 783.06 | 1035.08 | 0.70 |
| HistGradientBoosting | 850.74 | 1084.67 | 0.67 |

Linear Regression achieved the best overall performance, outperforming both the baseline and tree-based ensemble models.

---

##  Time-Series Cross Validation

Model robustness was evaluated using **TimeSeriesSplit**.

Average validation performance:

- Mean MAE: **791.17**
- Mean RMSE: **973.13**
- Mean R2: **0.04**

This validation strategy preserves temporal ordering and avoids data leakage.

---

##  Model Evaluation

Evaluation includes:

- Actual vs Predicted visualization
- Residual plot
- Residual distribution
- Linear Regression Coefficients
- Random Forest feature importance
- Model comparison table

---

##  Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

##  How to Run

Clone the repository.

```bash
git clone <repository-url>
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Run the notebooks in the following order:

1. EDA.ipynb
2. baseline.ipynb
3. feature_en.ipynb
4. models.ipynb

---

##  Future Improvements

- Hyperparameter optimization using GridSearchCV or Optuna
- XGBoost / LightGBM models
- More advanced rolling statistics
- Additional lag features
- Model deployment with Streamlit or FastAPI

---

##  Author

Đào Nhật Tân - dntan2524@clc.fitus.edu.vn




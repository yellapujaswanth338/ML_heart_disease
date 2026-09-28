# ML
# Heart Disease Prediction

A machine-learning project that predicts whether a patient has heart disease from 11 clinical features.
It compares **Logistic Regression**, **Random Forest** and **Gradient Boosting**, and also shows how to
tune the decision threshold to trade precision against recall.

## Dataset

- **File:** `data/heart.csv` (918 rows, 11 features + target)
- **Target:** `HeartDisease` (1 = heart disease, 0 = normal), roughly 55% / 45%
- **Source:** Heart Failure Prediction Dataset (Kaggle). Check the dataset's license before redistributing the CSV.

| Feature | Description |
|---|---|
| Age | Age in years |
| Sex | M / F |
| ChestPainType | TA, ATA, NAP, ASY |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl (1 / 0) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y / N) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Up, Flat, Down |

## Project structure

```
.
├── data/
│   └── heart.csv
├── heart_disease.ipynb
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

Run all cells from top to bottom. A fixed random seed (`42`) makes results reproducible.

## Methodology

1. **EDA:** missing values, duplicates, class balance, correlation heatmap.
2. **Cleaning:** `Cholesterol = 0` (172 rows) and `RestingBP = 0` (1 row) are impossible values, so they are treated as missing and median-imputed.
3. **Split:** 80/20 stratified train/test split.
4. **Preprocessing:** median imputation and scaling for numeric features, one-hot encoding for categorical features. This runs inside a scikit-learn `Pipeline`, so it is fitted on training data only (no leakage).
5. **Models:** Logistic Regression, Random Forest and Gradient Boosting, tuned with 5-fold `GridSearchCV`.
6. **Threshold tuning:** Youden's J, target-recall and target-precision thresholds, selected from out-of-fold training predictions and then applied to the test set.
7. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, log loss and confusion matrices.

## Possible improvements

- Repeated / nested cross-validation for more stable model comparison
- Feature importance and SHAP explanations
- Probability calibration
- Try XGBoost / LightGBM

## Disclaimer

This project is for learning purposes only and is **not** a medical diagnostic tool.

## License

MIT

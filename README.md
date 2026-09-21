# Student Performance — Machine Learning

Machine learning analysis of student performance using regression,
classification, ensemble models, feature selection, and neural networks.
The project predicts students' final grade (G3) in two subjects — **Math** and
**Portuguese** — framing it both as a regression problem (exact grade) and a
classification problem (performance level).

## Dataset

A merged dataset of **383 students** with **54 columns** (0 nulls, 0
duplicates), based on the UCI Student Performance data. It is split into two
subsets that share demographic/family features and add subject-specific ones:

- **Math** (`df_mat`)
- **Portuguese** (`df_por`)

**Targets:**
- **Regression** → `G3`, the final grade (0–20).
- **Classification** → `G3_cat`, a performance level derived from G3:
  - Bajo / Low (0–9), Medio / Medium (10–13), Alto / High (14–20).

Key features include prior grades (`G1`, `G2`), `failures`, `studytime`,
`absences`, alcohol consumption (`Dalc`, `Walc`), going out (`goout`), parental
education (`Medu`, `Fedu`), and school support (`schoolsup`).

## Methodology

1. **EDA** — distributions, outliers, correlation matrix, and grade breakdowns
   by gender, age, parental education, and school support.
2. **Preprocessing** — `LabelEncoder` for binary variables, one-hot encoding for
   nominal ones, and `StandardScaler` for models that need scaling.
3. **Feature selection** — four methods combined (ANOVA F-test filter, RFE
   wrapper, L1-Logistic embedded, and Random-Forest embedded). Variables selected
   by ≥3 methods formed the final subsets:
   - Math (10 features): `age, Medu, goout, Walc, G2, G1, failures, famrel, Dalc, schoolsup`
   - Portuguese (12 features): `school, age, G2, G1, higher, studytime, Medu, Dalc, freetime, internet, failures, schoolsup`
4. **Modeling** — 80/20 train/test split, `GridSearchCV`, and k-fold cross
   validation. Models span linear, tree-based, ensemble (bagging/boosting/voting),
   and neural networks (MLP).
5. **Evaluation** — R²/RMSE/MAE/MAPE for regression; accuracy/precision/recall/F1
   for classification, plus overfitting diagnostics.

## Results

### Regression (test set, target G3)

**Math** — best model: **Extra Trees**

| Model             | R²     | RMSE   | MAE    |
|-------------------|--------|--------|--------|
| Extra Trees       | 0.847  | 1.694  | 1.069  |
| Linear Regression | 0.837  | 1.747  | 1.287  |
| Gradient Boosting | 0.829  | 1.789  | 1.136  |

**Portuguese** — best model: **Bagged Trees**

| Model             | R²     | RMSE   | MAE    |
|-------------------|--------|--------|--------|
| Bagged Trees      | 0.664  | 1.937  | 0.933  |
| Linear Regression | 0.664  | 1.939  | 1.026  |
| LightGBM          | 0.569  | 2.195  | 1.113  |

### Classification (test set, target G3_cat)

**Math** — best model: **Decision Tree / AdaBoost** (tie)

| Model             | Accuracy | F1     |
|-------------------|----------|--------|
| Decision Tree     | 0.870    | 0.871  |
| AdaBoost          | 0.870    | 0.871  |
| Gradient Boosting | 0.857    | 0.855  |

**Portuguese** — best model: **Gradient Boosting**

| Model             | Accuracy | F1     |
|-------------------|----------|--------|
| Gradient Boosting | 0.896    | 0.894  |
| Voting            | 0.870    | 0.872  |
| Bagging           | 0.870    | 0.870  |

**Key insight:** across every model, the strongest predictors of the final grade
are the partial grades `G2` and `G1`, followed by `failures`. Ensemble methods
(Extra Trees, Bagged Trees, Gradient Boosting) consistently gave the best balance
of accuracy and generalization.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (regression, classification, ensembles, feature selection, MLP)
- XGBoost, LightGBM
- imbalanced-learn (SMOTE)
- TensorFlow / Keras (MLP regressor)
- matplotlib, seaborn

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/fernandoriosgz/student-performance-machine-learning.git
   cd student-performance-machine-learning

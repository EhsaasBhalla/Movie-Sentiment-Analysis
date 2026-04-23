# 🎬 Movie Review Sentiment Analysis

## 📋 Project Overview

This project builds a **machine learning pipeline** to classify movie review phrases into sentiment categories:

* **0 → Negative**
* **1 → Neutral**
* **2 → Positive**

The solution combines **text processing (TF-IDF)** with **numerical features** and evaluates multiple machine learning models.

---

## 📂 Dataset

* **Training Data**: Movie review phrases with sentiment labels
* **Test Data**: Unlabeled phrases for prediction
* **Features**:

  * `phrase` (text data)
  * `feature_1, feature_2, feature_3` (numeric)
  * Engineered features:

    * `review_length_chars`
    * `review_length_words`

---

## 🔍 Workflow

### 1. Data Exploration

* Checked dataset structure using `.info()` and `.describe()`
* Identified:

  * Missing values
  * Class imbalance
  * Feature distributions
* Visualizations:

  * Histograms
  * Boxplots
  * Correlation heatmap
  * Missing value heatmap

---

### 2. Data Preprocessing

* Filled missing text values with empty strings
* Removed duplicate rows
* Handled missing numeric values using **median imputation**
* Outliers treated using **IQR clipping**

---

### 3. Feature Engineering

Created additional features from text:

* Character length of review
* Word count of review

These features help models capture review size and structure.

---

### 4. Feature Encoding

#### 🔤 Text Features

* Used **TF-IDF Vectorizer**
* Character-level n-grams: `(3,5)`
* Max features: `200,000`

#### 🔢 Numeric Features

* Median imputation
* Standard scaling

#### 🔀 Combined using:

* `ColumnTransformer`

---

### 5. Train-Test Split

* 80% training, 20% validation
* Used **stratified sampling** to preserve class distribution

---

## 🤖 Models Trained

The following models were trained and evaluated:

* Logistic Regression (with GridSearchCV)
* Linear SVC (with GridSearchCV)
* Multinomial Naive Bayes (with GridSearchCV)
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier
* Support Vector Classifier (SVC)
* LightGBM Classifier

---

## ⚙️ Hyperparameter Tuning

Applied **GridSearchCV** on:

* Logistic Regression (Best Params: `C=1`, `penalty='l2'`)
* Linear SVC (Best Params: `C=0.1`)
* Multinomial Naive Bayes (Best Params: `alpha=0.1`)

---

## 📊 Evaluation Metrics

* **Accuracy Score**
* **F1 Score (Weighted)**

---

## 📈 Model Performance Results

After training all models, they were compared against the validation set:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **Logistic Regression** | **0.6393** | 0.5898 |
| Multinomial NB | 0.6350 | **0.5949** |
| Linear SVC | 0.6293 | 0.5708 |
| Random Forest (Expanded) | 0.6100 | 0.5535 |
| XGBoost | 0.6057 | 0.5710 |
| LightGBM | 0.6050 | 0.5741 |
| Gradient Boosting | 0.5736 | 0.5241 |
| SVC | 0.4243 | 0.2528 |

---

## 🏆 Model Selection

* All models were compared on validation data
* Best model selected based on **highest accuracy**: **Logistic Regression**
* The selected Logistic Regression model (Accuracy: 0.6393) was automatically chosen and retrained on the full dataset.

---

## 💡 Key Observations

* Text features contribute most to prediction power
* Numerical features alone are weak predictors
* Linear models (Logistic Regression, Linear SVC, Multinomial NB) perform particularly well on TF-IDF features and outperformed tree-based models like Random Forest and Gradient Boosting.
* Dataset is slightly imbalanced, which makes F1-score a pertinent supplemental metric.

---

## 📤 Final Output

* Generated predictions on test dataset
* Created submission file:

  ```
  submission.csv
  ```

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost, LightGBM
* Matplotlib, Seaborn

---

## 🚀 Conclusion

This project demonstrates an end-to-end ML pipeline for sentiment analysis using:

* Efficient text vectorization (TF-IDF)
* Multiple model comparisons
* Automated model selection yielding over 63% validation accuracy

The approach balances simplicity and performance, making it suitable for real-world NLP classification tasks.

https://www.kaggle.com/competitions/mlp-term-3-2025-kaggle-assignment-3/data

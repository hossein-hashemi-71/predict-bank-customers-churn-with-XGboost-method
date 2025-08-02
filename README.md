# predict-bank-customers-churn-with-XGboost-method
This project assigns labels to customers based on RFM scores and deposit-related features. After applying SMOTE for class balancing, XGBoost was used for prediction.
# 📊 Customer Churn Prediction Using RFM, SMOTE, and XGBoost

This project focuses on predicting customer churn by combining RFM scores with deposit-related features. The dataset is balanced using the SMOTE-Tomek method, and the final prediction is made using the XGBoost classification model.

---

## 🧠 Methodology

### 1. **Data Preprocessing**
- Remove customers with zero transaction/deposit behavior in each RFM segment.
- Apply logarithmic transformation to normalize the data.

### 2. **Labeling**
- Customers are labeled based on a combination of RFM scores and deposit behavior.

### 3. **Data Balancing**
- Use the **SMOTE-Tomek** technique to balance the dataset and reduce class bias.

### 4. **Modeling**
- Apply **XGBoost Classifier** to predict customer churn based on the processed features.

### 5. **Prediction**
- Predict churn for new customer data and export results to an Excel file.

---

## 🛠️ Tech Stack

- **Python** 3.x
- **Pandas**, **NumPy**
- **XGBoost**
- **Imbalanced-learn (SMOTE-Tomek)**
- **Scikit-learn**

---




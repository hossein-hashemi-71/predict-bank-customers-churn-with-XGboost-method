# Author: Hossein-Hashemi-71
# Description: Predict customer churn using RFM and deposit features with SMOTE-Tomek and XGBoost


import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_and_clean_data(filepath):
    df = pd.read_excel(filepath)
    df = df[df.iloc[:, 2:14].sum(axis=1) != 0]
    df = df[df.iloc[:, 14:26].sum(axis=1) != 0]
    df = df[df.iloc[:, 26:38].sum(axis=1) != 0]
    return df

def apply_log_transform(df):
    df_log = df.iloc[:, 1:38] + 1
    df_log = df_log.apply(np.log)
    return df_log

def prepare_features_labels(df, transformed_df):
    labels = df['churn'].reset_index(drop=True)
    features = transformed_df.reset_index(drop=True)
    return features, labels

def balance_data(x_train, y_train):
    smote_tomek = SMOTETomek(tomek=TomekLinks(), random_state=42)
    x_resampled, y_resampled = smote_tomek.fit_resample(x_train, y_train)
    return x_resampled, y_resampled

def train_xgboost(x_train, y_train):
    model = XGBClassifier()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def predict_and_export(model, new_data_path, output_path):
    df_new = pd.read_excel(new_data_path)
    df_new = df_new[df_new.iloc[:, 2:14].sum(axis=1) != 0]
    df_new = df_new[df_new.iloc[:, 14:26].sum(axis=1) != 0]
    df_new = df_new[df_new.iloc[:, 26:38].sum(axis=1) != 0]
    
    df_new_log = df_new.iloc[:, 1:38] + 1
    df_new_log = df_new_log.apply(np.log)
    
    predictions = model.predict(df_new_log)
    df_result = df_new.assign(label=predictions)
    df_result.to_excel(output_path, index=False)
    print(f"Predictions exported to {output_path}")

if __name__ == "__main__":
    # Step 1: Load and preprocess training data
    train_path = r"C:\Users\USER\Desktop\newbooksme1.xlsx"
    df = load_and_clean_data(train_path)
    df_log = apply_log_transform(df)
    x, y = prepare_features_labels(df, df_log)

    # Step 2: Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Step 3: Balance the training data
    x_train_bal, y_train_bal = balance_data(x_train, y_train)

    # Step 4: Train XGBoost model
    model = train_xgboost(x_train_bal, y_train_bal)

    # Step 5: Evaluate the model
    print("\n🔍 Evaluation on Test Set:")
    evaluate_model(model, x_test, y_test)

    # Step 6: Predict on new customer data and export
    new_data_path = r"C:\Users\USER\Desktop\finalsme.xlsx"
    output_path = r"C:\Users\USER\Desktop\predictsme1.xlsx"
    predict_and_export(model, new_data_path, output_path)

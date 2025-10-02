

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equal_opportunity_difference

# load dataset (Kaggle version)
url = "https://raw.githubusercontent.com/koaning/scikit-fairness/master/fairlearn/data/german_credit_data.csv"
df = pd.read_csv(url)

# basic preprocessing
df = df.dropna()
df['Risk'] = df['Risk'].map({'good': 0, 'bad': 1})  # target variable

# encode categorical features
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# features and target
X = df.drop('Risk', axis=1)
y = df['Risk']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
}

# define protected attributes
protected_features = {
    "gender": df["Sex"],                        # male/female
    "age_group": (df["Age"] > 30).astype(int)   # 0 = young, 1 = older
}

# evaluate fairness
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")

    # fairness metrics
    for feature, values in protected_features.items():
        mf = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=values.iloc[y_test.index]
        )
        dp = demographic_parity_difference(y_test, y_pred, sensitive_features=values.iloc[y_test.index])
        eo = equal_opportunity_difference(y_test, y_pred, sensitive_features=values.iloc[y_test.index])

        print(f"\nFairness Analysis by {feature}:")
        print("Selection rate by group:\n", mf.by_group)
        print("Demographic Parity Difference:", dp)
        print("Equal Opportunity Difference:", eo)

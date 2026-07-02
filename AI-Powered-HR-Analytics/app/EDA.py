import joblib
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

# =========================
# LOAD DATASET
# =========================

DATA_FILE = Path("C:/Users/ashwa/Downloads/emp_attrition.csv")
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found. Please place emp_attrition.csv next to {Path(__file__).name}"
    )

df = pd.read_csv(DATA_FILE)

print("="*50)
print("DATASET SHAPE")
print(df.shape)

print("="*50)
print("FIRST 5 ROWS")
print(df.head())

print("="*50)
print("MISSING VALUES")
print(df.isnull().sum())

# =========================
# ATTRITION ANALYSIS
# =========================

print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition")
plt.show()

# =========================
# DEPARTMENT ANALYSIS
# =========================

plt.figure(figsize=(10,5))
sns.countplot(x="Department", hue="Attrition", data=df)
plt.xticks(rotation=20)
plt.title("Department Wise Attrition")
plt.show()

# =========================
# OVERTIME ANALYSIS
# =========================

plt.figure(figsize=(6,4))
sns.countplot(x="OverTime", hue="Attrition", data=df)
plt.title("Overtime vs Attrition")
plt.show()

# =========================
# AGE DISTRIBUTION
# =========================

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="Age", hue="Attrition", kde=True)
plt.title("Age Distribution")
plt.show()

# =========================
# DATA PREPROCESSING
# =========================

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# =========================
# TRAIN TEST SPLIT
# =========================

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================
# MODEL TRAINING AND EVALUATION
# =========================

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(random_state=42, probability=True),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

if xgboost_available:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
else:
    print("\nXGBoost is not installed. Install it with 'pip install xgboost' to include XGBoost in model comparison.")

results = []

for name, model in models.items():
    print(f"\nTraining and evaluating: {name}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    print("Model Accuracy:", f"{accuracy*100:.2f}%")
    print("Classification Report")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} ({results_df.iloc[0]['Accuracy']*100:.2f}% accuracy)")

# =========================
# FEATURE IMPORTANCE (RANDOM FOREST)
# =========================

rf_model = models["Random Forest"]
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features")
print(importance.head(10))

plt.figure(figsize=(10,6))
plt.barh(importance["Feature"].head(10), importance["Importance"].head(10))
plt.title("Top Factors Affecting Attrition")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()

# =========================
# SAVE RESULTS
# =========================

importance.to_csv("feature_importance.csv", index=False)
joblib.dump(best_model, "attrition_model.pkl")

print("\nProject Completed Successfully!")
print(f"Feature importance saved as feature_importance.csv")
print(f"Best model saved as attrition_model.pkl ({best_model_name})")
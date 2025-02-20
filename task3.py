import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Import machine learning models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Import metrics for evaluating model performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set seaborn theme for better plots
sns.set_theme(style="white", rc={'figure.figsize': (12, 8)})

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train = pd.read_csv("C:\\Users\\himap\\Downloads\\Testing (2).csv")
test = pd.read_csv("C:\\Users\\himap\\Downloads\\Testing (2).csv")

# Drop unwanted column if it exists
if "Unnamed: 133" in train.columns:
    train.drop("Unnamed: 133", axis=1, inplace=True)

# Check for missing values
print("\n🔹 Missing Values:\n", train.isnull().sum())

# Encode target variable
encoder = LabelEncoder()
train['prognosis'] = encoder.fit_transform(train['prognosis'])
test['prognosis'] = encoder.transform(test['prognosis'])

# Split features and labels
x_train = train.drop(columns="prognosis")
y_train = train["prognosis"]
x_test = test.drop(columns="prognosis")
y_test = test["prognosis"]

# 🔹 Plot Feature Distributions
for col in x_train.columns[:5]:  # Only plotting first 5 features to avoid clutter
    plt.figure(figsize=(8, 5))
    sns.histplot(train[col], bins=30, kde=True, color="blue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# 🔹 Pie & Bar Charts for Categorical Data
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.pie(train["prognosis"].value_counts(), labels=encoder.classes_, autopct='%1.1f%%', startangle=90)
plt.title("Prognosis Distribution (Pie Chart)")

plt.subplot(1, 2, 2)
sns.countplot(y=train["prognosis"], palette="coolwarm")
plt.title("Prognosis Distribution (Bar Chart)")
plt.xlabel("Prognosis")
plt.ylabel("Frequency")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# 🔹 Define training function
def train_evaluate(x_train, x_test, y_train, y_test):
    print("\n🔹 Data Shapes:")
    print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}\n")

    models = {
        'SVC': SVC(C=1, kernel="rbf"),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(max_depth=50, random_state=42, criterion="entropy", max_features=7),
        'Random Forest': RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=50, max_features=7, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=.01, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42, learning_rate=1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    best_model = None
    best_accuracy = 0
    accuracies = {}

    for name, model in models.items():
        print(f"\n🔹 Training {name} Model...\n")
        
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies[name] = accuracy

        print(f"✅ {name} Accuracy: {accuracy:.2f}\n")
        print(classification_report(y_test, predictions))

        # Plot confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Track the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # 🔹 Plot Accuracy Comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()

    print(f"\n🏆 Best Model: {best_model} with Accuracy: {best_accuracy:.2f}")
    return best_model

# Run training and evaluation
best_model = train_evaluate(x_train, x_test, y_train, y_test)

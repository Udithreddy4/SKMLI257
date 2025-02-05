import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

file_path = r"C:\\Users\\himap\\Downloads\\diabetes.csv"
df = pd.read_csv(file_path)

print(df.head())

df = df.fillna(df.mean())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Unique labels in Outcome: {y.unique()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

plt.figure(figsize=(12, 8))
df.drop('Outcome', axis=1).hist(bins=15, color='blue', edgecolor='black', figsize=(12, 10))
plt.suptitle('Feature Distribution')
plt.show()

roc_display = RocCurveDisplay.from_estimator(nb_model, X_test, y_test)
roc_display.ax_.set_title('ROC Curve')
plt.show()

pr_display = PrecisionRecallDisplay.from_estimator(nb_model, X_test, y_test)
pr_display.ax_.set_title('Precision-Recall Curve')
plt.show()

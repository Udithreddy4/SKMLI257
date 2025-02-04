import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error

data = pd.read_csv("C:\\Users\\himap\\Downloads\\diabetes.csv")

X = data.drop('Outcome', axis=1)
y = data['Outcome']               

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
sns.countplot(x=y_test, palette='Blues')
plt.title("Actual Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.countplot(x=y_pred, palette='Greens')
plt.title("Predicted Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].pie(y_test.value_counts(), labels=y_test.value_counts().index, autopct='%1.1f%%', colors=['lightblue', 'blue'], startangle=90)
axs[0].set_title('Actual Outcome Distribution')

axs[1].pie(np.bincount(y_pred), labels=np.unique(y_pred), autopct='%1.1f%%', colors=['lightgreen', 'green'], startangle=90)
axs[1].set_title('Predicted Outcome Distribution')

plt.show()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

plt.figure(figsize=(14, 10))

for i, column in enumerate(X.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=y, y=data[column], palette="Blues")
    plt.title(f'{column} vs Outcome')

plt.tight_layout()
plt.show()

feature_importances = dt_classifier.feature_importances_

features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

features_df = features_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
plt.pie(features_df['Importance'], labels=features_df['Feature'], autopct='%1.1f%%', colors=sns.color_palette('Paired', len(features_df)), startangle=90, shadow=True)
plt.title("Feature Contribution to Outcome")
plt.axis('equal')  
plt.show()

# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Program:
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Placement_Data.csv")
print("First 5 rows of the dataset:")
print(data.head())

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())

print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())

X = data1.iloc[:, :-1]
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])
```

## Output:
<img width="737" height="322" alt="Screenshot 2026-08-08 103535" src="https://github.com/user-attachments/assets/79824fee-674d-47d7-b643-dde48c1b8355" />

<img width="795" height="315" alt="Screenshot 2026-08-08 103638" src="https://github.com/user-attachments/assets/b208e0b2-94f7-4e19-b527-0531f6ca682b" />

<img width="452" height="393" alt="Screenshot 2026-08-08 103725" src="https://github.com/user-attachments/assets/a3578c42-9f4f-47a9-a59d-2b7b22149d5c" />

<img width="770" height="318" alt="Screenshot 2026-08-08 103759" src="https://github.com/user-attachments/assets/d2071985-ef47-417e-b207-0a6a0f011197" />

<img width="852" height="602" alt="Screenshot 2026-08-08 103858" src="https://github.com/user-attachments/assets/978d82dc-a414-4688-ba7c-691349369c45" />

<img width="1267" height="365" alt="Screenshot 2026-08-08 103914" src="https://github.com/user-attachments/assets/edec2af7-da1b-45b3-8b73-9b3df37b08f5" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

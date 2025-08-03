
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("sales.csv")
print(data.head())
print(data.describe())
print(data.info())

print(data.isnull().sum())

# 5. Visualize the Data
sns.pairplot(data)
plt.title("Sales Data Pairplot")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
X = data.drop("Sales", axis=1) 
y = data["Sales"]              
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
df_results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_results.head())


plt.figure(figsize=(10,6))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred, label="Predicted", marker='x')
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.show()
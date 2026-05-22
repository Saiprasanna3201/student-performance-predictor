
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'attendance': [50, 60, 65, 70, 75, 80, 90, 95],
    'previous_score': [40, 50, 55, 60, 65, 70, 80, 90],
    'final_score': [45, 52, 58, 63, 67, 72, 85, 92]
}

df = pd.DataFrame(data)

X = df[['study_hours', 'attendance', 'previous_score']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("Predictions:", predictions)
print("Actual:", y_test.values)
print("Mean Squared Error:", mse)

# FIXED INPUT
new_data = pd.DataFrame([[5, 80, 70]], columns=['study_hours', 'attendance', 'previous_score'])
predicted_score = model.predict(new_data)

print("Predicted Score:", predicted_score[0])

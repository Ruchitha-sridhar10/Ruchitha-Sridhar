# 📌 Health Tracker - Activity & Calorie Prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# ✅ Step 1: Create a small sample dataset (no file upload needed)
data = pd.DataFrame({
    "steps": [2500, 5000, 8000, 3000, 10000, 1500, 7000, 12000, 4000, 6000],
    "heart_rate": [70, 85, 90, 72, 100, 65, 95, 105, 78, 88],
    "age": [20, 25, 30, 22, 35, 28, 24, 40, 33, 29],
    "activity": ["walking", "walking", "running", "walking", "running", "resting", "walking", "running", "walking", "walking"],
    "calories": [120, 250, 400, 150, 500, 70, 300, 600, 200, 350]
})

# ✅ Step 2: Split features & targets
X = data[["steps", "heart_rate", "age"]]
y_activity = data["activity"]
y_calories = data["calories"]

X_train, X_test, y_train_a, y_test_a = train_test_split(X, y_activity, test_size=0.2, random_state=42)
_, _, y_train_c, y_test_c = train_test_split(X, y_calories, test_size=0.2, random_state=42)

# ✅ Step 3: Train Models
activity_model = RandomForestClassifier()
activity_model.fit(X_train, y_train_a)

calorie_model = RandomForestRegressor()
calorie_model.fit(X_train, y_train_c)

# ✅ Step 4: Predictions
y_pred_activity = activity_model.predict(X_test)
y_pred_calories = calorie_model.predict(X_test)

print("🎯 Activity Prediction Accuracy:", accuracy_score(y_test_a, y_pred_activity))
print("🔥 Calorie Prediction MAE:", mean_absolute_error(y_test_c, y_pred_calories))

# ✅ Step 5: Test with new user input
new_data = np.array([[7500, 92, 26]])  # steps, heart rate, age
pred_activity = activity_model.predict(new_data)[0]
pred_calories = calorie_model.predict(new_data)[0]

print(f"\n👉 Predicted Activity: {pred_activity}")
print(f"👉 Estimated Calories Burned: {int(pred_calories)} kcal")
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# Load the dataset
file_path = 'synthetic_crop_yield_dataset.csv'  # Update the file path if necessary
data = pd.read_csv(file_path)

# Preprocessing the dataset
# Encode categorical variables
label_encoders = {}
categorical_columns = ['Type of Soil', 'Season', 'Type of Seeds',
                       'Type of Transplanting Method', 'Type of Irrigation Method',
                       'Type of Fertilizers Used', 'Yield Category']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Scale numerical variables
scaler = StandardScaler()
numerical_columns = ['Area Ploughed (in acres)', 'Average Rainfall (in mm)']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Separate features and target variable
X = data.drop(columns='Yield Category')  # Features
y = data['Yield Category']  # Target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize baseline models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Train and evaluate baseline models
model_performance = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    model_performance[name] = accuracy
    
    # Track the best model
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy

# Display baseline model performances
print("Baseline Model Performances:")
for model_name, acc in model_performance.items():
    print(f"{model_name}: {acc:.2f}")

# Advanced model: XGBoost with hyperparameter tuning
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring="accuracy", verbose=1)
grid_search.fit(X_train, y_train)

# Best XGBoost model
best_xgb = grid_search.best_estimator_

# Evaluate XGBoost on the test set
y_pred_xgb = best_xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

print("\nXGBoost Performance After Hyperparameter Tuning:")
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Save the best overall model
if xgb_accuracy > best_accuracy:
    best_model = best_xgb
    best_accuracy = xgb_accuracy

model_path = "cropyield.pkl"  # Adjust path as necessary
joblib.dump(best_model, model_path)
print(f"\nBest model saved at {model_path}")

# To load the model later:
# loaded_model = joblib.load("cropyield.pkl")

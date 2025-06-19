# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib

# Step 1: Load the dataset
file_path = 'synthetic_farmer_subsidy_dataset_with_names.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Step 2: Analyze the dataset
print("Dataset Shape:", data.shape)
print("Dataset Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe(include='all'))

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Step 3: Preprocessing
# Handle missing values (if any)
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    data[col].fillna(data[col].mean(), inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode target variable
label_encoder = LabelEncoder()
data['Subsidy Eligibility'] = label_encoder.fit_transform(data['Subsidy Eligibility'])

# Encode categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Separate features and target
X = data.drop(columns=["Subsidy Eligibility"])
y = data["Subsidy Eligibility"]

# Normalize/scale numerical data
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 4: Balancing the dataset with Random Oversampling
data_balanced = pd.concat([pd.DataFrame(X), pd.Series(y, name='Target')], axis=1)
majority_class = data_balanced[data_balanced['Target'] == data_balanced['Target'].mode()[0]]
minority_classes = data_balanced[data_balanced['Target'] != data_balanced['Target'].mode()[0]]

# Oversample minority classes
minority_classes_upsampled = minority_classes.groupby('Target').apply(
    lambda x: resample(x, replace=True, n_samples=majority_class.shape[0], random_state=42)
).reset_index(drop=True)

# Combine majority class with upsampled minority classes
data_balanced = pd.concat([majority_class, minority_classes_upsampled]).reset_index(drop=True)

# Shuffle the balanced dataset
data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and target after balancing
X_balanced = data_balanced.drop(columns=["Target"])
y_balanced = data_balanced["Target"]

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Step 6: Train various models
# Define Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Set up hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Step 7: Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("Best Model Parameters:", grid_search.best_params_)
print("\nClassification Report:\n", report)

# Step 8: Save the best model
model_path = 'best_subsidy_model_oversampling.pkl'
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")
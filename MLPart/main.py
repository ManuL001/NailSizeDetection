from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from createXTrain import create_dataset
from collections import Counter


# Function to preprocess features dynamically
def preprocess_features_dynamic(X):
    processed = []
    for sample in X:
        features = []
        for idx, feature in enumerate(sample):
            if isinstance(feature, list):
                # Extract sequence statistics
                features.extend([
                    len(feature),                  # Number of elements
                    np.mean(feature),             # Mean value
                    np.std(feature),              # Standard deviation
                    np.min(feature),              # Minimum value
                    np.max(feature),              # Maximum value
                    feature[0],                   # First value
                    feature[-1]                   # Last value
                ])
            else:
                features.append(feature)  # Scalar values directly
        processed.append(features)
    return np.array(processed)

# Function to create train and test splits
def create_train_test_split(guessedHammerHits):
    # Initialize X and y
    X = []
    y = []

    # Loop through the dictionary to populate X and y
    for dataset_name, dataset_values in guessedHammerHits.items():
        # Extract label based on the key name
        if dataset_name.startswith("20 mm"):
            label = 20
        elif dataset_name.startswith("40 mm"):
            label = 40
        elif dataset_name.startswith("80 mm"):
            label = 80
        else:
            raise ValueError(f"Unknown dataset label in key: {dataset_name}")

        # Add the features (X) and label (y)
        X.append(dataset_values)  # Each dataset_values is [num_hammer_hits, ...]
        y.append(label)

    # Convert X and y to NumPy arrays
    X = np.array(X, dtype=object)  # dtype=object because lists are nested
    y = np.array(y)

    # Split into train and test datasets (12 train, 3 test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)

    # Preprocess the train and test features
    X_train_processed = preprocess_features_dynamic(X_train)
    X_test_processed = preprocess_features_dynamic(X_test)

    return X_train_processed, X_test_processed, y_train, y_test

# Initialize the training data
train_data = create_dataset()

# Get the train data
X_train, X_test, y_train, y_test = create_train_test_split(train_data)

# Standardize the features
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

# Handle missing values with imputation
imputer = SimpleImputer(strategy='mean')  # You can also use 'median', 'constant', etc.
X_train_processed = imputer.fit_transform(X_train_processed)
X_test_processed = imputer.transform(X_test_processed)

param_grid = {
    'n_estimators': [40, 50, 100],       # Increase number of trees for stability
    'max_depth': [3, 5, 7, None],       # Limit tree depth to control overfitting
    'min_samples_split': [5, 10],       # Require more samples per split
    'min_samples_leaf': [2, 4]          # Require more samples in leaves
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train_processed, y_train)
best_rf = grid_search.best_estimator_

# Display the best hyperparameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Perform cross-validation on the training set
cv_scores = cross_val_score(best_rf, X_train_processed, y_train, cv=3)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Compare train and test performance
train_accuracy = best_rf.score(X_train_processed, y_train)
test_accuracy = best_rf.score(X_test_processed, y_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on the test set
y_pred = best_rf.predict(X_test_processed)

# Print classification report with zero_division parameter
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

print("Training label distribution:", Counter(y_train))
print("Test label distribution:", Counter(y_test))
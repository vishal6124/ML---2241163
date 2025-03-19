# Cell 1: Importing Libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Interpretation: Necessary libraries are imported for data manipulation and visualization.

# Cell 2: Importing the Dataset
# -----------------------------
# Replace 'dataset.csv' with the actual dataset file path
data = pd.read_csv("D:\\Desktop\\CHRIST\\SEM 6\\ML\\nutrition.csv")

# Display the first few rows of the dataset
print(data.head())

# Interpretation: Dataset is loaded and previewed to confirm its structure and contents.

# Cell 3: Handling Missing Data
# -----------------------------
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Fill missing categorical values with mode and numeric with median

for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)  # Corrected the typo in line
    else:
        data[col].fillna(data[col].median(), inplace=True)

print("Missing Values After Handling:\n", data.isnull().sum())

# Interpretation: Missing values are addressed to ensure consistency during analysis.

# Cell 4: Distribution of Numeric Variables (Histogram)
# -----------------------------------------------------
numeric_columns = ['Ages', 'Height', 'Weight', 'Daily Calorie Target', 
                   'Protein', 'Sugar', 'Sodium', 'Calories', 
                   'Carbohydrates', 'Fiber', 'Fat']

data[numeric_columns].hist(figsize=(15, 12), bins=15, color='skyblue')
plt.suptitle('Distribution of Numeric Variables', fontsize=16)
plt.show()

# Interpretation: Histograms provide an overview of numeric variable distributions, revealing potential skews or outliers.

# Cell 5: Distribution of Nominal Variables (Count Plot)
# ------------------------------------------------------
categorical_columns = ['Gender', 'Activity Level', 'Dietary Preference', 'Disease']

for col in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col, data=data, palette='Set2')  # Corrected syntax
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Interpretation: Count plots highlight the frequency of each category in the nominal variables.

# Cell 6: Check Class Distribution
# -------------------------------
print("Class Distribution in Target Variable 'Disease':")
print(data['Disease'].value_counts())

# Handle Underrepresented Classes
min_class_threshold = 2  # Minimum number of samples per class
class_counts = data['Disease'].value_counts()
underrepresented_classes = class_counts[class_counts < min_class_threshold].index

if len(underrepresented_classes) > 0:
    print("\nUnderrepresented Classes:", list(underrepresented_classes))
    # Option 1: Remove samples of underrepresented classes
    data = data[~data['Disease'].isin(underrepresented_classes)]
    print("\nUpdated Class Distribution:")
    print(data['Disease'].value_counts())

# Cell 7: Heatmap for Correlation
# -------------------------------
plt.figure(figsize=(12, 10))
corr_matrix = data[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

# Interpretation: Correlation heatmap reveals relationships between numeric attributes, aiding in understanding dependencies.

# Cell 8: Define Features and Target
# -------------------------------
X = data.drop('Disease', axis=1).select_dtypes(include=np.number).values  # Numeric features
y = data['Disease'].values  # Target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=21, stratify=y)

# KNN Model Initialization
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the KNN model
knn.fit(X_train, y_train)

# Running Predictions
y_pred = knn.predict(X_test)

# Validation
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))  # Set zero_division to 1 to avoid warnings

# Cell 9: Error Rate vs. K Value (Elbow Method)
# -------------------------------------------
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Plot the Error Rate
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# Cell 10: Retrain with Best K Value
# ----------------------------------
best_k = error_rate.index(min(error_rate)) + 1
print(f"Best K Value: {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_best = knn.predict(X_test)

# Evaluate with Best K
print("\nClassification Report with Best K:")
print(classification_report(y_test, y_pred_best, zero_division=1))  # Set zero_division to 1 to avoid warnings
print("Accuracy with Best K:", knn.score(X_test, y_test))

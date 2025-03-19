
# Cell 1: Importing Libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        data[col].fillna(data[col].mode()[0], inplace=True)
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

# Cell 6: Comparison of Numeric Variables (Scatterplot)
# -----------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Height', y='Weight', hue='Gender', data=data, palette='coolwarm')
plt.title('Height vs Weight by Gender', fontsize=14)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

# Interpretation: Scatterplots help visualize relationships between numeric variables like Height and Weight.

# Cell 7: Heatmap for Correlation
# -------------------------------
plt.figure(figsize=(12, 10))
corr_matrix = data[numeric_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

# Interpretation: Correlation heatmap reveals relationships between numeric attributes, aiding in understanding dependencies.

# Cell 8: Boxplot to Compare Values Within Groups
# -----------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='Activity Level', y='Calories', data=data, palette='muted')
plt.title('Calories by Activity Level', fontsize=14)
plt.xlabel('Activity Level')
plt.ylabel('Calories')
plt.show()

# Interpretation: Boxplots show variation and central tendencies of calories across different activity levels.

# Cell 9: Additional Plot (Violin Plot)
# -------------------------------------
plt.figure(figsize=(10, 6))
sns.violinplot(x='Dietary Preference', y='Sugar', data=data, palette='Set3')
plt.title('Sugar Intake by Dietary Preference', fontsize=14)
plt.xlabel('Dietary Preference')
plt.ylabel('Sugar (grams)')
plt.show()

# Interpretation: Violin plots reveal sugar intake distributions within different dietary preference groups.

# Cell 10: Additional Plot (Swarm Plot)
# -------------------------------------
# Swarmplot for Sodium intake by Disease
plt.figure(figsize=(10, 6))
sns.stripplot(x='Disease', y='Sodium', data=data, size=6, jitter=0.2, hue='Disease', palette='deep')
plt.title('Sodium Intake by Disease', fontsize=14)
plt.xlabel('Disease')
plt.ylabel('Sodium (mg)')
plt.show()


# Interpretation: Swarm plots provide detailed distributions of sodium intake across disease categories.

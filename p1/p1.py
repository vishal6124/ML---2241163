 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Replace 'your_dataset.csv' with the path to your CSV file
data = pd.read_csv('nutrition.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# 1. Overview of the dataset
print("\nDataset Info:")
print(data.info())

# Table: Overview of missing values
missing_values = data.isnull().sum().reset_index()
missing_values.columns = ['Attribute', 'Missing Count']
missing_values['Missing Percentage'] = (missing_values['Missing Count'] / len(data)) * 100
print("\nMissing Values Summary:")
print(missing_values)

# Univariate Analysis
# --------------------

# 1. Age distribution using a histogram and KDE
plt.figure(figsize=(8, 6))
sns.histplot(data['Ages'], kde=True, color='blue', bins=15)
plt.title('Age Distribution')
plt.xlabel('Ages')
plt.ylabel('Frequency')
plt.show()

# 2. Gender count using a pie chart
gender_counts = data['Gender'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Gender Distribution')
plt.show()

# 3. Activity Level count using a bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='Set2')
plt.title('Activity Level Count')
plt.xlabel('Activity Level')
plt.ylabel('Count')
plt.show()

# Bivariate Analysis
# -------------------

# 4. Scatter plot for Height vs Weight by Gender
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Height', y='Weight', hue='Gender', data=data, palette='coolwarm')
plt.title('Height vs Weight by Gender')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show      ()

# 5. Box plot for Calories by Activity Level
plt.figure(figsize=(10, 6))
sns.boxplot(x='Activity Level', y='Calories', data=data, palette='muted')
plt.title('Calories by Activity Level')
plt.xlabel('Activity Level')
plt.ylabel('Calories')
plt.show()

# 6. Violin plot for Sugar intake by Dietary Preference
plt.figure(figsize=(10, 6))
sns.violinplot(x='Dietary Preference', y='Sugar', data=data, palette='Set3')
plt.title('Sugar Intake by Dietary Preference')
plt.xlabel('Dietary Preference')
plt.ylabel('Sugar (grams)')
plt.show()

# Multivariate Analysis
# ----------------------

# 7. Pairplot of selected numerical attributes, colored by Gender
selected_features = ['Ages', 'Height', 'Weight', 'Calories', 'Protein']
sns.pairplot(data[selected_features + ['Gender']], hue='Gender', palette='husl', corner=True)
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# 8. Heatmap of correlations among numerical attributes
corr_matrix = data[selected_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# 9. Clustered bar chart for Dietary Preference by Disease
plt.figure(figsize=(12, 8))
sns.countplot(x='Dietary Preference', hue='Disease', data=data, palette='Set1')
plt.title('Dietary Preference by Disease')
plt.xlabel('Dietary Preference')
plt.ylabel('Count')
plt.show()

# 10. Swarmplot for Sodium intake by Disease
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Disease', y='Sodium', data=data, palette='deep')
plt.title('Sodium Intake by Disease')
plt.xlabel('Disease')
plt.ylabel('Sodium (mg)')
plt.show()

print("\nEDA Completed!")

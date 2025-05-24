# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary classes
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set styles for plots (uncomment if needed)
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# Load dataset
try:
    diamonds = sns.load_dataset('diamonds')
    print('Successfully loaded built-in dataset from Seaborn')
except Exception as e:
    print(f'Failed to load dataset: {e}')
    exit()

# Data Exploration
print('\n=== Dataset Overview ===')
print(f"Shape: {diamonds.shape}")
print("\nFirst 5 rows:")
print(diamonds.head())
print("\nBasic statistics:")
print(diamonds.describe())

#Data Cleaning
diamonds =diamonds.drop_duplicates()
if diamonds.isnull().sum().sum()>0:
    diamonds=diamonds.dropna

# First ensure categorical columns are encoded (if they exist as strings)
if 'cut' in diamonds.columns:
    diamonds['cut_code'] = LabelEncoder().fit_transform(diamonds['cut'])
if 'color' in diamonds.columns:
    diamonds['color_code'] = LabelEncoder().fit_transform(diamonds['color'])
if 'clarity' in diamonds.columns:
    diamonds['clarity_code'] = LabelEncoder().fit_transform(diamonds['clarity'])

# Exploratory Data Analysis (EDA) and Visualization

# 1. Price Distribution
plt.figure(figsize=(12,6))
sns.histplot(diamonds['price'], bins=50, kde=True)
plt.title('Diamonds Price Distribution', fontsize=16)
plt.xlabel('Price (USD)',fontsize=14)
plt.ylabel('Frequency',fontsize=14)
plt.show()

# 2. Price vs. Carat
plt.figure(figsize=(12,6))
sns.scatterplot(data=diamonds,x='carat', y='price', alpha=0.4)
plt.title('Price vs. Carat Weight', fontsize=16)
plt.xlabel('Carat', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.show()

# 3. Price by Cut Quality
plt.figure(figsize=(12,6))
sns.boxplot(data=diamonds,x='cut',y='price',order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
plt.title('Price Distribution by Cut Quality', fontsize=16)
plt.xlabel('Cut Quality', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.show()

# 4. Price by Color
plt.figure(figsize=(12, 6))
sns.boxplot(x='color', y='price', data=diamonds, order=['D', 'E', 'F', 'G', 'H', 'I', 'J'])
plt.title('Price Distribution by Color Grade', fontsize=16)
plt.xlabel('Color Grade (D is best)', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.show()

# 5. Price by Clarity
plt.figure(figsize=(12, 6))
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
sns.boxplot(x='clarity', y='price', data=diamonds, order=clarity_order)
plt.title('Price Distribution by Clarity Grade', fontsize=16)
plt.xlabel('Clarity Grade', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.show()

# 6. Correlation Matrix
plt.figure(figsize=(12,6))
corr=diamonds.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap='coolwarm',center=0)
plt.title('Correlation Matrix of Diamond Features',fontsize=16)
plt.show()

# 7. 3D Relationship between Carat, Depth, and Price
fig=plt.figure(figsize=(12,6))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(diamonds['carat'],diamonds['depth'],diamonds['price'], alpha=0.3)
ax.set_xlabel('Carat')
ax.set_ylabel('Depth')
ax.set_zlabel('Price')
plt.title('3D Relationship: Carat, Depth, and Price', fontsize=16)
plt.show()

# 8. Pairplot of numerical features
# sns.pairplot(diamonds[['carat', 'depth', 'table', 'price', 'cut']], hue='cut', height=2.5)
# plt.show()

# Advanced Analysis: Price Prediction Model

x=diamonds[['carat','depth','table','x','y','z','cut_code','color_code','clarity_code']]
y=diamonds['price']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Train a Random Forest model
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

# Make predictions and evaluate model
y_predict=model.predict(x_test)

mse=mean_squared_error(y_test,y_predict)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_predict)

print('\n===Model Perfromance===')
print(f"Root Mean squared Error: {rmse:.2F}")
print(f"R-squared: {r2:.2f}")

# Feature Importance
feature_importance=pd.DataFrame({
    'Feature': x.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance',ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=feature_importance,x='Importance',y='Feature')
plt.title('Feature Importance for Diamond Price Prediction', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.show()
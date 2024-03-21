import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso



# Load the dataset
file_path = 'polymer_reactor.txt'
data = pd.read_csv(file_path, sep=None, engine='python')

# Display the first few rows of the dataset to understand its structure
print(data.head())
print(data.shape)
print(data.describe())
print(data.info())
# ===============================================
# ==         Data Preprocessing                ==
# ===============================================
# 

data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
data['Unnamed: 0'] = data['Unnamed: 0'].astype('int64') // 10**9

# Dropping the 'Unnamed: 0' column as it represents timestamps and might not be necessary for our analysis
# data.drop(columns=['Unnamed: 0'], inplace=True)

column_mapping = {
    'Unnamed: 0': 'Time (UnixTimestamp)',
    "513FC31103.pv": "C3",
    "513HC31114-5.mv": "H2R",
    "513PC31201.pv": "Pressure",
    "513LC31202.pv": "Level",
    "513FC31409.pv": "C2",
    "513FC31114-5.pv": "Cat",
    "513TC31220.pv": "Temp",
    "MFR": "MFR"
}

data = data.rename(columns=column_mapping)
print(data.head())
# Checking for missing values
missing_values = data.isnull().sum()

# Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['MFR']))  # Excluding the target variable for scaling

# Creating a DataFrame for the scaled features
scaled_features_df = pd.DataFrame(scaled_features, columns=data.columns[:-1])  # Excluding the target variable column

print(missing_values,'\n', scaled_features_df.head())

# Imputing missing values using the mean
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(data.drop(columns=['MFR']))  # Excluding the target variable for imputation

# Creating a DataFrame for the imputed data
imputed_data_df = pd.DataFrame(imputed_data, columns=data.columns[:-1])  # Excluding the target variable column MFR

# Checking if there are any missing values left
missing_values_after_imputation = imputed_data_df.isnull().sum()

print(missing_values_after_imputation, '\n', imputed_data_df.head())

# Including the target variable MFR in the imputed dataset for correlation analysis
imputed_data_full_df = imputed_data_df.copy()
imputed_data_full_df['MFR'] = data['MFR'].values

# ===============================================
# ==         Basic Operations                  ==
# ===============================================



# Group by 'Pressure' and calculate the mean MFR for each group
mean_mfr_by_pressure = imputed_data_full_df.groupby('Pressure')['MFR'].mean().reset_index()

# Sorting the results for better visualization
mean_mfr_by_pressure = mean_mfr_by_pressure.sort_values(by='MFR', ascending=False)

print(mean_mfr_by_pressure.head())  # Display the top 5 groups

# Group by 'Time (UnixTimestamp)' and calculate the mean for each group
grouped_data = imputed_data_full_df.groupby('Time (UnixTimestamp)').mean().reset_index()
print(grouped_data.head())  # Display the top 5 groups


# Plotting
features_to_plot = ["C3", "H2R", "Pressure", "Level", "C2", "Cat", "Temp", "MFR"]
num_plots = len(features_to_plot)
plt.figure(figsize=(15, 10))

for i, feature in enumerate(features_to_plot, start=1):
    plt.subplot(math.ceil(num_plots / 2), 2, i)  # Arrange plots in 2 columns
    plt.plot(grouped_data['Time (UnixTimestamp)'], grouped_data[feature], label=feature)
    plt.xlabel('Time (UnixTimestamp)')
    plt.ylabel(feature)
    plt.title(f'{feature} Evolution Over Time')
    plt.legend()

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.show()


# Sorting data by time
sorted_data_by_time = imputed_data_full_df.sort_values(by='Time (UnixTimestamp)')

# Plotting MFR over time
plt.figure(figsize=(12, 6))
plt.plot(sorted_data_by_time['Time (UnixTimestamp)'], sorted_data_by_time['MFR'], marker='o', linestyle='-', markersize=2)
plt.title('MFR Trend over Time')
plt.xlabel('Time (UnixTimestamp)')
plt.ylabel('MFR')
plt.show()


# Visualizing the relationship between MFR and Pressure
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pressure', y='MFR', data=imputed_data_full_df)
plt.title('MFR vs. Pressure')
plt.show()


# ===============================================
# ==                Correlations               ==
# ===============================================


# Calculating Pearson correlation coefficients
correlation_matrix = imputed_data_full_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of all labes')
plt.show()

# Extracting correlations with the target variable
correlations_with_mfr = correlation_matrix['MFR'].sort_values(ascending=False)
print(correlations_with_mfr)



# Convert the Series to a DataFrame for easier plotting
correlations_with_mfr_df = correlations_with_mfr.reset_index()
correlations_with_mfr_df.columns = ['Feature', 'Correlation with MFR']

# Sort the DataFrame for better visualization
correlations_with_mfr_df = correlations_with_mfr_df.sort_values(by='Correlation with MFR', ascending=True)

# Plotting
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
sns.barplot(x='Correlation with MFR', y='Feature', data=correlations_with_mfr_df, palette='coolwarm')
plt.title('Correlation of Features with MFR')
plt.xlabel('Correlation with MFR')
plt.ylabel('Features')

plt.show()


# ===============================================
# ==                Models                     ==
# ===============================================

RMSE_results = {
    'RMSE_poly': 0,
    'RMSE_lasso': 0,
    'RMSE_ridge': 0}

R2_results: {
    'r2_Score_poly': 0,
    'r2_Score_ridge': 0,
    'r2_Score_lasso': 0,
    } 

# Preparing the data
X = imputed_data_df
y = imputed_data_full_df['MFR']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================================
# Polynomial Regression
# Creating polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Training the model
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Predicting on the test set
y_pred_poly = poly_reg.predict(X_test_poly)

# Evaluating the model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f' RMSE_poly:{mse_poly}\n r2_Score_poly: {r2_poly}')



# ===============================================
# Ridge Regression
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)

# Evaluating Ridge Regression
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# ===============================================
# Lasso Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)

# Evaluating Lasso Regression
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# ===============================================
# ==                Results                    ==
# ===============================================

print(f' rmse_ridge: {mse_ridge},\n r2_ridge: {r2_ridge},\n rmse_lasso: {mse_lasso},\n r2_lasso: {r2_lasso}')



from sklearn.ensemble import RandomForestRegressor

# Initialize the model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the number of trees

# Train the model
rf_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_reg.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Regression RMSE: {math.sqrt(mse_rf)}')
print(f'Random Forest Regression R^2 Score: {r2_rf}')


from sklearn.ensemble import GradientBoostingRegressor

# Initialize the model
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)  # Parameters can be adjusted

# Train the model
gb_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gb_reg.predict(X_test)

# Evaluate the model
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f'Gradient Boosting Regression RMSE: {math.sqrt(mse_gb)}')
print(f'Gradient Boosting Regression R^2 Score: {r2_gb}')


import xgboost as xgb

# Initialize the model
xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)  # Parameters can be adjusted

# Train the model
xgb_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_reg.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost Regression RMSE: {math.sqrt(mse_xgb)}')
print(f'XGBoost Regression R^2 Score: {r2_xgb}')

RMSE_results = {
    'RMSE_poly': math.sqrt(mse_poly),
    'RMSE_lasso': math.sqrt(mse_lasso),
    'RMSE_ridge': math.sqrt(mse_ridge),
    'RMSE_RF': math.sqrt(mse_rf),
    'RMSE_GBR': math.sqrt(mse_gb),
    'RMSE_xGB': math.sqrt(mse_xgb),
}
R2_results = {
    'r2_Score_ridge': r2_ridge,
    'r2_Score_poly': r2_poly,
    'r2_Score_lasso': r2_lasso,
    'r2_Score_RF': r2_rf,
    'r2_Score_GBR': r2_gb,
    'r2_Score_xGB': r2_xgb,
}
print(RMSE_results)
print(R2_results)

min_key, min_value = min(RMSE_results.items(), key=lambda x: x[1])
print(f'The best performing RMSE values are: {min_key} {min_value} ')

max_key, max_value = max(R2_results.items(), key=lambda x: x[1])
print(f'The best performing R2 Score values are: {max_key} {max_value} ')
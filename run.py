import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Load the dataset
file_path = 'polymer_reactor.txt'
# data = pd.read_csv(file_path, sep="\t")  # Assuming tab-separated values; this may need adjustment
data = pd.read_csv(file_path, sep=None, engine='python')

# Display the first few rows of the dataset to understand its structure
print(data.head())


# Data Preprocessing

data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
data['Time'] = data['Unnamed: 0'].astype('int64') // 10**9

# Dropping the 'Unnamed: 0' column as it represents timestamps and might not be necessary for our analysis
data.drop(columns=['Unnamed: 0'], inplace=True)
print(data.head())
column_mapping = {
    # 'Unnamed: 0': 'Time',
    "513FC31103.pv": "C3",
    "513HC31114-5.mv": "H2R",
    "513PC31201.pv": "Pressure",
    "513LC31202.pv": "Level",
    "513FC31409.pv": "C2",
    "513FC31114-5.pv": "Cat",
    "513TC31220.pv": "Temp",
    "MFR": "MFR"
}

# Assuming 'data' is your DataFrame
data = data.rename(columns=column_mapping)
# Checking for missing values
missing_values = data.isnull().sum()
data['Time'] = pd.to_datetime(data['Time'])

# Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['MFR']))  # Excluding the target variable for scaling

# Creating a DataFrame for the scaled features
scaled_features_df = pd.DataFrame(scaled_features, columns=data.columns[:-1])  # Excluding the target variable column

print(missing_values,'\n', scaled_features_df.head())

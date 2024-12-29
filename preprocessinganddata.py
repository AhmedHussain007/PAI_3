import torch
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.datasets import fetch_california_housing
from torch.utils.data import TensorDataset,DataLoader


def load_classificationdata():
    df = pd.read_excel('msint.xlsx')

    # last column (label) and first 784 columns (input features)
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   
    X = X / 255.0

    X = X.reshape(-1, 28, 28, 1)

    # Determine the number of unique classes in labels
    num_classes = len(np.unique(y))

    # One-hot encode the labels
    y = to_categorical(y, num_classes=num_classes)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train , y_train , X_val, X_test, y_val, y_test , num_classes

def cnn_parameters():
    return {
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu', 'tanh'],
    'batch_size': [32, 64],
    'epochs': [10, 20],
    'filters1': [32, 64],
    'filters2': [64, 128]
    }

def ann_cparameters():
    return {
        'epochs': [50],
        'learning_rates': [0.001, 0.01, 0.1],
        'optimizers': ['SGD', 'Adam', 'RMSprop'],
        'layer_configs': [
            [784, 80, 50, 35, 25, 10],
            [784, 100, 75, 50, 25, 10],
            [784, 120, 100, 60, 40, 10]
        ]
    }

def ann_rparameters():
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'epochs': [10, 20],
        'layer_sizes': [
            [8, 50, 1],  # Simple structure
            [8, 80, 50, 1],  # More complex
            [8, 80, 50, 35, 1]  # Even more complex
        ]
    }

    # Convert the grid into combinations
    return list(itertools.product(
        param_grid['learning_rate'],
        param_grid['epochs'],
        param_grid['layer_sizes']
    ))

def load_regression_data():
    data = fetch_california_housing()
    df=pd.DataFrame(data=data.data,columns=data.feature_names)
    df['MedHouseVal']=data.target

    numeric_features=df.drop(['MedHouseVal'],axis=1)
    Target_features=df['MedHouseVal']


    scaler = StandardScaler()
    numeric_features_scaled_df = pd.DataFrame(scaler.fit_transform(numeric_features) , columns=numeric_features.columns)
    df=pd.concat([numeric_features_scaled_df,df['MedHouseVal']],axis=1)
    X = df.drop(columns=['MedHouseVal'])  # Replace 'MedHouseVal' with your actual target column name
    y = df['MedHouseVal'].values
    X_tensor = torch.tensor(X.values, dtype=torch.float32)  # Ensure numeric features are properly scaled
    y_tensor = torch.tensor(y, dtype=torch.float32)  # Use float32 for regression tasks

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 256  

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader , val_loader , test_loader
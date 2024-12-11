# -*- coding: utf-8 -*-

"""
@author: Yunfei Zhang
@software: PyCharm
@file: Model Construction.py
@Start time: XX_XX
@Current time: 2024/12/11 12:47
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings

warnings.filterwarnings('ignore')


class DNN(nn.Module):
    def __init__(self, input_dim, device="cuda" if torch.cuda.is_available() else "cpu",
                 activation_function=nn.ReLU(), dropout_rate=0.5, n_classes=2, hidden_layers=[8, 8, 8, 8]):
        super(DNN, self).__init__()
        self.model = nn.ModuleList()
        in_features = input_dim

        # Build hidden layers
        for num_neurons in hidden_layers:
            layer = nn.Sequential(
                nn.Linear(in_features, num_neurons),
                activation_function,
                nn.Dropout(p=dropout_rate)
            )
            self.model.append(layer)
            in_features = num_neurons

        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], n_classes)
        self.device = device
        self.to(self.device)

    def forward(self, data):
        # Check input data type, convert DataFrame to NumPy array if necessary
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        out = data
        for layer in self.model:
            residual = out
            out = layer(out)
            if residual.shape == out.shape:
                out = out + residual  # Add residual connection
        return self.output_layer(out)

    def predict_proba(self, data):
        # Check input data type, convert DataFrame to NumPy array if necessary
        if isinstance(data, pd.DataFrame):
            data = data.values
        self.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
            outputs = torch.softmax(self(data_tensor), dim=1)
        self.train()
        return outputs.cpu().detach().numpy()

    def fit(self, X, y, epochs=5000, initial_lr=0.0005, min_lr=1e-5):
        # Check input data type, convert DataFrame to NumPy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=initial_lr, weight_decay=1e-6)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()


model_name = None  # Select Your Model Name
random_seed = None  # Please set Your random seed for reproducibility

if model_name == 'SVM':
    model = SVC(probability=True, random_state=random_seed, C=0.05, max_iter=1000)
elif model_name == 'LG':
    model = LogisticRegression(random_state=random_seed)
elif model_name == 'RF':
    model = RandomForestClassifier(random_state=random_seed, n_estimators=30, max_depth=3)
elif model_name == 'KNN':
    model = KNeighborsClassifier()
elif model_name == 'DNN':
    input_dim = None  # Enter Your data dimension
    model = DNN(input_dim)
elif model_name == 'DT':
    model = DecisionTreeClassifier(random_state=random_seed, max_depth=6)
elif model_name == 'AdaBoost':
    model = AdaBoostClassifier(random_state=random_seed)

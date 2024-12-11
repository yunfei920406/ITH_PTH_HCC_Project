import os
import pandas as pd
import numpy as np
import pingouin as pg
from tqdm import tqdm
from scipy import stats


#########Step 1: Feature selection based on ICC

def calculate_icc(measurements1, measurements2):
    """
    Calculate the Intraclass Correlation Coefficient (ICC) between the measurements of two observers.

    Parameters:
    measurements1 (array-like): Measurements provided by the first observer.
    measurements2 (array-like): Measurements provided by the second observer.

    Returns:
    float: The ICC value.
    """

    # Create a DataFrame containing the measurement results and the rater information
    df = pd.DataFrame({
        "Measurements": measurements1 + measurements2,
        "Reader": ["Reader1"] * len(measurements1) + ["Reader2"] * len(measurements2),
        "Index": [i for i in range(1, len(measurements1) + 1)] + [i for i in range(1, len(measurements2) + 1)]
    })

    # Calculate ICC
    icc_result = pg.intraclass_corr(data=df, targets='Index', raters='Reader', ratings='Measurements')

    return icc_result['ICC'].values[0]




#########Step 2: Univariate Analysis

def test_and_compare(data1, data2):
    """
    Test whether two data columns follow a normal distribution. If both follow a normal distribution,
    a t-test is performed; otherwise, a Mann-Whitney U test is used.

    Parameters:
    data1 (array-like): The first data column.
    data2 (array-like): The second data column.

    Returns:
    float: The p-value from the test.
    """

    # Normality test
    _, p1 = stats.shapiro(data1)
    _, p2 = stats.shapiro(data2)

    # Choose the appropriate test based on normality results
    if p1 > 0.05 and p2 > 0.05:  # Both datasets follow normal distribution
        _, p = stats.ttest_ind(data1, data2)
    else:  # At least one dataset does not follow normal distribution
        _, p = stats.mannwhitneyu(data1, data2)

    return p




###########Step 3: Feature selection using Decision Tree

######Feature selection function:
from sklearn.tree import DecisionTreeClassifier

def feature_selection_decision_tree(df, target_column, n=None, random_state=0):
    # Ensure the target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in the DataFrame")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Create a Decision Tree classifier and fit the model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X, y)

    # Get feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    # Sort features by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # If n is not None, limit the number of features
    if n is not None:
        importance_df = importance_df.head(n)

    return importance_df

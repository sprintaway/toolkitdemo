import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import hdbscan
import gower
import miceforest as mf

def is_numeric_column(col):
    col_clean = col.dropna().replace('', np.nan).dropna()
    return pd.to_numeric(col_clean, errors='coerce').notna().all()

def proposed_train_test_split(X, y, test_size=0.2, min_cluster_size=5, random_state=0, task='regression'):
    """
    This function splits the data into training and testing sets using a proposed method
    based on Gower distance and HDBSCAN clustering. It performs an initial split (stratified
    if the task is classification), imputes the training dataset, then clusters the imputed 
    data before making a final stratified split on the cluster labels.

    Args:
        X (pandas.DataFrame): The feature matrix.
        y (pandas.Series): The target variable.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        min_cluster_size (int, optional): The minimum number of samples in a cluster for HDBSCAN. Defaults to 5.
        random_state (int, optional): Controls the randomness of the train-test split. Defaults to 0.
        task (str, optional): The type of task, either 'regression' or 'classification'. Defaults to 'regression'.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """

    numerical_columns = [col for col in X.columns if is_numeric_column(X[col])]
    categorical_columns = [col for col in X.columns if col not in numerical_columns]

    
    X[numerical_columns] = X[numerical_columns].apply(pd.to_numeric, errors='coerce')

    if task == 'classification':
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    for col in categorical_columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    if X_train.isnull().values.any() or X_test.isnull().values.any():
        print("Missing values found. Proceeding with imputation...")

        for col in categorical_columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

      
        kernel_train = mf.ImputationKernel(data=X_train, save_all_iterations=True)
        kernel_train.mice(5, verbose=False)
        X_train_imputed = kernel_train.complete_data(dataset=0)

        kernel_test = mf.ImputationKernel(data=X_test, save_all_iterations=True, datasets=1)
        kernel_test.mice(5, verbose=False)
        X_test_imputed = kernel_test.complete_data(dataset=0)

        for col in categorical_columns:
            X_train_imputed[col] = X_train_imputed[col].astype(str)
            X_test_imputed[col] = X_test_imputed[col].astype(str)
    else:
        print("No missing values found. Skipping imputation...")
        X_train_imputed, X_test_imputed = X_train, X_test  

    for col in categorical_columns:
        X_train_imputed[col] = X_train_imputed[col].astype(str)
        X_test_imputed[col] = X_test_imputed[col].astype(str)

    X_combined = pd.concat([X_train_imputed, X_test_imputed])
    gower_dist_matrix = gower.gower_matrix(X_combined)

    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size)
    clusters = clusterer.fit_predict(gower_dist_matrix.astype(np.float64))

    X_combined['cluster'] = clusters

    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X_combined, y, test_size=test_size, stratify=X_combined['cluster'], random_state=random_state
    )

    X_train_final = X_train_final.drop(columns=['cluster'])
    X_test_final = X_test_final.drop(columns=['cluster'])

    return X_train_final, X_test_final, y_train_final, y_test_final




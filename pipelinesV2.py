import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import FunctionTransformer
import miceforest as mf

class ColumnTypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_columns = None
        self.categorical_columns = None

    def fit(self, X, y=None):
        if isinstance(X, tuple):
            X = X[0]
        self.numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        return self

    def transform(self, X):
        if isinstance(X, tuple):
            X = X[0]
        X = X.copy()
        X[self.numerical_columns] = X[self.numerical_columns].apply(pd.to_numeric, errors='coerce')
        X[self.categorical_columns] = X[self.categorical_columns].astype('category')
        return X

class RemoveMissingYValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is not None:
            mask = ~pd.isnull(y)
            return X[mask], y[mask]
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

class MICEImputer(BaseEstimator, TransformerMixin):
    def __init__(self, datasets=4, iterations=4):
        self.datasets = datasets
        self.iterations = iterations
        self.kernel = None

    def fit(self, X, y=None):
        self.kernel = mf.ImputationKernel(data=X, datasets=self.datasets, save_all_iterations=True)
        self.kernel.mice(self.iterations, verbose=False)
        return self

    def transform(self, X):
        return self.kernel.impute_new_data(new_data=X).complete_data(dataset=0)

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        self.label_encoder.fit(y)
        return self

    def transform(self, X, y=None):
        if y is not None:
            return self.label_encoder.transform(y)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return X, self.transform(y)

class NBPipeline(Pipeline):
    def __init__(self, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        self.label_encoder = LabelEncoder() if task_type == 'classification' else None
        

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, lambda x: x.select_dtypes(include=['int64', 'float64']).columns),
                ('cat', categorical_transformer, lambda x: x.select_dtypes(include=['category', 'object']).columns)
            ])

        steps = [
            ('remove_missing_y', RemoveMissingYValues()),
            ('converter', ColumnTypeConverter()),
            ('mice_imputer', MICEImputer()),
            ('preprocessor', preprocessor),
            ('classifier', GaussianNB())
        ]

        super().__init__(steps)

    def fit(self, X, y):
        mask = ~pd.isnull(y)
        X = X[mask]
        y = y[mask]
        
        # Encode target variable if it's a classification task
        if self.task_type == 'classification' and self.label_encoder is not None:
            y = self.label_encoder.fit_transform(y)
        
        return super().fit(X, y)

    def predict(self, X):
        y_pred = super().predict(X)
        if self.task_type == 'classification' and self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred)
        return y_pred
    
class GLMPipeline(Pipeline):
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        self.label_encoder = LabelEncoder() if task_type == 'classification' else None

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, lambda x: x.select_dtypes(include=['int64', 'float64']).columns),
                ('cat', categorical_transformer, lambda x: x.select_dtypes(include=['category', 'object']).columns)
            ])

        steps = [
            ('remove_missing_y', RemoveMissingYValues()),
            ('converter', ColumnTypeConverter()),
            ('mice_imputer', MICEImputer()),
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ]

        super().__init__(steps)

    def fit(self, X, y):
        # Remove missing y values
        mask = ~pd.isnull(y)
        X = X[mask]
        y = y[mask]
        
        # Continue with the rest of the fit method
        if self.task_type == 'classification' and self.label_encoder is not None:
            y = self.label_encoder.fit_transform(y)
        
        return super().fit(X, y)

    def predict(self, X):
        y_pred = super().predict(X)
        if self.task_type == 'classification' and self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred)
        return y_pred
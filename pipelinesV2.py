import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LinearRegression, LogisticRegression
import miceforest as mf
from sklearn.model_selection import train_test_split

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
        self.has_missing_values = False

    def fit(self, X, y=None):
        self.has_missing_values = X.isnull().any().any()
        if self.has_missing_values:
            self.kernel = mf.ImputationKernel(data=X, datasets=self.datasets, save_all_iterations=True)
            self.kernel.mice(self.iterations, verbose=False)
        return self

    def transform(self, X):
        if self.has_missing_values:
            return self.kernel.impute_new_data(new_data=X).complete_data(dataset=0)
        return X

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

class SaveEncodedY(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoded_y = None

    def fit(self, X, y=None):
        self.encoded_y = y
        return self

    def transform(self, X):
        return X

class SavePreprocessedX(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.preprocessed_X = None

    def fit(self, X, y=None):
        self.preprocessed_X = pd.DataFrame(X, columns=X.columns if hasattr(X, 'columns') else None)
        return self

    def transform(self, X):
        return X

class SaveMissingYMask(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mask = None

    def fit(self, X, y=None):
        if y is not None:
            self.mask = ~pd.isnull(y)
        return self

    def transform(self, X, y=None):
        return X
    
class NBPipeline(Pipeline):
    def __init__(self, task_type='classification', steps=None, memory=None, verbose=False):
        self.task_type = task_type
        self.label_encoder = LabelEncoder() if task_type == 'classification' else None
        
        if steps is None:
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
                ('save_missing_y_mask', SaveMissingYMask()),
                ('converter', ColumnTypeConverter()),
                ('mice_imputer', MICEImputer()),
                ('preprocessor', preprocessor),
                ('save_preprocessed_x', SavePreprocessedX()),
                ('save_encoded_y', SaveEncodedY()),
                ('classifier', GaussianNB())
            ]

        super().__init__(steps=steps, memory=memory, verbose=verbose)

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
    def __init__(self, task_type='classification', steps=None, memory=None, verbose=False):
        self.task_type = task_type
        self.label_encoder = LabelEncoder() if task_type == 'classification' else None

        if steps is None:
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
                ('save_missing_y_mask', SaveMissingYMask()),
                ('converter', ColumnTypeConverter()),
                ('mice_imputer', MICEImputer()),
                ('preprocessor', preprocessor),
                ('save_preprocessed_x', SavePreprocessedX()),
                ('save_encoded_y', SaveEncodedY()),
                ('classifier', LogisticRegression())
            ]

        super().__init__(steps=steps, memory=memory, verbose=verbose)

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


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import FunctionTransformer
import miceforest as mf
from sklearn.model_selection import train_test_split

class ColumnTypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_columns = None
        self.categorical_columns = None

    def fit(self, X, y=None):
        self.numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_columns = X.select_dtypes(include=['object']).columns
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numerical_columns] = X[self.numerical_columns].apply(pd.to_numeric, errors='coerce')
        X[self.categorical_columns] = X[self.categorical_columns].astype('category')
        return X

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

class CombinedNaiveBayes(BaseEstimator):
    def __init__(self):
        self.gnb = GaussianNB()
        self.cnb = CategoricalNB(alpha=1)
        self.numerical_columns = None
        self.categorical_columns = None

    def fit(self, X, y):
        self.numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_columns = X.select_dtypes(include=['category']).columns

        X_num = X[self.numerical_columns]
        X_cat = X[self.categorical_columns]

        self.gnb.fit(X_num, y)
        if not X_cat.empty:
            self.cnb.fit(X_cat, y)
        return self

    def predict(self, X):
        X_num = X[self.numerical_columns]
        log_proba_cont = self.gnb.predict_log_proba(X_num)

        if len(self.categorical_columns) > 0:
            X_cat = X[self.categorical_columns]
            log_proba_cat = self.cnb.predict_log_proba(X_cat)
            log_proba_combined = log_proba_cont + log_proba_cat
        else:
            log_proba_combined = log_proba_cont

        return np.argmax(log_proba_combined, axis=1)

class NaiveBayesPipeline:
    def __init__(self, X_train, y_train, task_type='classification'):
        self.X_train = X_train
        self.y_train = y_train
        self.task_type = task_type
        self.pipeline = None
        self.label_encoder = None

    def create_pipeline(self):
        categorical_transformer = Pipeline(steps=[
            ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        self.pipeline = Pipeline([
            ('converter', ColumnTypeConverter()),
            ('imputer', MICEImputer()),
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, lambda x: x.select_dtypes(include=['category']).columns)
                ],
                remainder='passthrough'
            )),
            ('to_dataframe', FunctionTransformer(lambda x: pd.DataFrame(x, columns=self.X_train.columns), validate=False)),
            ('classifier', CombinedNaiveBayes())
        ])

    def fit(self):
        # Remove rows with missing y values
        mask = ~pd.isnull(self.y_train)
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]

        if self.task_type == 'classification':
            self.label_encoder = LabelEncoder()
            self.y_train = self.label_encoder.fit_transform(self.y_train)

        self.create_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        y_pred = self.pipeline.predict(X_test)
        if self.task_type == 'classification' and self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred)
        return y_pred
    






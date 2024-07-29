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
from sklearn.pipeline import _name_estimators
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if
from sklearn.base import clone



class CustomPipeline(Pipeline):
    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)
        self._validate_steps()
        self._final_estimator_step = self.steps[-1][1]

    def _validate_steps(self):
        names, estimators = zip(*self.steps)
        self._validate_names(names)
        self._named_steps_dict = dict(self.steps)

    @property
    def named_steps(self):
        return self._named_steps_dict

    @property
    def _final_estimator(self):
        return self._final_estimator_step
    
    @property
    def _estimator_type(self):
        return self.steps[-1][1]._estimator_type

    def _fit(self, X, y=None, **fit_params):
        self.steps = list(self.steps)
        self._validate_steps()
        
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = X
        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if transformer is None or transformer == 'passthrough':
                continue

            if hasattr(transformer, "fit_transform"):
                Xt = transformer.fit_transform(Xt, y, **fit_params_steps[name])
            else:
                Xt = transformer.fit(Xt, y, **fit_params_steps[name]).transform(Xt)

        if self._final_estimator != 'passthrough':
            self._final_estimator.fit(Xt, y, **fit_params_steps[self.steps[-1][0]])

        return self

    def _fit_transform_one(self, X, y, weight, fit_params, message_clsname='', message=None, **fit_params_steps):
        if message_clsname and message:
            with _print_elapsed_time(message_clsname, message):
                if weight is None:
                    return self._fit_transform_one_no_weight(X, y, fit_params, **fit_params_steps)
                else:
                    return self._fit_transform_one_weighted(X, y, weight, fit_params, **fit_params_steps)
        else:
            if weight is None:
                return self._fit_transform_one_no_weight(X, y, fit_params, **fit_params_steps)
            else:
                return self._fit_transform_one_weighted(X, y, weight, fit_params, **fit_params_steps)

    def _fit_transform_one_no_weight(self, X, y, fit_params, **fit_params_steps):
        Xt = X
        for name, transformer in self.steps[:-1]:  # excluding final estimator
            if transformer is None or transformer == 'passthrough':
                continue
            if hasattr(transformer, 'fit_transform'):
                Xt = transformer.fit_transform(Xt, y, **fit_params_steps.get(name, {}))
            else:
                Xt = transformer.fit(Xt, y, **fit_params_steps.get(name, {})).transform(Xt)
        return Xt, fit_params

    def _fit_transform_one_weighted(self, X, y, weight, fit_params, **fit_params_steps):
        Xt = X
        for name, transformer in self.steps[:-1]:  # excluding final estimator
            if transformer is None or transformer == 'passthrough':
                continue
            if hasattr(transformer, 'fit_transform'):
                Xt = transformer.fit_transform(Xt, y, sample_weight=weight, **fit_params_steps[name])
            else:
                Xt = transformer.fit(Xt, y, sample_weight=weight, **fit_params_steps[name]).transform(Xt)
        return Xt, fit_params

    def fit_transform(self, X, y=None, **fit_params):
        Xt, fit_params = self._fit_transform_one(
            X, y, None, fit_params, _final_estimator="passthrough"
        )
        if self._final_estimator != "passthrough":
            if hasattr(self._final_estimator, 'fit_transform'):
                Xt = self._final_estimator.fit_transform(Xt, y, **fit_params)
            else:
                self._final_estimator.fit(Xt, y, **fit_params)
                Xt = self._final_estimator.transform(Xt)
        return Xt

    def predict(self, X, **predict_params):
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt, **predict_params)

    def __getitem__(self, ind):
        if isinstance(ind, slice):
            return self.__class__(self.steps[ind])
        elif isinstance(ind, str):
            return self.named_steps[ind]
        elif isinstance(ind, int):
            return self.steps[ind][1]
        raise ValueError(f"Index {ind} is not supported")
    
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
    
class NBPipeline(CustomPipeline):
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

class GLMPipeline(CustomPipeline):
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


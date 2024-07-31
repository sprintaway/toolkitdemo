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

# This is pipelines V3!

# class CustomPipeline(Pipeline):
#     def __init__(self, steps, *, memory=None, verbose=False):
#         super().__init__(steps, memory=memory, verbose=verbose)
#         self._validate_steps()
#         self._final_estimator_step = self.steps[-1][1]

#     @property
#     def named_steps(self):
#         return self._named_steps_dict

#     @property
#     def _final_estimator(self):
#         return self._final_estimator_step
    
#     @property
#     def _estimator_type(self):
#         return self.steps[-1][1]._estimator_type

#     def _fit(self, X, y=None, **fit_params):
#         self.steps = list(self.steps)
#         self._validate_steps()
        
#         fit_params_steps = self._check_fit_params(**fit_params)
#         Xt = X
#         for (step_idx,
#              name,
#              transformer) in self._iter(with_final=False,
#                                         filter_passthrough=False):
#             if transformer is None or transformer == 'passthrough':
#                 continue

#             if hasattr(transformer, "fit_transform"):
#                 Xt = transformer.fit_transform(Xt, y, **fit_params_steps[name])
#             else:
#                 Xt = transformer.fit(Xt, y, **fit_params_steps[name]).transform(Xt)

#         if self._final_estimator != 'passthrough':
#             self._final_estimator.fit(Xt, y, **fit_params_steps[self.steps[-1][0]])

#         return self

#     def _fit_transform_one(self, X, y, weight, fit_params, message_clsname='', message=None, **fit_params_steps):
#         if message_clsname and message:
#             with _print_elapsed_time(message_clsname, message):
#                 if weight is None:
#                     return self._fit_transform_one_no_weight(X, y, fit_params, **fit_params_steps)
#                 else:
#                     return self._fit_transform_one_weighted(X, y, weight, fit_params, **fit_params_steps)
#         else:
#             if weight is None:
#                 return self._fit_transform_one_no_weight(X, y, fit_params, **fit_params_steps)
#             else:
#                 return self._fit_transform_one_weighted(X, y, weight, fit_params, **fit_params_steps)

#     def _fit_transform_one_no_weight(self, X, y, fit_params, **fit_params_steps):
#         Xt = X
#         for name, transformer in self.steps[:-1]:  # excluding final estimator
#             if transformer is None or transformer == 'passthrough':
#                 continue
#             if hasattr(transformer, 'fit_transform'):
#                 Xt = transformer.fit_transform(Xt, y, **fit_params_steps.get(name, {}))
#             else:
#                 Xt = transformer.fit(Xt, y, **fit_params_steps.get(name, {})).transform(Xt)
#         return Xt, fit_params

#     def _fit_transform_one_weighted(self, X, y, weight, fit_params, **fit_params_steps):
#         Xt = X
#         for name, transformer in self.steps[:-1]:  # excluding final estimator
#             if transformer is None or transformer == 'passthrough':
#                 continue
#             if hasattr(transformer, 'fit_transform'):
#                 Xt = transformer.fit_transform(Xt, y, sample_weight=weight, **fit_params_steps[name])
#             else:
#                 Xt = transformer.fit(Xt, y, sample_weight=weight, **fit_params_steps[name]).transform(Xt)
#         return Xt, fit_params

#     def fit_transform(self, X, y=None, **fit_params):
#         Xt, fit_params = self._fit_transform_one(
#             X, y, None, fit_params, _final_estimator="passthrough"
#         )
#         if self._final_estimator != "passthrough":
#             if hasattr(self._final_estimator, 'fit_transform'):
#                 Xt = self._final_estimator.fit_transform(Xt, y, **fit_params)
#             else:
#                 self._final_estimator.fit(Xt, y, **fit_params)
#                 Xt = self._final_estimator.transform(Xt)
#         return Xt

#     def predict(self, X, **predict_params):
#         Xt = X
#         for _, _, transform in self._iter(with_final=False):
#             Xt = transform.transform(Xt)
#         return self.steps[-1][-1].predict(Xt, **predict_params)

# class CustomColumnTransformer(ColumnTransformer):
#     def fit(self, X, y=None):
#         super().fit(X, y)
#         print("Fitting X!", X)
#         return self

#     def fit_transform(self, X, y=None):
#         Xt = super().fit_transform(X, y)
#         print("Fitting + Transforming X!", X)
#         return Xt, y

#     def transform(self, X, y=None):
#         Xt = super().transform(X)
#         print("Transforming X!", X)
#         if y is None:
#             return Xt
#         return Xt, y

# Because making it inherit from columntransformer just doesn't work :(

class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformers):
        self.transformers = transformers
        self.fitted_transformers = None

    def fit(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X

        self.fitted_transformers = []
        for name, transformer, columns in self.transformers:
            if callable(columns):
                columns = columns(X)
            X_subset = X[columns]
            fitted_transformer = clone(transformer).fit(X_subset, y)
            self.fitted_transformers.append((name, fitted_transformer, columns))
        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X

        result = X.copy()
        for name, transformer, columns in self.fitted_transformers:
            if callable(columns):
                columns = columns(X)
            X_subset = X[columns]
            transformed = transformer.transform(X_subset)
            
            if isinstance(transformed, np.ndarray) and transformed.ndim == 2:
                for i, col in enumerate(columns):
                    result[col] = transformed[:, i]
            else:
                result[columns] = transformed
        
        print(f"CustomColumnTransformer output shape: {result.shape}")
        print(f"Result (X) is now: {result}")
        return (result, y) if y is not None else result

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

class CustomPipeline(Pipeline):
    def fit(self, X, y=None, **fit_params):
        Xt = X
        yt = y
        for name, transform in self.steps[:-1]:
            print(f"Fitting step: {name}")
            print(f"Before {name}: X shape = {Xt.shape}, y type = {type(yt)}")
            if transform is not None:
                if hasattr(transform, "fit_transform"):
                    result = transform.fit_transform(Xt, yt, **fit_params)
                else:
                    transform.fit(Xt, yt, **fit_params)
                    result = transform.transform(Xt, yt)
                
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        Xt, yt = result[:2]
                    else:
                        Xt = result[0]
                else:
                    Xt = result
            print(f"After {name}: X shape = {Xt.shape}, y type = {type(yt)}")
            if isinstance(yt, pd.Series):
                print(f"y unique values: {yt.unique()}")
            elif isinstance(yt, np.ndarray):
                print(f"y unique values: {np.unique(yt)}")
        
        # Fit the final estimator
        self.steps[-1][1].fit(Xt, yt, **fit_params)
        return self
    
    def fit_transform(self, X, y=None, **fit_params):
        Xt = X
        yt = y
        for name, transform in self.steps[:-1]:
            print(f"Processing step: {name}")
            print(f"Before {name}: X shape = {Xt.shape}, y type = {type(yt)}")
            if transform is not None:
                if hasattr(transform, "fit_transform"):
                    result = transform.fit_transform(Xt, yt, **fit_params)
                else:
                    transform.fit(Xt, yt, **fit_params)
                    result = transform.transform(Xt, yt)
                
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        Xt, yt = result[:2]
                    else:
                        Xt = result[0]
                else:
                    Xt = result
            print(f"After {name}: X shape = {Xt.shape}, y type = {type(yt)}")
            if isinstance(yt, pd.Series):
                print(f"y unique values: {yt.unique()}")
            elif isinstance(yt, np.ndarray):
                print(f"y unique values: {np.unique(yt)}")
        return Xt, yt

    def transform(self, X, y=None):
        Xt = X
        yt = y
        for name, transform in self.steps[:-1]:
            print(f"Before {name}: X shape = {Xt.shape}, y type = {type(yt)}")
            if transform is not None:
                result = transform.transform(Xt, yt)
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        Xt, yt = result[:2]
                    else:
                        Xt = result[0]
                else:
                    Xt = result
            print(f"After {name}: X shape = {Xt.shape}, y type = {type(yt)}")
        return Xt, yt
    

class ColumnTypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_columns = None
        self.categorical_columns = None

    def fit(self, X, y=None):
        if isinstance(X, tuple):
            X = X[0]#[0][0]
            #print("Fitting, returning X", X)
        self.numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X[0][0][0], X[1]
            print("Transforming, returning X", X)  
        X = X.copy()
        X[self.numerical_columns] = X[self.numerical_columns].apply(pd.to_numeric, errors='coerce')
        X[self.categorical_columns] = X[self.categorical_columns].astype('category')
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        print("Fitting + Transforming, returning X", X)
        return self.transform(X, y)
    

class RemoveMissingYValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is not None:
            mask = ~pd.isnull(y)
            return X[mask], y[mask]
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

class MICEImputer(BaseEstimator, TransformerMixin):
    def __init__(self, datasets=4, iterations=4):
        self.datasets = datasets
        self.iterations = iterations
        self.kernel = None
        self.has_missing_values = False

    def fit(self, X, y=None):
        #if isinstance(X, tuple):
            #X = X[0]
        print("X columns imputation fit!", X.dtypes)
        self.has_missing_values = X.isnull().any().any()
        if self.has_missing_values:
            self.kernel = mf.ImputationKernel(data=X, datasets=self.datasets, save_all_iterations=True)
            self.kernel.mice(self.iterations, verbose=False)
        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X = X[0]
        print("Before imputation X", X)
        print("X columns imputation transform!", X.dtypes)
        if self.has_missing_values:
            X = self.kernel.impute_new_data(new_data=X).complete_data(dataset=0)
        print("Imputation of X done!")
        print(f"MICEImputer output shape: {X.shape}")
        return X, y
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        self.label_encoder.fit(y)
        return self

    def transform(self, X, y=None):
        if y is not None:
            return self.label_encoder.transform(y)
        return X, y

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return X, self.transform(y)

class SaveEncodedY(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        if y is not None:
            self.label_encoder.fit(y)
        return self

    def transform(self, X, y=None):
        if y is not None:
            y = pd.Series(self.label_encoder.transform(y))
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)
    
    def inverse_transform(self, y):
        return self.label_encoder.inverse_transform(y)

class SavePreprocessedX(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.preprocessed_X = None

    def fit(self, X, y=None):
        self.preprocessed_X = pd.DataFrame(X, columns=X.columns if hasattr(X, 'columns') else None)
        return self

    def transform(self, X, y=None):
        return X, y
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

class SaveMissingYMask(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mask = None

    def fit(self, X, y=None):
        if y is not None:
            self.mask = ~pd.isnull(y)
        return self

    def transform(self, X, y=None):
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)
    
class NBPipeline(CustomPipeline):
    def __init__(self, task_type='classification', steps=None, memory=None, verbose=False):
        self.task_type = task_type
        self.missing_y_mask = None
        
        if steps is None:
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            preprocessor = CustomColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, lambda x: x.select_dtypes(include=['int64', 'float64']).columns),
                    ('cat', categorical_transformer, lambda x: x.select_dtypes(include=['category', 'object']).columns)
                ])

            steps = [
                ('remove_missing_y', RemoveMissingYValues()),
                ('save_missing_y_mask', SaveMissingYMask()),
                ('save_encoded_y', SaveEncodedY()),
                ('converter', ColumnTypeConverter()),
                ('mice_imputer', MICEImputer()),
                ('preprocessor', preprocessor),
                ('save_preprocessed_x', SavePreprocessedX()),
                ('classifier', GaussianNB())
            ]

        super().__init__(steps=steps, memory=memory, verbose=verbose)

    def fit(self, X, y):
        self.missing_y_mask = ~pd.isnull(y)
        return super().fit(X, y)

    def predict(self, X):
        if isinstance(X, tuple):
            X = X[0]  
        y_pred = super().predict(X)
        y_pred_original = self.named_steps['save_encoded_y'].inverse_transform(y_pred)
        return y_pred_original

    def score(self, X, y):
        print("Scoring X!", X)
        if isinstance(X, tuple):
            X = X[0]  
        y_pred = self.predict(X)
        mask = self.missing_y_mask if self.missing_y_mask is not None else ~pd.isnull(y)
        y_clean = y[mask]
        y_pred_clean = y_pred[mask]
        return (y_clean == y_pred_clean).mean()

class GLMPipeline(CustomPipeline):
    def __init__(self, task_type='classification', steps=None, memory=None, verbose=False):
        self.task_type = task_type
        self.missing_y_mask = None

        if steps is None:
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = CustomColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, lambda x: x.select_dtypes(include=['int64', 'float64']).columns),
                    ('cat', categorical_transformer, lambda x: x.select_dtypes(include=['category', 'object']).columns)
                ])

            steps = [
                ('remove_missing_y', RemoveMissingYValues()),
                ('save_missing_y_mask', SaveMissingYMask()),
                ('save_encoded_y', SaveEncodedY()),
                ('converter', ColumnTypeConverter()),
                ('mice_imputer', MICEImputer()),
                ('preprocessor', preprocessor),
                ('save_preprocessed_x', SavePreprocessedX()),
                ('classifier', LogisticRegression())
            ]

        super().__init__(steps=steps, memory=memory, verbose=verbose)

    def fit(self, X, y):
        self.missing_y_mask = ~pd.isnull(y)
        return super().fit(X, y)

    def predict(self, X):
        if isinstance(X, tuple):
            X = X[0]
        y_pred = super().predict(X)
        y_pred_original = self.named_steps['save_encoded_y'].inverse_transform(y_pred)
        return y_pred_original

    def score(self, X, y):
        print("Scoring X!", X)
        if isinstance(X, tuple):
            X = X[0]  
        y_pred = self.predict(X)
        mask = self.missing_y_mask if self.missing_y_mask is not None else ~pd.isnull(y)
        y_clean = y[mask]
        y_pred_clean = y_pred[mask]
        return (y_clean == y_pred_clean).mean()

    def predict_proba(self, X):
        if isinstance(X, tuple):
            X = X[0]
        return self.steps[-1][1].predict_proba(self.transform(X)[0])


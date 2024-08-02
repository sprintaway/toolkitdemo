import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from sklearn.feature_selection import f_classif, f_regression
from itertools import combinations
from celer import GroupLassoCV, GroupLasso
import miceforest as mf
import category_encoders as ce
from sklearn.linear_model import LogisticRegression, LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import ttest_ind
from sklearn.inspection import permutation_importance
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureSelector:
    def __init__(self, X, y, task_type='classification'):
        self.X = X
        self.y = y
        self.task_type = task_type
        self.numerical_columns = None
        self.categorical_columns = None
        self.imputed_X = None  
        self.methods = {
            'classification': [
                'permutation',
                'symmetrical_uncertainty',
                'f_score',
                'jmim',
                'LASSO',
                'sequential',
                'random_forest',
            ],
            'regression': [
                'permutation',
                'symmetrical_uncertainty',
                'f_score',
                'jmim',
                'LASSO',
                'sequential',
                'random_forest'
            ]
        }
        self.results = {}

    @staticmethod
    def is_numeric_column(col):
        col_clean = col.dropna().replace('', np.nan).dropna()
        return pd.to_numeric(col_clean, errors='coerce').notna().all()

    def run_selection(self):
        self.X = self.X.copy()
        self.X = self.X.loc[self.y.notna()]
        self.y = self.y.loc[self.y.notna()]

        self.numerical_columns = [col for col in self.X.columns if FeatureSelector.is_numeric_column(self.X[col])]
        self.categorical_columns = [col for col in self.X.columns if col not in self.numerical_columns]

        self.X[self.numerical_columns] = self.X[self.numerical_columns].apply(pd.to_numeric, errors='coerce')
        for col in self.categorical_columns:
            self.X[col] = self.X[col].astype('category')

        if self.X.isnull().sum().sum() > 0:
            kernel = mf.ImputationKernel(self.X, save_all_iterations=True, random_state=0)
            self.imputed_X = kernel.complete_data()
        else:
            self.imputed_X = self.X.copy()  

        methods = self.methods['classification'] if self.task_type == 'classification' else self.methods['regression']

        for method in methods:
            print(f"Running {method} feature selection...")
            if method == 'jmim':
                selected_features = getattr(self.__class__, f"{method}_feature_selection")(self.imputed_X, self.y, self.task_type, max_features=50)
            else:
                selected_features = getattr(self.__class__, f"{method}_feature_selection")(self.imputed_X, self.y, self.task_type)
            self.results[method] = selected_features

    def feature_plot(self, features):
        for feature in features:
            fig, ax = plt.subplots(figsize=(10, 6))

            is_categorical = (not self.is_numeric_column(self.X[feature])) or (self.X[feature].nunique() < 25)

            if not is_categorical:
                if self.task_type == 'regression':
                    sns.scatterplot(x=self.imputed_X[feature], y=self.y, ax=ax)
                    ax.set_title(f'{feature} vs {self.y.name} (Scatter Plot)')
                else:
                    sns.boxplot(x=self.y, y=self.imputed_X[feature], ax=ax)
                    ax.set_title(f'{self.y.name} vs {feature} (Box Plot)')
            else:
                if self.task_type == 'regression':
                    sns.boxplot(x=self.X[feature], y=self.y, ax=ax)  
                    ax.set_title(f'{feature} vs {self.y.name} (Box Plot)')
                else:
                    # Use original X for categorical features
                    contingency_table = pd.crosstab(self.X[feature], self.y, normalize='index')
                    contingency_table.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title(f'{feature} vs {self.y.name} (Stacked Bar Chart)')

            ax.set_xlabel(self.y.name if not is_categorical and self.task_type != 'regression' else feature)
            ax.set_ylabel(feature if not is_categorical and self.task_type != 'regression' else self.y.name)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    def get_results_table(self):
        all_features = self.X.columns
        methods = list(self.results.keys())

        results_dict = {feature: [] for feature in all_features}

        for method in methods:
            selected_features = self.results[method]
            for feature in all_features:
                if feature in selected_features:
                    results_dict[feature].append('✓')
                else:
                    results_dict[feature].append('✗')

        results_df = pd.DataFrame(results_dict, index=methods)
        return results_df.transpose()

    def get_count(self):
        results_df = self.get_results_table()

        feature_counts = results_df.apply(lambda row: row.value_counts().get('✓', 0), axis=1)

        total_methods = len(results_df.columns)

        count_df = pd.DataFrame({
            'Count': feature_counts.astype(str) + " / " + str(total_methods)
        })

        count_df = count_df.sort_values('Count', ascending=False)

        return count_df

    def all_features_selected(self):
        results_df = self.get_results_table()
        return results_df[results_df.eq('✓').all(axis=1)].index.tolist()

    def no_features_selected(self):
        results_df = self.get_results_table()
        return results_df[results_df.eq('✗').all(axis=1)].index.tolist()

    def selected_by_at_least(self, number=1):

        total_methods = len(self.methods[self.task_type])
        if number > total_methods:
            raise ValueError(f"Invalid input: 'number' ({number}) cannot exceed the number of feature selection methods ({total_methods}).")

        count_df = self.get_count()

        count_df["NumericCount"] = count_df["Count"].apply(
            lambda x: int(x.split(" / ")[0])
        )

        return count_df[count_df['NumericCount'] >= number].index.tolist()
    
    def selected_by_at_most(self, number=1):
        
        total_methods = len(self.methods[self.task_type])
        if number > total_methods:
            raise ValueError(f"Invalid input: 'number' ({number}) cannot exceed the number of feature selection methods ({total_methods}).")

        count_df = self.get_count()

        count_df["NumericCount"] = count_df["Count"].apply(
            lambda x: int(x.split(" / ")[0])
        )

        return count_df[count_df['NumericCount'] <= number].index.tolist()


    def symmetrical_uncertainty_feature_selection(X, y, task_type='classification'):
        """
        Feature selection using symmetrical uncertainty (SU) with permutation testing for significance.

        :param X: Pandas DataFrame containing features.
        :param y: Pandas Series or array-like containing the target variable.
        :param task_type: 'classification' or 'regression' (default='classification').
        :return: NumPy array of selected feature names.
        """

        mi_func = mutual_info_regression if task_type == 'regression' else mutual_info_classif

        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)

        n_features = X.shape[1]
        selected_features = []
        threshold_values = []

        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        oe = OrdinalEncoder()
        X[categorical_features] = oe.fit_transform(X[categorical_features])

        #print("Feature\t\tMutual Information\tThreshold")
        #print("-" * 50)

        for i in range(n_features):
            xi = X.iloc[:, i].values
            theta_i = mi_func(xi.reshape(-1, 1), y, discrete_features='auto')[0]  

            theta_perm = np.zeros(100)
            for j in range(100):
                xi_permuted = shuffle(xi)
                theta_perm[j] = mi_func(xi_permuted.reshape(-1, 1), y, discrete_features='auto')[0]

            theta_c = np.percentile(theta_perm, 95)
            threshold_values.append(theta_c)

            #print(f"{X.columns[i]:<15}{theta_i:<25}{theta_c:<15}")

            if theta_i > theta_c:
                selected_features.append(X.columns[i])

        #print("\nSelected features:", selected_features)
        return np.array(selected_features)


    def f_score_feature_selection(X, y, task_type='classification'):
        """
        Feature selection using F-test (f_classif or f_regression) with a p-value threshold.

        :param X: Pandas DataFrame containing features.
        :param y: Pandas Series or array-like containing the target variable.
        :param task_type: 'classification' or 'regression' (default='classification').
        :return: List of selected feature names.
        """

        f_func = f_regression if task_type == 'regression' else f_classif

        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)

        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        oe = OrdinalEncoder()
        X[categorical_features] = oe.fit_transform(X[categorical_features])

        f_scores, p_values = f_func(X, y)
        significant_features = X.columns[p_values < 0.05]

        #print("\nF-scores and p-values:")
        #for feature, f_score, p_value in zip(X.columns, f_scores, p_values):
            #print(f"{feature}: F-score = {f_score:.4f}, p-value = {p_value:.4f}")

        #print(f"\nSelected features:", list(significant_features))
        return list(significant_features)


    def jmim_feature_selection(X, y, task_type='classification', max_features=50):
        """
        Performs Joint Mutual Information Maximization (JMIM) feature selection.

        :param X: Pandas DataFrame containing features.
        :param y: Pandas Series or array-like containing the target variable.
        :param task_type: 'classification' or 'regression'.
        :return: List of selected features.
        """

        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[categorical_features] = oe.fit_transform(X[categorical_features])

        if task_type == 'regression':
            mutual_info_func = mutual_info_regression
            y = y.astype(float)  
        else:
            mutual_info_func = mutual_info_classif
            le = LabelEncoder()
            y = le.fit_transform(y)

        features = X.columns.tolist()
        
        mi_scores = mutual_info_func(X[features], y, discrete_features='auto', random_state=0)
        mi_scores = pd.Series(mi_scores, index=X.columns)  

        selected_features = []
        best_jmim = -np.inf

        while len(selected_features) < max_features:
            max_jmim = best_jmim  
            best_feature_to_add = None
            
            for feature_to_add in combinations(set(features) - set(selected_features), 1):
                candidate_features = selected_features + list(feature_to_add)
                joint_mi = mutual_info_func(X[candidate_features], y, discrete_features='auto', random_state=0)[0]

                min_mi = joint_mi  
                for selected in selected_features:
                    pairwise_mi = mutual_info_func(X[[feature_to_add[0], selected]], y, discrete_features='auto', random_state=0)[0]
                    min_mi = min(min_mi, pairwise_mi)  

                jmim_score = mi_scores[feature_to_add[0]] - min_mi  

                if jmim_score > 0 and jmim_score > max_jmim:
                    max_jmim = jmim_score
                    best_feature_to_add = feature_to_add[0]  

            if best_feature_to_add is not None:  
                selected_features.append(best_feature_to_add)
                #print(f"Selected Feature: {best_feature_to_add} (JMIM: {max_jmim:.4f})")
            else:  
                break
                
        return selected_features


    def LASSO_feature_selection(X, y, task_type='classification'):
        """
        Performs feature selection using Group LASSO with GLMM or OHE encoding.

        :param X: Pandas DataFrame containing features.
        :param y: Pandas Series or array-like containing the target variable.
        :param task_type: 'classification' or 'regression'.
        :return: List of selected features.
        """

        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(exclude=['object', 'category']).columns

        encoded_feature_names = []
        X_encoded_parts = [X[numerical_features]]

        feature_groups = [[i] for i in range(len(numerical_features))]
        start_idx = len(numerical_features)
        
        for cat_feature in categorical_features:
            if X[cat_feature].nunique() >= 10:
                encoder = ce.GLMMEncoder(cols=[cat_feature])
                X_encoded_part = encoder.fit_transform(X[[cat_feature]], y)
                X_encoded_parts.append(X_encoded_part)
                encoded_feature_names.extend([f"{cat_feature}_GLMM"])
                feature_groups.append([start_idx])
                start_idx += 1
            else:
                encoder = ce.OneHotEncoder(cols=[cat_feature])
                X_encoded_part = encoder.fit_transform(X[[cat_feature]])
                encoded_feature_names.extend(encoder.get_feature_names_out([cat_feature]))
                num_dummy_vars = X_encoded_part.shape[1]
                feature_groups.append(list(range(start_idx, start_idx + num_dummy_vars)))
                X_encoded_parts.append(X_encoded_part)
                start_idx += num_dummy_vars

        X_encoded = pd.concat(X_encoded_parts, axis=1)
        feature_names = list(numerical_features) + encoded_feature_names

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        if task_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            model_cv = GroupLassoCV(groups=feature_groups, eps=0.01, n_alphas=100, cv=5)
            model_cv.fit(X_scaled, y_encoded)
            group_lasso = GroupLasso(alpha=model_cv.alpha_, groups=feature_groups)
            group_lasso.fit(X_scaled, y_encoded)
        else:
            model_cv = GroupLassoCV(groups=feature_groups, eps=0.01, n_alphas=100, cv=5)
            model_cv.fit(X_scaled, y)
            group_lasso = GroupLasso(alpha=model_cv.alpha_, groups=feature_groups)
            group_lasso.fit(X_scaled, y)
        
        selected_original_features = []
        feature_importances = {}

        # Get feature importances for GLMM and OHE encoded features
        for i, col in enumerate(feature_names):
            importance = abs(group_lasso.coef_[i])
            if importance > 0:
                if col.endswith('_GLMM'):
                    selected_original_features.append(col.split('_GLMM')[0])
                    feature_importances[col.split('_GLMM')[0]] = importance
                elif col in X.columns:  # Check if feature exists in original columns
                    selected_original_features.append(col)  # It's a numerical feature
                    feature_importances[col] = importance
                else:  # One-hot encoded feature
                    original_feature = col.split('_')[0]
                    selected_original_features.append(original_feature)
                    if original_feature not in feature_importances:
                        feature_importances[original_feature] = 0
                    feature_importances[original_feature] += importance

        for feature in categorical_features:
            if feature not in selected_original_features:
                continue
            dummy_cols = [c for c in feature_names if c.startswith(f"{feature}_")]
            feature_importances[feature] /= len(dummy_cols)

        selected_original_features = list(set(selected_original_features))
        importance_df = pd.DataFrame.from_dict(feature_importances, orient='index', columns=['Importance'])
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        #print("\nFeature Importance Ranking:")
        #print(importance_df)

        return selected_original_features


    def sequential_feature_selection(X, y, task_type='classification'):
        """
        Performs sequential feature selection with an automatic threshold.

        :param X: Pandas DataFrame containing features.
        :param y: Pandas Series or array-like containing the target variable.
        :param task_type: 'classification' or 'regression'.
        :return: List of selected features.
        """

        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(exclude=['object', 'category']).columns
        
        encoded_feature_names = []
        X_encoded_parts = [X[numerical_features]]

        feature_groups = [[i] for i in range(len(numerical_features))]
        start_idx = len(numerical_features)
        
        for cat_feature in categorical_features:
            if X[cat_feature].nunique() >= 10:
                encoder = ce.GLMMEncoder(cols=[cat_feature])
                X_encoded_part = encoder.fit_transform(X[[cat_feature]], y)
                X_encoded_parts.append(X_encoded_part)
                encoded_feature_names.extend([f"{cat_feature}_GLMM"])
                feature_groups.append([start_idx])
                start_idx += 1
            else:
                encoder = ce.OneHotEncoder(cols=[cat_feature])
                X_encoded_part = encoder.fit_transform(X[[cat_feature]])
                encoded_feature_names.extend(encoder.get_feature_names_out([cat_feature]))
                num_dummy_vars = X_encoded_part.shape[1]
                feature_groups.append(list(range(start_idx, start_idx + num_dummy_vars)))
                X_encoded_parts.append(X_encoded_part)
                start_idx += num_dummy_vars

        X_encoded = pd.concat(X_encoded_parts, axis=1)
        feature_names = list(numerical_features) + encoded_feature_names
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        if task_type == "classification":
            model = LogisticRegression(max_iter=1000)
            scoring = 'accuracy'
        else:
            model = LinearRegression()
            scoring = 'r2'
        
        sfs = SFS(
            model,
            k_features='best',
            forward=False,
            floating=False,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            feature_groups=feature_groups
        )
        
        sfs = sfs.fit(X_scaled, y)
        
        selected_features_indices = list(sfs.k_feature_idx_)
        selected_feature_names = [feature_names[i] for i in selected_features_indices]

        selected_original_features = []
        for i, feature_group in enumerate(categorical_features):
            for feature in selected_feature_names:
                if feature.startswith(feature_group + '_'):
                    selected_original_features.append(feature_group)
                    break

        selected_original_features.extend([f for f in numerical_features if f in selected_feature_names])
        
        #print(f"\nSelected original features ({task_type}):", selected_original_features)
        
        return selected_original_features


    def random_forest_feature_selection(X, y, task_type):

        """
        Performs random forest feature selection.

        :param X: Pandas DataFrame containing features.
        :param y: Pandas Series or array-like containing the target variable.
        :param task_type: 'classification' or 'regression'.
        :return: List of selected features.
        """

        if task_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

        encoder = ce.GLMMEncoder()
        X_train_encoded = encoder.fit_transform(X_train, y_train)
        X_test_encoded = encoder.transform(X_test)
        
        if task_type == 'classification':
            model = RandomForestClassifier
            metric = accuracy_score
            positive_performance = True  
        else:
            model = RandomForestRegressor
            metric = mean_squared_error
            positive_performance = False  
        
        feature_importance = {col: [] for col in X.columns}
        performance_drops = []
        all_dropped_features = []

        for i in range(50):
            full_model = model(random_state=0)
            full_model.fit(X_train_encoded, y_train)
            
            y_pred = full_model.predict(X_test_encoded)
            full_performance = metric(y_test, y_pred)
            
            features = list(X_train_encoded.columns)
            np.random.shuffle(features)
            dropped_features = features[:len(features)//2]
            all_dropped_features.append(dropped_features)
            
            X_train_dropped = X_train_encoded.drop(columns=dropped_features)
            X_test_dropped = X_test_encoded.drop(columns=dropped_features)
            
            dropped_model = model(random_state=0)
            dropped_model.fit(X_train_dropped, y_train)
            
            y_pred_dropped = dropped_model.predict(X_test_dropped)
            dropped_performance = metric(y_test, y_pred_dropped)
            
            performance_drop = full_performance - dropped_performance if positive_performance else dropped_performance - full_performance
            performance_drops.append(performance_drop)
            
            for feature in dropped_features:
                feature_importance[feature].append(performance_drop)
        
        mean_performance_drops = {feature: np.mean(drops) for feature, drops in feature_importance.items()}
        variance_performance_drops = {feature: np.var(drops) for feature, drops in feature_importance.items()}
        
        significant_features = []

        for feature, drops in feature_importance.items():
            drops_without_feature = [drop for i, drop in enumerate(performance_drops) if feature not in all_dropped_features[i]]
            
            t_stat, p_value = ttest_ind(drops, drops_without_feature)
            
            #print(f"Feature: {feature}, p-value: {p_value}")
            #print(f"Performance drops when feature {feature} is included: {drops}")
            #print(f"Performance drops when feature {feature} is not included: {drops_without_feature}")

            if p_value < 0.05 and mean_performance_drops[feature] > 0:
                significant_features.append(feature)
        
        return significant_features

    def permutation_feature_selection(X, y, task_type):
        """
        Performs permutation feature selection.

        :param X: Pandas DataFrame containing features.
        :param y: Pandas Series or array-like containing the target variable.
        :param task_type: 'classification' or 'regression'.
        :return: List of selected features.
        """
        
        if task_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
        
        # Encode categorical variables
        encoder = ce.GLMMEncoder()
        X_encoded = encoder.fit_transform(X, y_encoded)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=0)

        # Initialize model based on task type
        if task_type == 'classification':
            model = RandomForestClassifier(random_state=0)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(random_state=0)
            scoring = 'neg_mean_squared_error'
        
        # Fit the model
        model.fit(X_train, y_train)

        # Get original importances
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0, scoring=scoring)
        original_importances = result.importances_mean

        # Initialize array for permuted importances
        n_repeats = 10
        permuted_importances = np.zeros((n_repeats, X.shape[1]))

        for i in range(n_repeats):
            y_permuted = np.random.permutation(y_test)
            permuted_result = permutation_importance(model, X_test, y_permuted, n_repeats=10, random_state=0, scoring=scoring)
            permuted_importances[i, :] = permuted_result.importances_mean

        # Calculate p-values
        p_values = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            observed_importance = original_importances[i]
            permuted_distribution = permuted_importances[:, i]
            mean = np.mean(permuted_distribution)
            std = np.std(permuted_distribution)
            if std > 0:
                z_score = (observed_importance - mean) / std
                p_values[i] = 2 * (1 - norm.cdf(np.abs(z_score)))  
            else:
                p_values[i] = 1.0  

        # Select significant features
        significant_features = [X.columns[i] for i in range(len(p_values)) if p_values[i] < 0.05]

        return significant_features

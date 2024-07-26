import numpy as np
import pandas as pd
import miceforest as mf
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.sod import SOD
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class OutlierDetector:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.contamination_level = 0.1
        self.detectors = {
            'HBOS': HBOS(n_bins='auto', contamination=self.contamination_level),
            'KNN': KNN(contamination=self.contamination_level),
            'LOF': LOF(contamination=self.contamination_level),
            'SOD': SOD(contamination=self.contamination_level),
            'IForest': IForest(contamination=self.contamination_level),
            'OCSVM': OCSVM(contamination=self.contamination_level),
            'ABOD': ABOD(contamination=self.contamination_level)
        }
        self.results = None
        self.numerical_columns = None
        self.imputed_data = None

    def _get_numerical_columns(self):
        return self.X_train.select_dtypes(include=[np.number]).columns

    def _impute_missing_values(self):
        if self.X_train[self.numerical_columns].isnull().any().any():

            kernel = mf.ImputationKernel(
                self.X_train[self.numerical_columns],
                save_all_iterations=True,
                random_state=0
            )
            self.imputed_data = kernel.complete_data()
        else:
            self.imputed_data = self.X_train[self.numerical_columns]

    def pcaplot(self, outlier_indices=None):
        if self.results is None:
            raise ValueError("Please run 'run_selection()' method first.")
        
        if self.imputed_data is None:
            raise ValueError("No imputed data available. Make sure run_selection() has been called.")

        if self.numerical_columns is None or len(self.numerical_columns) == 0:
            raise ValueError("No numerical columns found in the dataset.")

        numerical_data = self.imputed_data[self.numerical_columns]

        pca = PCA(n_components='mle', svd_solver='full')
        pca_result = pca.fit_transform(numerical_data)

        pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

        if outlier_indices is None:
            outlier_indices = []

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], alpha=0.3, color='blue', label='Normal')

        outliers = pca_df.iloc[outlier_indices]
        ax.scatter(outliers.iloc[:, 0], outliers.iloc[:, 1], alpha=0.7, color='red', s=50, label='Outliers')

        ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title('PCA of Numerical Data with Outliers Highlighted')
        ax.legend()

        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def run_selection(self):
        self.numerical_columns = self._get_numerical_columns()
        if len(self.numerical_columns) == 0:
            raise ValueError("No numerical columns found in the dataset.")
        
        self._impute_missing_values()
        X_numerical = self.imputed_data.to_numpy()
        self.results = pd.DataFrame(index=range(len(self.X_train)))
        
        for name, detector in self.detectors.items():
            detector.fit(X_numerical)
            predictions = detector.predict(X_numerical)
            self.results[name] = predictions
        
        return self.results

    def get_results_table(self):
        if self.results is None:
            raise ValueError("Please run 'run_selection()' method first.")
        
        formatted_results = self.results.map(lambda x: '✓' if x == 1 else '✗')
        return formatted_results

    def get_count(self):
        if self.results is None:
            raise ValueError("Please run 'run_selection()' method first.")
        
        outlier_counts = self.results.sum(axis=1)
        
        total_detectors = len(self.detectors)
        
        count_df = pd.DataFrame({
            'Count': outlier_counts.astype(str) + " / " + str(total_detectors)
        })

        count_df = count_df.sort_values('Count', ascending=False)
        
        return count_df

    def all_detection_selected(self):
        if self.results is None:
            raise ValueError("Please run 'run_selection()' method first.")
        
        all_selected = self.results[self.results.sum(axis=1) == len(self.detectors)].index.tolist()
        return all_selected

    def no_detection_selected(self):
        if self.results is None:
            raise ValueError("Please run 'run_selection()' method first.")
        
        none_selected = self.results[self.results.sum(axis=1) == 0].index.tolist()
        return none_selected

    def selected_by_at_least(self, number=1):
        if self.results is None:
            raise ValueError("Please run 'run_selection()' method first.")
        
        total_detectors = len(self.detectors)
        if number > total_detectors:
            raise ValueError(f"Invalid input: 'number' ({number}) cannot exceed the number of detectors ({total_detectors}).")
        
        selected = self.results[self.results.sum(axis=1) >= number].index.tolist()
        return selected

    def get_numerical_columns(self):
        if self.numerical_columns is None:
            raise ValueError("Please run 'run_selection()' method first.")
        return list(self.numerical_columns)
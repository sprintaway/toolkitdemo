import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, ks_2samp, chisquare
from sklearn.model_selection import train_test_split

class DistributionComparison:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.categorical_features = X_train.select_dtypes(include=['object', 'category'])
        self.numerical_features = X_train.select_dtypes(include=['float64', 'int64', 'int32'])

    def plot_categorical_distribution(self, col_name):
        """
        Visualizes the distribution of a categorical feature in the training and test sets.

        Args:
            col_name (str): Name of the categorical column to plot.
        """

        plt.figure(figsize=(10, 6)) 

        train_counts = self.X_train[col_name].value_counts(normalize=True).sort_index()
        test_counts = self.X_test[col_name].value_counts(normalize=True).sort_index()

        all_categories = train_counts.index.union(test_counts.index)
        train_counts = train_counts.reindex(all_categories, fill_value=0)
        test_counts = test_counts.reindex(all_categories, fill_value=0)

        width = 0.4
        x = range(len(all_categories))

        plt.bar(train_counts.index, train_counts, width=width, alpha=0.5, label='Training Set')
        plt.bar(test_counts.index, test_counts, width=width, alpha=0.5, label='Test Set', color='orange')

        plt.title(f"Distribution of {col_name}", fontsize=14)
        plt.xlabel(col_name)
        plt.ylabel("Proportion")
        plt.xticks(x, all_categories, rotation=45, ha="right")  
        plt.legend()
        plt.tight_layout()  
        plt.show()


    def compare_categorical_distributions(self):
        """
        Compares the distributions of categorical features in training and test sets
        using the G-test and applies Bonferroni correction for multiple comparisons.
        """

        # Bonferroni Correction
        alpha = 0.05
        num_categorical_features = len(self.categorical_features.columns)
        adjusted_alpha = alpha / num_categorical_features

        for col in self.categorical_features:
            self.plot_categorical_distribution(col)

            # Perform G-test and compare p-value with adjusted alpha
            p_value = self.perform_g_test(col)

            if p_value < adjusted_alpha:
                print(
                    f"  - Conclusion: The distribution of {col} is significantly different between the train and test sets (reject null hypothesis after Bonferroni correction).\n"
                )
            else:
                print(
                    f"  - Conclusion: The distribution of {col} is not significantly different between the train and test sets (fail to reject null hypothesis after Bonferroni correction).\n"
                )

    def perform_g_test(self, col_name): 
        """
        Performs the G-test (log-likelihood ratio test) for the specified 
        categorical feature and prints the results.

        Args:
            col_name (str): The name of the categorical column to test.
        """

        # Create contingency table
        train_counts_raw = self.X_train[col_name].value_counts()
        test_counts_raw = self.X_test[col_name].value_counts()

        # Filter out categories with less than 5 counts in either train or test set
        valid_categories = train_counts_raw[train_counts_raw >= 5].index.intersection(
            test_counts_raw[test_counts_raw >= 5].index
        )

        if len(valid_categories) == 0:
            print(f"No valid categories for {col_name} with at least 5 counts in both train and test sets.\n")
            return

        contingency_table = pd.DataFrame(
            {
                "train": train_counts_raw.reindex(valid_categories).fillna(0),
                "test": test_counts_raw.reindex(valid_categories).fillna(0),
            }
        ).fillna(0)

        # Laplace Smoothing
        observed = contingency_table.values + 0.5
        total = observed.sum(axis=0)
        expected = np.outer(observed.sum(axis=1), total) / total.sum()

        g_stat = 2 * (observed * np.log(observed / expected)).sum()

        df = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)

        p_value = 1 - chi2.cdf(g_stat, df)

        print(f"G-test for {col_name}:")
        print(f"  - P-value: {p_value:.4f}")

        return p_value


    def plot_numerical_distribution(self, col_name, bins=10):
        """
        Visualizes the distribution of a numerical feature in the training and test sets.

        Args:
            col_name (str): Name of the numerical column to plot.
            bins (int, optional): Number of bins for the histogram (default: 10).
        """

        plt.figure(figsize=(10, 6))

        plt.hist(self.X_train[col_name], density=True, bins=20, alpha=0.5, label='Training Set')
        plt.hist(self.X_test[col_name], density=True, bins=20, alpha=0.5, label='Test Set', color='orange')
        plt.title(f"Distribution of {col_name}")
        plt.xlabel(col_name)
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def compare_numerical_distributions(self):
        """
        Compares the distributions of numerical features in training and test sets
        using the Kolmogorov-Smirnov test and applies Bonferroni correction for 
        multiple comparisons.
        """
        # Bonferroni Correction
        alpha = 0.05
        num_numerical_features = len(self.numerical_features.columns)
        adjusted_alpha = alpha / num_numerical_features

        for col in self.numerical_features:
            self.plot_numerical_distribution(col)

            # Perform KS-test and compare p-value with adjusted alpha
            p_value = self.perform_ks_test(col)

            if p_value < adjusted_alpha:
                print(
                    f"  - Conclusion: The distribution of {col} is significantly different between the train and test sets (reject null hypothesis after Bonferroni correction).\n"
                )
            else:
                print(
                    f"  - Conclusion: The distribution of {col} is not significantly different between the train and test sets (fail to reject null hypothesis after Bonferroni correction).\n"
                )


    def perform_ks_test(self, col_name):
        """
        Performs the Kolmogorov-Smirnov test for the specified numerical feature
        and prints the results.

        Args:
            col_name (str): The name of the numerical column to test.
        """

        ks_stat, p_value = ks_2samp(self.X_train[col_name], self.X_test[col_name])

        print(f"Kolmogorov-Smirnov test for {col_name}:")
        print(f"  - P-value: {p_value:.4f}") 
        return p_value

def compare_train_test_distributions(X_train, X_test):
    """
    Compares the distributions of features in training and test sets.

    Args:
        X_train (pd.DataFrame): Training dataset.
        X_test (pd.DataFrame): Test dataset.
    """
    comparator = DistributionComparison(X_train, X_test)
    comparator.compare_categorical_distributions()
    comparator.compare_numerical_distributions()


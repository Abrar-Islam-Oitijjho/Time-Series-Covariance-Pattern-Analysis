import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


class Clustering:
    """
    Helpers for clustering variables (columns) of a time-series DataFrame.
    Assumes df has a 'DateTime' column which will be dropped before analysis.
    Each variable (column) is standardized across time before clustering.
    """

    def __init__(self):
        pass

    @staticmethod
    def _prepare_and_scale(df: pd.DataFrame):
        """
        Drop DateTime (if present), convert to numeric matrix, standardize each variable
        across time and return (df_edit, data_scaled) where data_scaled has shape
        (n_variables, n_timepoints).
        """
        if 'DateTime' in df.columns:
            df_edit = df.drop(columns=['DateTime'])
        else:
            df_edit = df.copy()

        # Ensure numeric and no stray non-numeric columns
        df_edit = df_edit.select_dtypes(include=[np.number]).copy()

        if df_edit.empty:
            raise ValueError("No numeric columns found for clustering after dropping 'DateTime'.")

        # Standardize each variable (column) across time (rows)
        scaler = StandardScaler()
        # scaler expects shape (n_samples, n_features) => here samples=timepoints, features=variables
        scaled_cols = scaler.fit_transform(df_edit.values)  # shape: (n_timepoints, n_variables)
        data_scaled = scaled_cols.T  # now shape: (n_variables, n_timepoints)

        return df_edit, data_scaled

    def run_kmeans(self, df: pd.DataFrame, n_clusters: int):
        """
        Cluster variables using KMeans. Returns DataFrame mapping Variable -> Cluster.
        """
        df_edit, data_scaled = self._prepare_and_scale(df)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data_scaled)  # each row = variable

        df_clusters = pd.DataFrame({
            'Variable': df_edit.columns,
            'Cluster': kmeans.labels_
        })

        return df_clusters

    def kmeans_elbow(self, df: pd.DataFrame, max_k: int = 10):
        """
        Compute WCSS (inertia) for 1..max_k to inspect elbow.
        Returns list of inertias (index 0 -> k=1).
        """
        _, data_scaled = self._prepare_and_scale(df)
        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data_scaled)
            wcss.append(kmeans.inertia_)
        return wcss

    def run_ahc(self, df: pd.DataFrame, n_clusters: int):
        """
        Agglomerative (hierarchical) clustering of variables.
        Returns DataFrame mapping Variable -> Cluster.
        """
        df_edit, data_scaled = self._prepare_and_scale(df)

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',  # use metric param (newer sklearn)
            linkage='ward'
        )
        labels = model.fit_predict(data_scaled)

        df_clusters = pd.DataFrame({
            'Variable': df_edit.columns,
            'Cluster': labels
        })

        return df_clusters

    def plot_dendrogram(self, df: pd.DataFrame, title: str = "Dendrogram"):
        """
        Plot dendrogram for variables using Ward linkage. Variables are used as observations.
        """
        df_edit, data_scaled = self._prepare_and_scale(df)

        # linkage expects shape (n_observations, n_features) -> here each observation is a variable
        linked = linkage(data_scaled, method='ward')

        plt.figure(figsize=(10, 6))
        dendrogram(linked, labels=df_edit.columns, orientation="top")
        plt.title(title)
        plt.tight_layout()
        plt.show()


class PCAAnalysis:
    """
    PCA on variables (each row should be a variable, each column a feature/timepoint).
    Provide 'data_scaled' in shape (n_variables, n_timepoints) and matching column_labels (variable names).
    """

    def __init__(self, data_scaled: np.ndarray, column_labels: list[str]):
        self.data_scaled = data_scaled
        self.column_labels = list(column_labels)
        self.pca = PCA()
        self.pca_components = None  # transformed data (scores)

    def fit_pca(self, n_components: int | None = None):
        """
        Fit PCA and return transformed components (scores).
        If n_components is provided, use it.
        """
        if n_components is not None:
            self.pca = PCA(n_components=n_components)
        self.pca_components = self.pca.fit_transform(self.data_scaled)
        return self.pca_components

    def explained_variance(self):
        var_ratio = self.pca.explained_variance_ratio_
        cum_var = var_ratio.cumsum()
        return var_ratio, cum_var

    def plot_scree(self, save_path: str | None = None):
        var_ratio, cum_var = self.explained_variance()
        components = np.arange(1, len(var_ratio) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(components, var_ratio, marker='o', linestyle='--')
        for i, var in enumerate(var_ratio):
            plt.text(i + 1.05, var, f"{var:.2f}", fontsize=8)
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Scree Plot - Explained Variance Ratio")
        if save_path:
            plt.savefig(save_path + "_explained_variance.png", bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(components, cum_var, marker='o', linestyle='--')
        for i, var in enumerate(cum_var):
            plt.text(i + 1.05, var, f"{var:.2f}", fontsize=8)
        plt.xlabel("Principal Component")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Scree Plot - Cumulative Explained Variance")
        if save_path:
            plt.savefig(save_path + "_cumulative_variance.png", bbox_inches='tight')
        plt.show()

    def plot_biplot(self, scale_arrows: float = 1.0, save_path: str | None = None):
        """
        Simple biplot using the first two principal components.
        scale_arrows: multiplier for loading vectors to make arrows visible.
        """
        if self.pca_components is None:
            raise RuntimeError("PCA not fitted. Call fit_pca() before plot_biplot().")

        # loadings: components_.T has shape (n_features, n_components)
        loading_vectors = self.pca.components_.T  # shape (n_timepoints, n_components)
        # We fit PCA with variables as rows; loadings correspond to original features (timepoints).
        # For a variable-oriented biplot, you might want to plot the scores (self.pca_components).
        scores = self.pca_components  # shape (n_variables, n_components)

        plt.figure(figsize=(8, 6))
        plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
        for i, var in enumerate(self.column_labels):
            plt.text(scores[i, 0], scores[i, 1], var, fontsize=9)

        # Optionally draw loading vectors (from features). They will often be dense for time-series features.
        # Use the first two components of the loadings scaled by scale_arrows.
        if loading_vectors.shape[1] >= 2:
            for i in range(loading_vectors.shape[0]):
                x = loading_vectors[i, 0] * scale_arrows
                y = loading_vectors[i, 1] * scale_arrows
                plt.arrow(0, 0, x, y, head_width=0.02, head_length=0.02, linewidth=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Biplot (variables as points)")
        plt.axhline(0, linestyle="--", linewidth=0.5)
        plt.axvline(0, linestyle="--", linewidth=0.5)
        if save_path:
            plt.savefig(save_path + "_biplot.png", bbox_inches='tight')
        plt.show()


class SubGroupAnalysis:
    """
    Break DataFrame into RAP-based segments and perform clustering/PCA per segment.
    """

    def __init__(self):
        self.clusterer = Clustering()

    def clustering_by_segments(self, df: pd.DataFrame, labels: list[str], breakpoints: tuple[float, float],
                               method: str = "kmeans", method_args: dict | None = None):
        """
        df: DataFrame that must include 'RAP' column and variable columns (with optional DateTime).
        labels: list of 3 strings naming the segments (e.g., ['RAP<0', '0<=RAP<=0.4', 'RAP>0.4'])
        breakpoints: tuple (bp_low, bp_high)
        method: 'kmeans' | 'ahc' | 'pca'
        method_args: kwargs to pass to the selected method (e.g., {'n_clusters': 3} for kmeans/ahc)
        Returns concatenated DataFrame with a 'Segment' column and cluster labels or PCA scores.
        """
        if method_args is None:
            method_args = {}

        if 'RAP' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'RAP' column for segmentation.")

        seg1 = df[df['RAP'] < breakpoints[0]]
        seg2 = df[(df['RAP'] >= breakpoints[0]) & (df['RAP'] <= breakpoints[1])]
        seg3 = df[df['RAP'] > breakpoints[1]]

        segments = [seg1, seg2, seg3]
        results = []

        for i, seg in enumerate(segments):
            if seg.empty:
                continue

            if method == "kmeans":
                df_clusters = self.clusterer.run_kmeans(seg, **method_args)

            elif method == "ahc":
                df_clusters = self.clusterer.run_ahc(seg, **method_args)

            elif method == "pca":
                # For PCA we will return PCA scores for variables in the segment.
                # Prepare scaled data and run PCAAnalysis
                df_edit, data_scaled = self.clusterer._prepare_and_scale(seg)
                pca = PCAAnalysis(data_scaled, list(df_edit.columns))
                pca.fit_pca(n_components=2 if data_scaled.shape[0] >= 2 else None)
                scores = pca.pca_components  # shape (n_variables, n_components)
                # build a DataFrame of scores
                comp_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
                df_clusters = pd.DataFrame(scores, columns=comp_cols)
                df_clusters.insert(0, "Variable", df_edit.columns)

            else:
                raise ValueError(f"Unknown method '{method}'. Choose 'kmeans', 'ahc', or 'pca'.")

            # Add segment label
            if 'Variable' not in df_clusters.columns:
                df_clusters.insert(0, "Variable", df_edit.columns)
            df_clusters.insert(0, "Segment", labels[i])
            results.append(df_clusters)

        if not results:
            return pd.DataFrame()  # empty result if all segments were empty

        final_df = pd.concat(results, ignore_index=True)
        return final_df

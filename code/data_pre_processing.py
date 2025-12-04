import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

class DataLoader:
    """
    Class to load and preprocess minute-by-minute cerebral physiology data.
    """
    def __init__(self, directory):
        self.directory = directory

    def load_and_clean_data(self):
        """
        Loads all CSV files, removes NaNs, filters out extreme ICP/MAP values,
        and concatenates into a single DataFrame.
        """
        df_all = pd.DataFrame()
        files = os.listdir(self.directory)

        for file in files:
            df_individual = pd.read_csv(os.path.join(self.directory, file)).dropna()
            filter_condition = (
                (df_individual['mean_ICP'] > 100) |
                (df_individual['mean_ICP'] < -15) |
                (df_individual['MAP'] > 200) |
                (df_individual['MAP'] < 0)
            )
            df_individual = df_individual[~filter_condition]
            df_all = pd.concat([df_all, df_individual], axis=0, ignore_index=True)

        df_all.columns = ['DateTime', 'RAP', 'ICP', 'AMP', 'MAP', 'CPP']
        return df_all




class Preprocessor:
    """
    Class to preprocess data for PCA or clustering.
    """
    def __init__(self, df, drop_columns=['DateTime']):
        self.df = df.copy()
        self.drop_columns = drop_columns

    def scale_data(self):
        """
        Scales numerical features and returns a numpy array.
        """
        df_scaled = self.df.drop(columns=self.drop_columns, errors='ignore')
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_scaled)
        return data_scaled, df_scaled.columns

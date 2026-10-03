import pandas as pd
import numpy as np
import os

class InputsTransformer:
    def __init__(self, df, fit_df_path):
        if not os.path.exists(fit_df_path): return
        self.df = df
        self.transformed_df = pd.DataFrame()
        self.fit_df = pd.read_csv(fit_df_path)
        self.normalize_cols()
        self.reduce_numeric_size()

    def get_result(self):
        return self.transformed_df

    def get_min_max_sub(self, column):
        min = self.fit_df[column][0]
        max_min = self.fit_df[column][2]
        return min, max_min

    def inverse_transformation(self, column, input):
        min = self.fit_df[column][0]
        max_min = self.fit_df[column][2]
        return np.add(np.multiply(input, max_min), min)

    def normalize_cols(self):
        for column in self.df.columns:
            new_column = f'{column}_Transformed'
            if self.df[column].dtype == 'int64':
                min, max_min = self.get_min_max_sub(column)
                self.transformed_df[new_column] = np.int16(np.divide(np.subtract(self.df[column].values, min), max_min))
            else:
                mask = (self.df[column] == 'Low') | (self.df[column] == 'Medium') | (self.df[column] == 'High')
                if mask.sum() > 0:
                    # low/medium/high
                    ids = self.df[self.df[column] == 'Low'].index
                    self.transformed_df.loc[ids, new_column] = np.float16(0)
                    ids = self.df[self.df[column] == 'Medium'].index
                    self.transformed_df.loc[ids, new_column] = np.float16(0.5)
                    ids = self.df[self.df[column] == 'High'].index
                    self.transformed_df.loc[ids, new_column] = np.float16(1)
                    continue
                mask = (self.df[column] == 'No') | (self.df[column] == 'Yes')
                if mask.sum() > 0:
                    # yes/no
                    self.transformed_df[new_column] = np.int16(np.where(self.df[column] == 'Yes', 1, 0))
                    continue
                mask = (self.df[column] == 'Public') | (self.df[column] == 'Private')
                if mask.sum() > 0:
                    # public/private
                    self.transformed_df[new_column] = np.int16(np.where(self.df[column] == 'Public', 1, 0))
                    continue
                mask = (self.df[column] == 'Positive') | (self.df[column] == 'Negative') | (self.df[column] == 'Neutral')
                if mask.sum() > 0:
                    # postivie/negative/neutral
                    ids = self.df[self.df[column] == 'Positive'].index
                    self.transformed_df.loc[ids, new_column] = np.int16(1)
                    ids = self.df[self.df[column] == 'Neutral'].index
                    self.transformed_df.loc[ids, new_column] = np.int16(0)
                    ids = self.df[self.df[column] == 'Negative'].index
                    self.transformed_df.loc[ids, new_column] = np.int16(-1)
                    continue
                mask = (self.df[column] == 'High School') | (self.df[column] == 'College') | (self.df[column] == 'Postgraduate')
                if mask.sum() > 0:
                    # school
                    ids = self.df[self.df[column] == 'High School'].index
                    self.transformed_df.loc[ids, new_column] = np.float16(0)
                    ids = self.df[self.df[column] == 'College'].index
                    self.transformed_df.loc[ids, new_column] = np.float16(0.5)
                    ids = self.df[self.df[column] == 'Postgraduate'].index
                    self.transformed_df.loc[ids, new_column] = np.float16(1)
                    continue
                mask = (self.df[column] == 'Near') | (self.df[column] == 'Moderate') | (self.df[column] == 'Far')
                if mask.sum() > 0:
                    # school
                    ids = self.df[self.df[column] == 'Near'].index
                    self.transformed_df.loc[ids, new_column] = np.float16(0)
                    ids = self.df[self.df[column] == 'Moderate'].index
                    self.transformed_df.loc[ids, new_column] = np.float16(0.5)
                    ids = self.df[self.df[column] == 'Far'].index
                    self.transformed_df.loc[ids, new_column] = np.float16(1)
                    continue
                # gender
                self.transformed_df[new_column] = np.int16(np.where(self.df[column] == 'Male', 1, 0))
                continue

    def reduce_numeric_size(self):
        for column in self.transformed_df.columns:
            if 'float' in str(self.transformed_df[column].dtype):
                self.transformed_df[column] = np.float16(self.transformed_df[column].values)
            else:
                self.transformed_df[column] = np.int16(self.transformed_df[column].values)
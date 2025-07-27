import numpy as np
import pandas as pd
import warnings

class DataPreprocess():
    """
    Data preprocessing for the intrusion detection systems.

    Author:
        Noah Ogilvie

    Version:
        1.0
    """

    def __init__(self, fileName: str) -> None:
        """
        Initializes DataPreprocess

        Args:
            fileName (str): The file name
        """
        if not fileName.endswith('.csv'):
            SystemError("Input an appropriate CSV file")

        self.__df = pd.read_csv(fileName)

    def run(self, *, givenTargets: dict[str, int] | None = None, targetName: str | None = None) -> pd.DataFrame:
        """
        Runs the entire program to clean the data.

        Args:
            givenTargets (dict[str, int] | None): Mapping to convert categorical to numeric. Defaults to None.
            targetName (str | None): Name of the target column in the dataframe. Defaults to None.

        Returns:
            pd.DataFrame: The cleaned and filtered dataframe.
        """
        # Drops rows with any missing values
        if self.__df.isnull().values.any():
            self.__df = self.__df.dropna(axis=1)

        # Drops rows with duplicate values
        if self.__df.duplicated().values.any():
            self.__df.drop_duplicate(keep='first', inplace=True)
            self.__df.reset_index(drop=True, inplace=True)

        # replace() method has been deprecated, we ignore the warning
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Change the targets from categorical to numeric
        if givenTargets is not None and targetName is not None:
            self.__df[targetName] = self.__df[targetName].replace(givenTargets)

        df_removed = self.__remove_high_corr()

        quantitative_data = df_removed.select_dtypes(include='number').copy()
        target = quantitative_data['label']
        features = quantitative_data.drop(columns=['label'])

        filtered_features, filtered_target = self.__remove_outliers_per_class(features, target, k=1.5)

        df_filtered = filtered_features.copy()
        df_filtered['label'] = filtered_target
        return df_filtered

    def __remove_high_corr(self) -> pd.DataFrame:
        """
        Removes highly correlated features from the dataframe

        Returns:
            pd.Dataframe: DataFrame with highly correlated columns removed
        """
        corr_df = self.__df.corr().abs()
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        tri_df = corr_df.mask(mask)
        to_drop = [c for c in tri_df.columns if any(tri_df[c] >= 0.9)]
        return self.__df.drop(columns=to_drop, axis=1)

    def __remove_outliers_per_class(self, features: pd.DataFrame, target: pd.Series, k=1.5) -> tuple[pd.DataFrame, pd.Series]:
        """
        Removes outliers from the featurews on a per-class basis using the IQR method.

        Args:
            features (pd.DataFrame): The feature columns.
            target (pd.Series): The target labels corresponding to features
            k (float): The multiplier for the IQR to determine outliers. Default is 1.5

        Returns:
            tuple[pd.DataFrame, pd.Series]: Filtered features and target with outliers removed.
        """
        indices_to_keep = []
        for label in target.unique():
            class_data = features[target == label]
            outlier_indices = set()

            for col in class_data.columns:
                Q1 = class_data[col].quantile(0.25)
                Q3 = class_data[col].quantile(0.75)
                IQR = Q3 - Q1
                LB = Q1 - k * IQR
                UB = Q3 + k * IQR

                outliers_col = class_data[(class_data[col] < LB) | (class_data[col] > UB)].index
                outlier_indices.update(outliers_col)

            # Keep samples that are NOT outliers for this class
            keep_indices = set(class_data.index) - outlier_indices
            indices_to_keep.extend(keep_indices)

        return features.loc[indices_to_keep], target.loc[indices_to_keep]
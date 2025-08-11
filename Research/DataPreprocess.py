import numpy as np
import pandas as pd
import warnings

class DataPreprocess():
    """
    Data preprocessing for the intrusion detection systems.

    Author:
        Noah Ogilvie

    Version:
        1.5.0
    """

    def __init__(self, fileName: str | None = None) -> None:
        """
        Initializes DataPreprocess

        Args:
            fileName (str): The file name
        """
        if fileName and not fileName.endswith('.csv'):
            SystemError("Input an appropriate CSV file")

        elif fileName:
            self.__df = pd.read_csv(fileName, low_memory=False)
        else:
            self.__df = pd.DataFrame()

    def runNew(
        self,
        *,
        targetName: str | None = None,
        featuresToBinary: bool = False,
        targetToBinary: bool = False,
        givenDF: pd.DataFrame | None = None,
        normalBehavior: str | None = 'benign'
    ) -> pd.DataFrame:
        """
        Runs the entire program to clean the data.

        Args:
            targetName (str | None): Name of the target column in the dataframe. Defaults to None.
            featuresToBinary (bool): Boolean value to change categorical features into numeric. Defaults to False.
            targetToBinary (bool): Boolean value to change categorical target into numeric. Defaults to False.

        Returns:
            pd.DataFrame: The cleaned and filtered dataframe.
        """
        if givenDF is not None:
            self.__df = givenDF.copy()

        if self.__df.empty:
            raise ValueError('[ERROR] DataFrame is empty. Please check the input file.')
        
        if targetName is None:
            raise ValueError('[ERROR] targetName must be specified')
        
        if targetName not in self.__df.columns:
            raise KeyError('[ERROR] targetName not found in DataFrame columns')
        
        dropThreshold = 0.95

        print(f"[INFO] Initial dataset shape: {self.__df.shape}")
        
        self.__df = self.__df.dropna(axis=0)
        self.__df.drop_duplicates(keep='first', inplace=True)
        self.__df.reset_index(drop=True, inplace=True)

        print(f"[INFO] Initial dataset shape after removing duplicates and NaNs: {self.__df.shape}")

        target = self.__df[targetName]
        features = self.__df.drop(columns=[targetName])

        for current in features.columns:
            if features[current].dtype == 'object':
                converted = pd.to_numeric(features[current], errors='coerce')

                # Conversion to numeric, then replace column
                if converted.notna().sum() / len(features) > dropThreshold:
                    features[current] = converted
                
                else:
                    # Keep object for encoding later
                    features[current] = features[current].astype('category').cat.codes

        # We will drop columns that are mostly NaN
        features = features.loc[:, features.notna().mean() > dropThreshold]

        if featuresToBinary:
            encoded_parts = []
            oneHotThreshold = 10
            
            for current in features.select_dtypes(include='object').columns:
                n_unique = features[current].nunique()

                if n_unique == 2:
                    mapping = {v: i for i, v in enumerate(features[current].unique())}
                    features[current] = features[current].map(mapping)

                elif n_unique <= oneHotThreshold:
                    one_hot = pd.get_dummies(features[current], prefix=current)
                    encoded_parts.append(one_hot)
                    features = features.drop(columns=[current])

                else:
                    # Drop high-cardinality strings (e.g., IPs, payloads)
                    features = features.drop(columns=[current])

            if encoded_parts:
                features = pd.concat([features] + encoded_parts, axis=1)

        if targetToBinary:
            target = target.apply(lambda x: 0 if str(x).lower() == normalBehavior.lower() else 1)
        else:
            print(f"[INFO] Target column is multiclass with unique values: {target.unique()} OR no need for mapping")

        processed_df = pd.concat([features, target], axis=1)
        processed_df = self.__remove_target_leakage(df=processed_df, targetName=targetName)

        print(f"[INFO] Processed DataFrame shape: {processed_df.shape}")

        filtered_features = self.__remove_high_corr(targetName=targetName, df=processed_df)
        df_removed = pd.concat([filtered_features, processed_df[targetName]], axis=1)

        print(f"[INFO] DataFrame shape after removing high correlation features: {df_removed.shape}")

        #quantitative_data = df_removed.select_dtypes(include='number').copy()
        quantitative_data = df_removed.select_dtypes(include='number')

        target = quantitative_data[targetName]
        features = quantitative_data.drop(columns=[targetName])

        filtered_features, filtered_target = self.__remove_outliers_per_class(features=features, target=target)

        df_filtered = pd.concat([filtered_features, filtered_target], axis=1)

        #df_filtered = filtered_features.copy()
        #df_filtered[targetName] = filtered_target

        print(f"[INFO] DataFrame shape after removing outliers: {df_filtered.shape}")

        df_filtered = self.__remove_target_leakage(df=df_filtered, targetName=targetName)

        if df_filtered.isnull().any().any():
            print("[WARNING] NaNs still present after filling! Dropping affected rows.")
            df_filtered.dropna(axis=0, inplace=True)
            df_filtered.drop_duplicates(keep='first', inplace=True)
            df_filtered.reset_index(drop=True, inplace=True)

        df_filtered = self.__remove_target_leakage(df=df_filtered, targetName=targetName)
        
        print(f"[INFO] Final dataset shape: {df_filtered.shape}")
        print(f"[INFO] Columns: {list(df_filtered.columns)}")
        print(f"[INFO] Any NaNs left? {df_filtered.isnull().any().any()}")
        print('[END] Data preprocessing completed successfully.')

        return df_filtered

    def __remove_high_corr(self, targetName: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes highly correlated features from the dataframe

        Args:
            targetName (str): The name of the target

        Returns:
            pd.Dataframe: DataFrame with highly correlated columns removed
        """

        if not targetName:
            raise ValueError('[ERROR] targetName was not passed into the parameter for __remove_high_corr')

        corr_df = df.drop(columns=[targetName]).corr().abs()
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        tri_df = corr_df.mask(mask)

        to_drop = [col for col in tri_df.columns if tri_df[col].max(skipna=True) >= 0.9]
        filtered_features = df.drop(columns=to_drop)

        return filtered_features
    
    def __remove_target_leakage(self, df: pd.DataFrame, targetName: str, threshold: float = 0.9) -> pd.DataFrame:
        """
        Removes features that are highly correlated with the target to prevent data leakage.

        Args:
            df (pd.DataFrame): The input dataframe.
            targetName (str): The name of the target column.
            threshold (float): The absolute correlation threshold.

        Returns:
            pd.DataFrame: DataFrame with leakage-prone features removed.
        """
        corr_with_target = df.corr()[targetName].drop(targetName).abs()
        leaking_features = corr_with_target[corr_with_target > threshold].index.tolist()

        if leaking_features:
            print(f"[LEAKAGE WARNING] Dropping features correlated with target > {threshold}: {leaking_features}")
            df = df.drop(columns=leaking_features)
        else:
            print("[LEAKAGE CHECK] No data leakage detected.")

        return df
        
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

        if isinstance(target, pd.DataFrame):
            target = target.iloc[:, 0]

        # 🔹 Step 1: Global IQR filtering
        Q1 = features.quantile(0.25)
        Q3 = features.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        global_mask = ~((features < lower) | (features > upper)).any(axis=1)
        features = features.loc[global_mask]
        target = target.loc[global_mask]

        # 🔹 Step 2: Per-class IQR filtering
        indices_to_keep = []
        for label in target.unique():
            class_data = features[target == label]
            class_target = target[target == label]
            outlier_indices = set()
            for col in class_data.columns:
                Q1 = class_data[col].quantile(0.25)
                Q3 = class_data[col].quantile(0.75)
                IQR = Q3 - Q1
                LB = Q1 - k * IQR
                UB = Q3 + k * IQR
                outlier_indices.update(class_data[(class_data[col] < LB) | (class_data[col] > UB)].index)
            keep_indices = set(class_data.index) - outlier_indices
            indices_to_keep.extend(keep_indices)

        indices_to_keep = sorted(indices_to_keep)
        filtered_features = features.loc[indices_to_keep].reset_index(drop=True)
        filtered_target = target.loc[indices_to_keep].reset_index(drop=True)

        return filtered_features, filtered_target
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.__df.copy()
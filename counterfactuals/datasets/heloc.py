import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from counterfactuals.datasets.base import AbstractDataset


class HelocDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/heloc.csv", transform=True, shuffle=True):
        """
        Initialize the HELOC dataset.
        """
        self.categorical_features = []
        self.features = [
            "ExternalRiskEstimate",
            "MSinceOldestTradeOpen",
            "MSinceMostRecentTradeOpen",
            "AverageMInFile",
            "NumSatisfactoryTrades",
            "NumTrades60Ever2DerogPubRec",
            "NumTrades90Ever2DerogPubRec",
            "PercentTradesNeverDelq",
            "MSinceMostRecentDelq",
            "MaxDelq2PublicRecLast12M",
            "MaxDelqEver",
            "NumTotalTrades",
            "NumTradesOpeninLast12M",
            "PercentInstallTrades",
            "MSinceMostRecentInqexcl7days",
            "NumInqLast6M",
            "NumInqLast6Mexcl7days",
            "NetFractionRevolvingBurden",
            "NetFractionInstallBurden",
            "NumRevolvingTradesWBalance",
            "NumInstallTradesWBalance",
            "NumBank2NatlTradesWHighUtilization",
            "PercentTradesWBalance",
            "RiskPerformance",
        ]
        self.raw_data = self.load(file_path=file_path, index_col=False)
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y, shuffle=shuffle
        )
        if transform:
            self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        # Remove rows where all NaN
        raw_data = raw_data[(raw_data.iloc[:, 1:] >= 0).any(axis=1)].copy()

        target_column = "RiskPerformance"
        raw_data[target_column] = (
            raw_data[target_column].replace({"Bad": "0", "Good": "1"}).astype(int)
        )

        raw_data[raw_data < 0] = np.nan
        raw_data = raw_data.apply(lambda col: col.fillna(col.median()), axis=0)

        self.feature_columns = raw_data.columns.drop(target_column)

        self.numerical_columns = list(range(0, len(self.feature_columns)))
        self.actionable_features = list(range(0, 6))
        self.not_actionable_features = list(range(6, len(self.feature_columns)))
        self.categorical_columns = []

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()

        return X, y

    def transform(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        Transform the loaded data by applying Min-Max scaling to the features.
        """
        self.feature_transformer = MinMaxScaler()
        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        # self.target_transformer = LabelEncoder()
        # y_train = self.target_transformer.fit_transform(y_train)
        # y_test = self.target_transformer.transform(y_test)

        X_train = np.array(X_train.astype(np.float32))
        X_test = np.array(X_test.astype(np.float32))
        y_train = np.array(y_train.astype(np.int64))
        y_test = np.array(y_test.astype(np.int64))

        self.categorical_features = []
        self.numerical_features = list(range(0, len(self.feature_columns)))

        return X_train, X_test, y_train, y_test

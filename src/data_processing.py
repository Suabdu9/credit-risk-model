import os
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


# =====================================================
# Aggregate Customer Features
# =====================================================

class AggregateFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create customer-level aggregate transaction features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        customer_agg = (
            df.groupby("CustomerId")["Amount"]
            .agg(
                total_transaction_amount="sum",
                avg_transaction_amount="mean",
                transaction_count="count",
                std_transaction_amount="std",
                max_transaction_amount="max",
                min_transaction_amount="min"
            )
            .reset_index()
        )

        customer_agg["std_transaction_amount"] = (
            customer_agg["std_transaction_amount"]
            .fillna(0)
        )

        df = df.merge(
            customer_agg,
            on="CustomerId",
            how="left"
        )

        return df


# =====================================================
# Datetime Feature Extraction
# =====================================================

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract date and time features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"]
        )

        df["transaction_hour"] = (
            df["TransactionStartTime"].dt.hour
        )

        df["transaction_day"] = (
            df["TransactionStartTime"].dt.day
        )

        df["transaction_month"] = (
            df["TransactionStartTime"].dt.month
        )

        df["transaction_year"] = (
            df["TransactionStartTime"].dt.year
        )

        df["day_of_week"] = (
            df["TransactionStartTime"].dt.dayofweek
        )

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        return df


# =====================================================
# Main Processing Function
# =====================================================

def create_rfm_target(df):
    """
    Create proxy credit risk target using RFM analysis and KMeans clustering.
    """

    rfm_df = df.copy()

    # Ensure datetime
    rfm_df["TransactionStartTime"] = pd.to_datetime(
        rfm_df["TransactionStartTime"]
    )

    # Snapshot date
    snapshot_date = (
        rfm_df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    # Calculate RFM
    rfm = (
        rfm_df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    # Scale RFM
    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    # Cluster customers
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    # Determine high-risk cluster
    cluster_summary = (
        rfm.groupby("cluster")
        .agg(
            {
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean"
            }
        )
    )

    print("\nCluster Summary")
    print(cluster_summary)
    cluster_summary.to_csv(
        "data/processed/rfm_cluster_summary.csv"
    )

    high_risk_cluster = (
        cluster_summary
        .sort_values(
            by=["Frequency", "Monetary", "Recency"],
            ascending=[True, True, False],
        )
        .index[0]
    )

    rfm["is_high_risk"] = (
        rfm["cluster"] == high_risk_cluster
    ).astype(int)

    rfm.to_csv(
        "data/processed/rfm_customer_segments.csv",
        index=False
    )

    return rfm[
        [
            "CustomerId",
            "is_high_risk"
        ]
    ]


def preprocess_data(df):
    """
    Apply feature engineering and preprocessing.
    """

    # Feature Engineering
    df = AggregateFeatureCreator().transform(df)
    df = DatetimeFeatureExtractor().transform(df)

    risk_labels = create_rfm_target(df)

    df = df.merge(
         risk_labels,
         on="CustomerId",
         how="left"
    )

    # Drop unnecessary columns
    drop_columns = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "TransactionStartTime",
        "CountryCode"
    ]

    df = df.drop(columns=drop_columns)

    # Feature groups
    categorical_features = [
        "CurrencyCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId"
    ]

    target_column = "is_high_risk"

    numerical_features = [
        col
        for col in df.columns
        if col not in categorical_features
        and col != target_column
    ]

    # Pipelines
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    processed_array = preprocessor.fit_transform(df)

    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(
        processed_array,
        columns=feature_names
    )

    processed_df[target_column] = (
        df[target_column].values
    )

    return processed_df, preprocessor


# =====================================================
# Script Entry Point
# =====================================================

if __name__ == "__main__":

    RAW_DATA_PATH = "data/raw/data.csv"
    OUTPUT_DATA_PATH = "data/processed/model_data.csv"
    PIPELINE_PATH = "data/processed/preprocessor.pkl"

    os.makedirs("data/processed", exist_ok=True)

    try:
        print("Loading data...")

        df = pd.read_csv(RAW_DATA_PATH)

        print(f"Dataset shape: {df.shape}")

        processed_df, pipeline = preprocess_data(df)

        processed_df.to_csv(
            OUTPUT_DATA_PATH,
            index=False
        )

        joblib.dump(
            pipeline,
            PIPELINE_PATH
        )

        print("\nProcessing completed successfully.")
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Saved dataset to: {OUTPUT_DATA_PATH}")
        print(f"Saved pipeline to: {PIPELINE_PATH}")

    except FileNotFoundError:
        print(
            f"File not found: {RAW_DATA_PATH}"
        )

    except Exception as e:
        print(
            f"Processing failed: {e}"
        )

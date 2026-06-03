import pandas as pd

from src.data_processing import (
    AggregateFeatureCreator,
    DatetimeFeatureExtractor
)


def test_aggregate_columns_created():

    df = pd.DataFrame({
        "CustomerId": ["A", "A", "B"],
        "Amount": [100, 200, 50]
    })

    result = AggregateFeatureCreator().transform(df)

    expected = [
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount"
    ]

    for col in expected:
        assert col in result.columns


def test_datetime_features_created():

    df = pd.DataFrame({
        "TransactionStartTime": [
            "2020-01-01 10:00:00"
        ]
    })

    result = DatetimeFeatureExtractor().transform(df)

    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns
    assert "transaction_month" in result.columns


def test_aggregate_feature_creation():

    sample = pd.DataFrame({
        "CustomerId": ["A", "A", "B"],
        "Amount": [100, 200, 50]
    })

    transformer = AggregateFeatureCreator()

    result = transformer.transform(sample)

    expected_cols = [
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount"
    ]

    for col in expected_cols:
        assert col in result.columns

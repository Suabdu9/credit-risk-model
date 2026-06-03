import pandas as pd
from src.data_processing import AggregateFeatureCreator


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
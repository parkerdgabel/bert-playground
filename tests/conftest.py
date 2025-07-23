"""Pytest configuration and shared fixtures for all tests."""

# Add project root to path
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.entities.dataset import DatasetSpecification
from domain.entities.model import TaskType


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_titanic_data():
    """Create sample Titanic-like dataset."""
    data = {
        "PassengerId": [1, 2, 3, 4, 5],
        "Pclass": [3, 1, 3, 1, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath",
            "Allen, Mr. William Henry",
        ],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
        "SibSp": [1, 1, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
        "Fare": [7.2500, 71.2833, 7.9250, 53.1000, 8.0500],
        "Cabin": [None, "C85", None, "C123", None],
        "Embarked": ["S", "C", "S", "S", "S"],
        "Survived": [0, 1, 1, 1, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_house_prices_data():
    """Create sample house prices dataset."""
    data = {
        "Id": [1, 2, 3, 4, 5],
        "MSSubClass": [60, 20, 60, 70, 60],
        "MSZoning": ["RL", "RL", "RL", "RL", "RL"],
        "LotArea": [8450, 9600, 11250, 9550, 14260],
        "Street": ["Pave", "Pave", "Pave", "Pave", "Pave"],
        "LotShape": ["Reg", "Reg", "IR1", "IR1", "IR1"],
        "Neighborhood": ["Collins", "Veenker", "Collins", "Crawfor", "NoRidge"],
        "OverallQual": [7, 6, 7, 7, 8],
        "OverallCond": [5, 8, 5, 5, 5],
        "YearBuilt": [2003, 1976, 2001, 1915, 2000],
        "TotalBsmtSF": [856, 1262, 920, 756, 1145],
        "GrLivArea": [1710, 1262, 1786, 1717, 2198],
        "BedroomAbvGr": [3, 3, 3, 3, 4],
        "KitchenQual": ["Gd", "TA", "Gd", "Gd", "Gd"],
        "SalePrice": [208500, 181500, 223500, 140000, 250000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_multiclass_data():
    """Create sample multiclass classification dataset."""
    data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature2": ["A", "B", "C", "A", "B", "C"],
        "feature3": [10, 20, 30, 40, 50, 60],
        "text_feature": [
            "This is sample text one",
            "Another piece of text",
            "Third text sample here",
            "Fourth text example",
            "Fifth sample text",
            "Sixth text entry",
        ],
        "target": [0, 1, 2, 0, 1, 2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset."""
    data = {
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "category": ["A", "B", "A", "B", "A"],
        "text": [
            "Short text",
            "Medium length text here",
            "Brief",
            "Longer text with more words",
            "Text",
        ],
        "y": [1.5, 3.2, 4.8, 6.1, 7.9],
    }
    return pd.DataFrame(data)


@pytest.fixture
def titanic_dataset_spec():
    """Create DatasetSpecification for Titanic competition."""
    return DatasetSpecification(
        id="titanic",
        name="Titanic Competition",
        path=Path("/tmp/titanic"),
        task_type=TaskType.CLASSIFICATION,
        num_samples=5,
        num_features=11,
        target_column="Survived",
        text_columns=["Name", "Ticket"],
        categorical_columns=["Pclass", "Sex", "Embarked"],
        numerical_columns=["Age", "SibSp", "Parch", "Fare"],
        num_labels=2,
        metadata={
            "class_distribution": {"0": 2, "1": 3},
            "is_balanced": True
        }
    )


@pytest.fixture
def house_prices_dataset_spec():
    """Create DatasetSpecification for house prices competition."""
    return DatasetSpecification(
        id="house_prices",
        name="House Prices Competition",
        path=Path("/tmp/house_prices"),
        task_type=TaskType.REGRESSION,
        num_samples=5,
        num_features=13,
        target_column="SalePrice",
        text_columns=[],
        categorical_columns=[
            "MSZoning",
            "Street",
            "LotShape",
            "Neighborhood",
            "KitchenQual",
        ],
        numerical_columns=[
            "LotArea",
            "OverallQual",
            "OverallCond",
            "YearBuilt",
            "TotalBsmtSF",
            "GrLivArea",
            "BedroomAbvGr",
        ],
        num_labels=1,
    )


@pytest.fixture
def sample_competition_metadata():
    """Create sample competition metadata."""
    # Return a simple dict for now as CompetitionMetadata is removed
    return {
        "competition_name": "test_competition",
        "competition_type": TaskType.CLASSIFICATION,
        "train_file": "train.csv",
        "test_file": "test.csv",
        "submission_file": "sample_submission.csv",
        "recommended_batch_size": 32,
        "recommended_max_length": 256,
    }


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""

    class MockTokenizer:
        def __init__(self):
            self.vocab = {"[CLS]": 0, "[SEP]": 1, "[PAD]": 2, "test": 3, "word": 4}

        def __call__(
            self,
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors=None,
        ):
            if isinstance(texts, str):
                texts = [texts]

            encodings = {"input_ids": [], "attention_mask": []}

            for text in texts:
                # Simple tokenization for testing
                tokens = text.lower().split()[
                    : max_length - 2
                ]  # Reserve space for special tokens
                input_ids = (
                    [0] + [self.vocab.get(token, 3) for token in tokens] + [1]
                )  # CLS + tokens + SEP

                if padding and len(input_ids) < max_length:
                    attention_mask = [1] * len(input_ids) + [0] * (
                        max_length - len(input_ids)
                    )
                    input_ids = input_ids + [2] * (
                        max_length - len(input_ids)
                    )  # Pad with PAD token
                else:
                    attention_mask = [1] * len(input_ids)

                encodings["input_ids"].append(input_ids)
                encodings["attention_mask"].append(attention_mask)

            if return_tensors == "np":
                import numpy as np

                encodings["input_ids"] = np.array(encodings["input_ids"])
                encodings["attention_mask"] = np.array(encodings["attention_mask"])

            return encodings

    return MockTokenizer()


@pytest.fixture
def temp_csv_files(test_data_dir, sample_titanic_data, sample_house_prices_data):
    """Create temporary CSV files for testing."""
    files = {}

    # Titanic files
    titanic_dir = test_data_dir / "titanic"
    titanic_dir.mkdir()

    train_file = titanic_dir / "train.csv"
    sample_titanic_data.to_csv(train_file, index=False)
    files["titanic_train"] = train_file

    test_file = titanic_dir / "test.csv"
    sample_titanic_data.drop("Survived", axis=1).to_csv(test_file, index=False)
    files["titanic_test"] = test_file

    # House prices files
    house_dir = test_data_dir / "house_prices"
    house_dir.mkdir()

    house_train = house_dir / "train.csv"
    sample_house_prices_data.to_csv(house_train, index=False)
    files["house_train"] = house_train

    house_test = house_dir / "test.csv"
    sample_house_prices_data.drop("SalePrice", axis=1).to_csv(house_test, index=False)
    files["house_test"] = house_test

    return files


@pytest.fixture
def mlx_arrays():
    """Create sample MLX arrays for testing."""
    return {
        "input_ids": mx.array([[1, 2, 3, 0], [1, 4, 5, 0]], dtype=mx.int32),
        "attention_mask": mx.array([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=mx.int32),
        "labels": mx.array([0, 1], dtype=mx.float32),
    }


# Test configuration
def pytest_configure(config):
    """Configure pytest."""
    # Set MLX to CPU for testing
    mx.set_default_device(mx.cpu)

    # Suppress warnings for cleaner test output
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit test marker
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration test marker
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Add benchmark marker
        elif "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)


# Utility functions for tests
def assert_mlx_arrays_equal(a: mx.array, b: mx.array, rtol=1e-5, atol=1e-8):
    """Assert two MLX arrays are equal."""
    assert a.shape == b.shape, f"Shapes don't match: {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"Dtypes don't match: {a.dtype} vs {b.dtype}"

    # Convert to numpy for comparison
    import numpy as np

    a_np = np.array(a)
    b_np = np.array(b)

    np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol)


def create_mock_dataset_files(
    base_dir: Path, competition_name: str, data: pd.DataFrame, target_col: str = None
):
    """Create mock dataset files for testing."""
    comp_dir = base_dir / competition_name
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Train file
    train_file = comp_dir / "train.csv"
    data.to_csv(train_file, index=False)

    # Test file (without target)
    if target_col and target_col in data.columns:
        test_data = data.drop(target_col, axis=1)
    else:
        test_data = data

    test_file = comp_dir / "test.csv"
    test_data.to_csv(test_file, index=False)

    # Sample submission
    if target_col:
        submission_data = pd.DataFrame(
            {"Id": range(len(data)), target_col: [0] * len(data)}
        )
        submission_file = comp_dir / "sample_submission.csv"
        submission_data.to_csv(submission_file, index=False)

    return {
        "train": train_file,
        "test": test_file,
        "submission": comp_dir / "sample_submission.csv" if target_col else None,
        "directory": comp_dir,
    }

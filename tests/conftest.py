"""Contains all shared pytest fixtures."""

from pathlib import Path

import pandas as pd
import pytest

from config import PublicConfig


# NOTE: dotenv package FAILS in CI pipeline without mocking
# required to run pytest from package root directory, relative paths then refer to root
# directory rather than directory of test file
@pytest.fixture
def root_path() -> Path:
    return Path.cwd()


@pytest.fixture
def config() -> PublicConfig:
    return PublicConfig()


@pytest.fixture
def data(root_path, config) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_pickle(
        root_path / "data" / "data_clustering_testing.pkl"
    )
    return data.drop(columns=config.TARGET_COL)

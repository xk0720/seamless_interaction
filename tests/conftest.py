# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS


@pytest.fixture
def sample_filelist() -> pd.DataFrame:
    """Create a sample filelist DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "file_id": "V00_S0809_I00000582_P0947",
                "label": "improvised",
                "split": "dev",
                "batch_idx": 0,
                "archive_idx": 5,
            },
            {
                "file_id": "V00_S0809_I00000582_P0948",
                "label": "improvised",
                "split": "dev",
                "batch_idx": 0,
                "archive_idx": 5,
            },
            {
                "file_id": "V00_S0809_I00000583_P0949",
                "label": "improvised",
                "split": "dev",
                "batch_idx": 0,
                "archive_idx": 5,
            },
            {
                "file_id": "V00_S0809_I00000583_P0950",
                "label": "improvised",
                "split": "dev",
                "batch_idx": 0,
                "archive_idx": 5,
            },
            {
                "file_id": "V01_S0223_I00000127_P1505",
                "label": "improvised",
                "split": "dev",
                "batch_idx": 0,
                "archive_idx": 3,
            },
            {
                "file_id": "V01_S0223_I00000127_P1506",
                "label": "improvised",
                "split": "dev",
                "batch_idx": 0,
                "archive_idx": 3,
            },
            {
                "file_id": "V03_S0456_I00000789_P2001",
                "label": "naturalistic",
                "split": "train",
                "batch_idx": 1,
                "archive_idx": 10,
            },
            {
                "file_id": "V03_S0456_I00000789_P2002",
                "label": "naturalistic",
                "split": "train",
                "batch_idx": 1,
                "archive_idx": 10,
            },
            {
                "file_id": "V00_S1111_I00000999_P3001",
                "label": "improvised",
                "split": "train",
                "batch_idx": 2,
                "archive_idx": 15,
            },
            {
                "file_id": "V01_S2222_I00001000_P4001",
                "label": "naturalistic",
                "split": "test",
                "batch_idx": 3,
                "archive_idx": 20,
            },
        ]
    )


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def test_config(temp_directory):
    """Create a test configuration with temporary directory."""
    return DatasetConfig(
        local_dir=temp_directory,
        num_workers=2,
        label="improvised",
        split="dev",
        preferred_vendors_only=True,
        seed=42,
    )


@pytest.fixture
def mock_fs(sample_filelist, test_config):
    """Create a mocked SeamlessInteractionFS instance for testing."""
    with patch.object(SeamlessInteractionFS, "_load_filelist_cache"):
        fs = SeamlessInteractionFS(config=test_config)
        fs._cached_filelist = sample_filelist
        return fs


@pytest.fixture
def mock_hf_api() -> Mock:
    """Create a mock HuggingFace API for testing."""
    mock_api = Mock()
    mock_info = Mock()
    mock_info.size = 1024 * 1024 * 1024  # 1GB
    mock_api.get_paths_info.return_value = [mock_info]
    return mock_api


@pytest.fixture
def sample_numpy_data() -> Dict[str, Any]:
    """Create sample numpy data for testing."""
    import numpy as np

    return {
        "smplh:body_pose": np.random.random((100, 63)).astype(np.float32),
        "smplh:global_orient": np.random.random((100, 3)).astype(np.float32),
        "movement:emotion_arousal": np.random.random((100, 1)).astype(np.float32),
        "boxes_and_keypoints:keypoints": np.random.random((100, 68, 2)).astype(
            np.float32
        ),
    }


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for testing."""
    return {
        "id": "V00_S0809_I00000582_P0947",
        "metadata:transcript": [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ],
        "metadata:vad": [
            {"start": 0.0, "end": 0.5, "is_speech": True},
            {"start": 0.5, "end": 1.0, "is_speech": False},
        ],
        "annotations:1P-IS": {
            "interaction_id": "V00_S0809_I00000582",
            "participant_id": "P0947",
            "annotations": ["label1", "label2"],
        },
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment with common patches."""
    with patch("os.makedirs"):
        yield


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "network: marks tests requiring network access")


# Common test utilities
def create_mock_file_structure(base_path: Path, file_ids: list[str]):
    """Create a mock file structure for testing."""
    for file_id in file_ids:
        # Create different file types
        for ext in [".wav", ".mp4", ".json", ".npz"]:
            file_path = base_path / f"{file_id}{ext}"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()


def assert_valid_file_id(file_id: str):
    """Assert that a file ID follows the expected format."""
    import re

    pattern = r"V\d+_S\d+_I\d+_P\d+"
    assert re.match(pattern, file_id), f"Invalid file ID format: {file_id}"


def assert_valid_interaction_key(interaction_key: str):
    """Assert that an interaction key follows the expected format."""
    import re

    pattern = r"V\d+_S\d+_I\d+"
    assert re.match(pattern, interaction_key), (
        f"Invalid interaction key format: {interaction_key}"
    )


def assert_valid_session_key(session_key: str):
    """Assert that a session key follows the expected format."""
    import re

    pattern = r"V\d+_S\d+"
    assert re.match(pattern, session_key), f"Invalid session key format: {session_key}"


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "performance" in item.name.lower() or "large" in item.name.lower():
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.name.lower() or "workflow" in item.name.lower():
            item.add_marker(pytest.mark.integration)

        # Mark network tests
        if "network" in item.name.lower() or "download" in item.name.lower():
            item.add_marker(pytest.mark.network)

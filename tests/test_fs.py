# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from seamless_interaction.fs import (
    DatasetConfig,
    InteractionKey,
    SeamlessInteractionFS,
)


class TestDatasetConfig:
    """Test suite for DatasetConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Test DatasetConfig with default values."""
        config = DatasetConfig()

        assert config.label == "improvised"
        assert config.split == "dev"
        assert config.preferred_vendors_only is True
        assert config.num_workers is not None and config.num_workers > 0
        assert config.local_dir is not None and config.local_dir.endswith(
            "datasets/seamless_interaction"
        )
        assert config.seed is None

    def test_custom_initialization(self) -> None:
        """Test DatasetConfig with custom values."""
        config = DatasetConfig(
            label="naturalistic",
            split="train",
            preferred_vendors_only=False,
            num_workers=16,
            local_dir="/custom/path",
            seed=42,
        )

        assert config.label == "naturalistic"
        assert config.split == "train"
        assert config.preferred_vendors_only is False
        assert config.num_workers == 16
        assert config.local_dir == "/custom/path"
        assert config.seed == 42

    def test_auto_worker_detection(self) -> None:
        """Test automatic worker count detection."""
        with patch("os.cpu_count", return_value=16):
            config = DatasetConfig(num_workers=None)
            assert config.num_workers == 10  # min(10, max(1, 16-2))

        with patch("os.cpu_count", return_value=4):
            config = DatasetConfig(num_workers=None)
            assert config.num_workers == 2  # min(10, max(1, 4-2))

        with patch("os.cpu_count", return_value=1):
            config = DatasetConfig(num_workers=None)
            assert config.num_workers == 1  # min(10, max(1, 1-2))

    def test_default_local_dir(self) -> None:
        """Test default local directory path construction."""
        config = DatasetConfig(local_dir=None)
        expected_path = str(Path.home() / "datasets/seamless_interaction")
        assert config.local_dir == expected_path


class TestInteractionKey:
    """Test suite for InteractionKey dataclass."""

    def test_from_file_id_valid(self) -> None:
        """Test parsing valid file IDs."""
        file_id = "V00_S0809_I00000582_P0947"
        key = InteractionKey.from_file_id(file_id)

        assert key.vendor == "00"
        assert key.session == "0809"
        assert key.interaction == "00000582"
        assert key.participant == "0947"

    def test_from_file_id_invalid(self) -> None:
        """Test parsing invalid file IDs raises ValueError."""
        invalid_ids: List[str] = [
            "invalid_format",
            "V00_S0809_I00000582",  # Missing participant
            "V00_S0809_P0947",  # Missing interaction
            "S0809_I00000582_P0947",  # Missing vendor
            "",
            "V_S_I_P",  # Missing numbers
        ]

        for invalid_id in invalid_ids:
            with pytest.raises(ValueError, match="Invalid file ID format"):
                InteractionKey.from_file_id(invalid_id)

    def test_from_interaction_key_valid(self) -> None:
        """Test parsing valid interaction keys."""
        interaction_key = "V00_S0809_I00000582"
        key = InteractionKey.from_interaction_key(interaction_key)

        assert key.vendor == "00"
        assert key.session == "0809"
        assert key.interaction == "00000582"
        assert key.participant is None

    def test_from_interaction_key_invalid(self) -> None:
        """Test parsing invalid interaction keys raises ValueError."""
        invalid_keys: List[str] = [
            "invalid_format",
            "V00_S0809",  # Missing interaction
            "V00_I00000582",  # Missing session
            "S0809_I00000582",  # Missing vendor
            "",
        ]

        for invalid_key in invalid_keys:
            with pytest.raises(ValueError, match="Invalid interaction key format"):
                InteractionKey.from_interaction_key(invalid_key)

    def test_from_session_key_valid(self) -> None:
        """Test parsing valid session keys."""
        session_key = "V00_S0809"
        key = InteractionKey.from_session_key(session_key)

        assert key.vendor == "00"
        assert key.session == "0809"
        assert key.interaction is None
        assert key.participant is None

    def test_from_session_key_invalid(self) -> None:
        """Test parsing invalid session keys raises ValueError."""
        invalid_keys: List[str] = [
            "invalid_format",
            "V00",  # Missing session
            "S0809",  # Missing vendor
            "",
        ]

        for invalid_key in invalid_keys:
            with pytest.raises(ValueError, match="Invalid session key format"):
                InteractionKey.from_session_key(invalid_key)

    def test_file_id_property(self) -> None:
        """Test file_id property generation."""
        key = InteractionKey("00", "0809", "00000582", "0947")
        assert key.file_id == "V00_S0809_I00000582_P0947"

    def test_file_id_property_missing_participant(self) -> None:
        """Test file_id property raises error when participant is missing."""
        key = InteractionKey("00", "0809", "00000582")
        with pytest.raises(ValueError, match="Participant required for file ID"):
            _ = key.file_id

    def test_interaction_key_property(self) -> None:
        """Test interaction_key property generation."""
        key = InteractionKey("00", "0809", "00000582", "0947")
        assert key.interaction_key == "V00_S0809_I00000582"

    def test_interaction_key_property_missing_interaction(self) -> None:
        """Test interaction_key property raises error when interaction is missing."""
        key = InteractionKey("00", "0809", None, "0947")
        with pytest.raises(
            ValueError, match="Interaction required for interaction key"
        ):
            _ = key.interaction_key

    def test_session_key_property(self) -> None:
        """Test session_key property generation."""
        key = InteractionKey("00", "0809", "00000582", "0947")
        assert key.session_key == "V00_S0809"


class TestSeamlessInteractionFS:
    """Test suite for SeamlessInteractionFS class."""

    @pytest.fixture
    def mock_filelist_df(self) -> pd.DataFrame:
        """Create a mock filelist DataFrame for testing."""
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
                    "file_id": "V01_S0223_I00000127_P1505",
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
            ]
        )

    @pytest.fixture
    def fs_instance(self, mock_filelist_df: pd.DataFrame) -> SeamlessInteractionFS:
        """Create a SeamlessInteractionFS instance with mocked dependencies."""
        with patch.object(SeamlessInteractionFS, "_load_filelist_cache"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = DatasetConfig(local_dir=tmp_dir, num_workers=2)
                fs = SeamlessInteractionFS(config=config)
                fs._cached_filelist = mock_filelist_df
                return fs

    def test_initialization_default_config(self) -> None:
        """Test SeamlessInteractionFS initialization with default config."""
        with patch.object(SeamlessInteractionFS, "_load_filelist_cache"):
            with patch("os.makedirs"):
                fs = SeamlessInteractionFS()
                assert fs.config.label == "improvised"
                assert fs.config.split == "dev"
                assert fs.config.preferred_vendors_only is True

    def test_initialization_custom_config(self) -> None:
        """Test SeamlessInteractionFS initialization with custom config."""
        with patch.object(SeamlessInteractionFS, "_load_filelist_cache"):
            with patch("os.makedirs"):
                config = DatasetConfig(label="naturalistic", split="train")
                fs = SeamlessInteractionFS(config=config)
                assert fs.config.label == "naturalistic"
                assert fs.config.split == "train"

    def test_initialization_parameter_override(self) -> None:
        """Test parameter override during initialization."""
        with patch.object(SeamlessInteractionFS, "_load_filelist_cache"):
            with patch("os.makedirs"):
                config = DatasetConfig(local_dir="/original", num_workers=4)
                fs = SeamlessInteractionFS(
                    config=config, local_dir="/override", num_workers=8
                )
                assert fs.config.local_dir == "/override"
                assert fs.config.num_workers == 8

    def test_get_preferred_vendors(self, fs_instance: SeamlessInteractionFS) -> None:
        """Test preferred vendors list."""
        vendors: List[str] = fs_instance._get_preferred_vendors()
        assert vendors == ["V00", "V01"]

    def test_filter_candidates_default(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test filtering candidates with default config."""
        candidates: pd.DataFrame = fs_instance._filter_candidates()

        # Should filter for improvised/dev with preferred vendors only
        assert len(candidates) == 3  # V00 and V01 files
        assert all(candidates["label"] == "improvised")
        assert all(candidates["split"] == "dev")
        assert all(candidates["file_id"].str.startswith(("V00", "V01")))

    def test_filter_candidates_no_vendor_preference(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test filtering candidates without vendor preference."""
        candidates: pd.DataFrame = fs_instance._filter_candidates(
            preferred_vendors_only=False
        )

        # Should include all improvised/dev files
        assert len(candidates) == 3  # V00, V01, but not V03 (naturalistic)
        assert all(candidates["label"] == "improvised")
        assert all(candidates["split"] == "dev")

    def test_filter_candidates_custom_params(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test filtering candidates with custom parameters."""
        candidates: pd.DataFrame = fs_instance._filter_candidates(
            label="naturalistic", split="train", preferred_vendors_only=False
        )

        assert len(candidates) == 1
        assert candidates.iloc[0]["file_id"] == "V03_S0456_I00000789_P2001"

    def test_sample_random_file_ids(self, fs_instance: SeamlessInteractionFS) -> None:
        """Test random file ID sampling."""
        with patch("random.seed") as mock_seed:
            with patch("random.sample", return_value=["V00_S0809_I00000582_P0947"]):
                file_ids: List[str] = fs_instance.sample_random_file_ids(
                    num_samples=1, seed=42
                )

                mock_seed.assert_called_once_with(42)
                assert file_ids == ["V00_S0809_I00000582_P0947"]

    def test_sample_random_file_ids_no_candidates(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test random file ID sampling with no candidates."""
        with pytest.raises(ValueError, match="No candidates found"):
            fs_instance.sample_random_file_ids(label="nonexistent", num_samples=1)

    def test_get_available_keys_session(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting available session keys."""
        keys: List[str] = fs_instance.get_available_keys("session")
        expected_keys = ["V00_S0809", "V01_S0223"]  # From improvised/dev
        assert sorted(keys) == sorted(expected_keys)

    def test_get_available_keys_interaction(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting available interaction keys."""
        keys: List[str] = fs_instance.get_available_keys("interaction")
        expected_keys = ["V00_S0809_I00000582", "V01_S0223_I00000127"]
        assert sorted(keys) == sorted(expected_keys)

    def test_get_available_keys_file(self, fs_instance: SeamlessInteractionFS) -> None:
        """Test getting available file keys."""
        keys: List[str] = fs_instance.get_available_keys("file", limit=2)
        assert len(keys) == 2
        assert all("V0" in key for key in keys)

    def test_get_available_keys_invalid_type(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting available keys with invalid type."""
        with pytest.raises(ValueError, match="Invalid key_type"):
            fs_instance.get_available_keys("invalid_type")  # type: ignore[arg-type]

    def test_get_interaction_metadata(self, fs_instance: SeamlessInteractionFS) -> None:
        """Test getting interaction metadata."""
        metadata: Dict[str, Any] = fs_instance.get_interaction_metadata(
            "V00_S0809_I00000582"
        )

        assert metadata["interaction_key"] == "V00_S0809_I00000582"
        assert metadata["session_key"] == "V00_S0809"
        assert metadata["vendor"] == "00"
        assert metadata["session"] == "0809"
        assert metadata["interaction"] == "00000582"
        assert metadata["num_participants"] == 2
        assert "0947" in metadata["participants"]
        assert "0948" in metadata["participants"]

    def test_get_interaction_metadata_not_found(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting metadata for non-existent interaction."""
        metadata: Dict[str, Any] = fs_instance.get_interaction_metadata(
            "V99_S9999_I99999999"
        )
        assert metadata == {}

    def test_keys_to_file_ids_interaction(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test converting interaction keys to file IDs."""
        file_ids: List[str] = fs_instance._keys_to_file_ids(
            ["V00_S0809_I00000582"], "interaction"
        )

        expected_ids = ["V00_S0809_I00000582_P0947", "V00_S0809_I00000582_P0948"]
        assert sorted(file_ids) == sorted(expected_ids)

    def test_keys_to_file_ids_session(self, fs_instance: SeamlessInteractionFS) -> None:
        """Test converting session keys to file IDs."""
        file_ids: List[str] = fs_instance._keys_to_file_ids(["V00_S0809"], "session")

        expected_ids = ["V00_S0809_I00000582_P0947", "V00_S0809_I00000582_P0948"]
        assert sorted(file_ids) == sorted(expected_ids)

    def test_keys_to_file_ids_invalid_interaction_key(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test converting invalid interaction key raises error."""
        with pytest.raises(ValueError, match="Invalid interaction key format"):
            fs_instance._keys_to_file_ids(["invalid_key"], "interaction")

    def test_keys_to_file_ids_invalid_session_key(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test converting invalid session key raises error."""
        with pytest.raises(ValueError, match="Invalid session key format"):
            fs_instance._keys_to_file_ids(["invalid_key"], "session")

    def test_get_interaction_pairs_with_keys(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting interaction pairs with specific keys."""
        pairs: List[List[str]] = fs_instance.get_interaction_pairs(
            interaction_keys="V00_S0809_I00000582"
        )

        assert len(pairs) == 1
        assert len(pairs[0]) == 2
        assert "V00_S0809_I00000582_P0947" in pairs[0]
        assert "V00_S0809_I00000582_P0948" in pairs[0]

    def test_get_interaction_pairs_auto_sample(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test auto-sampling interaction pairs."""
        with patch.object(fs_instance, "_sample_interaction_pairs") as mock_sample:
            mock_sample.return_value = [["file1", "file2"]]

            pairs: List[List[str]] = fs_instance.get_interaction_pairs(num_pairs=1)

            mock_sample.assert_called_once_with(1, None, None, None)
            assert pairs == [["file1", "file2"]]

    def test_get_session_groups_with_keys(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting session groups with specific keys."""
        groups: List[List[str]] = fs_instance.get_session_groups(
            session_keys="V00_S0809", interactions_per_session=2
        )

        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert all("V00_S0809" in file_id for file_id in groups[0])

    def test_get_session_groups_auto_sample(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test auto-sampling session groups."""
        with patch.object(fs_instance, "_sample_session_groups") as mock_sample:
            mock_sample.return_value = [["file1", "file2", "file3"]]

            groups: List[List[str]] = fs_instance.get_session_groups(num_sessions=1)

            mock_sample.assert_called_once_with(1, 4, None, None, None)
            assert groups == [["file1", "file2", "file3"]]

    def test_list_batches(self, fs_instance: SeamlessInteractionFS) -> None:
        """Test listing available batches."""
        batches: List[int] = fs_instance.list_batches()
        assert batches == [0]  # Only batch 0 in improvised/dev

    def test_list_archives(self, fs_instance: SeamlessInteractionFS) -> None:
        """Test listing available archives."""
        archives: List[int] = fs_instance.list_archives(batch=0)
        assert sorted(archives) == [3, 5]  # Archives from test data

    def test_list_archives_empty(self, fs_instance: SeamlessInteractionFS) -> None:
        """Test listing archives for non-existent batch."""
        with patch("seamless_interaction.fs.logger") as mock_logger:
            archives: List[int] = fs_instance.list_archives(batch=999)
            assert archives == []
            mock_logger.error.assert_called_once()

    def test_get_path_list_for_file_id_s3(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting S3 paths for a file ID."""
        fs_instance._load_filelist_cache()
        paths: List[str] = fs_instance.get_path_list_for_file_id_s3(
            "V03_S0228_I00000581_P1528"
        )

        # Should include audio, video, and feature paths
        assert any("audio" in path for path in paths)
        assert any("video" in path for path in paths)
        assert any("smplh" in path for path in paths)
        assert any("movement" in path for path in paths)

    def test_get_path_list_for_file_id_s3_not_found(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting S3 paths for non-existent file ID."""
        with pytest.raises(ValueError, match="File ID .* not found"):
            fs_instance.get_path_list_for_file_id_s3("V99_S9999_I99999999_P9999")

    def test_get_path_list_for_file_id_local(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting local paths for a file ID."""
        with patch(
            "glob.glob", return_value=["/path/to/file.wav", "/path/to/file.mp4"]
        ):
            paths: List[str] = fs_instance.get_path_list_for_file_id_local(
                "V00_S0809_I00000582_P0947"
            )
            assert len(paths) == 2

    def test_get_path_list_for_file_id_local_not_found(
        self, fs_instance: SeamlessInteractionFS
    ) -> None:
        """Test getting local paths when no files exist."""
        with patch("glob.glob", return_value=[]):
            with patch("seamless_interaction.fs.logger") as mock_logger:
                paths: List[str] = fs_instance.get_path_list_for_file_id_local(
                    "V00_S0809_I00000582_P0947"
                )
                assert paths == []
                mock_logger.warning.assert_called_once()

    @patch("multiprocessing.Pool")
    def test_download_batch_from_hf(self, mock_pool, fs_instance):
        """Test downloading batch from HuggingFace."""
        mock_pool.return_value.__enter__.return_value.map.return_value = [
            (True, "/path")
        ]

        with patch.object(fs_instance, "list_archives", return_value=[0, 1]):
            result = fs_instance.download_batch_from_hf(batch_idx=0)

            assert result is True
            mock_pool.assert_called_once()

    @patch("seamless_interaction.fs.wget")
    def test_wget_download_from_s3_wav(self, mock_wget, fs_instance):
        """Test downloading WAV file from S3."""
        url = "http://example.com/file.wav"
        tmp_dir = "/tmp"
        target_path = "/target"
        shared_np_data = {}
        shared_json_data = {}
        lock = Mock()

        with patch("os.rename"):
            fs_instance._wget_download_from_s3(
                url, tmp_dir, target_path, shared_np_data, shared_json_data, lock
            )

            mock_wget.download.assert_called_once()

    def test_num_workers_property(self, fs_instance):
        """Test num_workers property getter and setter."""
        assert fs_instance.num_workers == 2

        fs_instance.num_workers = 8
        assert fs_instance.num_workers == 8
        assert fs_instance.config.num_workers == 8


def mock_open(read_data=""):
    """Helper to create a mock open function for file operations."""

    def _mock_open(*args, **kwargs):
        mock_file = Mock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.readlines.return_value = read_data.split("\n")
        mock_file.__iter__.return_value = iter(read_data.split("\n"))
        return mock_file

    return _mock_open


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_malformed_csv_handling(self):
        """Test handling of malformed CSV file."""
        with patch("pandas.read_csv", side_effect=Exception("CSV error")):
            with patch("os.makedirs"):
                with patch("seamless_interaction.fs.logger") as mock_logger:
                    fs = SeamlessInteractionFS()
                    mock_logger.warning.assert_called_once()
                    assert fs._cached_filelist.empty

    def test_mkdir_failure(self):
        """Test handling of directory creation failure."""
        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError):
                SeamlessInteractionFS()


if __name__ == "__main__":
    pytest.main([__file__])

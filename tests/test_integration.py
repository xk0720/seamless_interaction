# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from typing import Generator
from unittest.mock import patch

import pytest

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS


@pytest.mark.integration
class TestSeamlessInteractionFSIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def integration_fs(
        self, sample_filelist
    ) -> Generator[SeamlessInteractionFS, None, None]:
        """Create FS instance for integration testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = DatasetConfig(
                local_dir=tmp_dir, num_workers=2, label="improvised", split="dev"
            )

            with patch.object(SeamlessInteractionFS, "_load_filelist_cache"):
                fs = SeamlessInteractionFS(config=config)
                fs._cached_filelist = sample_filelist
                yield fs

    def test_interaction_pair_discovery_and_download(self, integration_fs):
        """Test discovering and downloading interaction pairs."""
        # Test interaction pair discovery
        pairs = integration_fs.get_interaction_pairs(
            interaction_keys="V00_S0809_I00000582"
        )

        assert len(pairs) == 1
        assert len(pairs[0]) == 2

        # Verify both participants from same interaction
        file_ids = pairs[0]
        assert "V00_S0809_I00000582_P0947" in file_ids
        assert "V00_S0809_I00000582_P0948" in file_ids

        # Test batch download
        with patch.object(integration_fs, "gather_file_id_data_from_s3"):
            with patch("multiprocessing.Pool") as mock_pool:
                mock_pool.return_value.__enter__.return_value.map.return_value = [
                    None,
                    None,
                ]

                result = integration_fs.download_batch_from_s3(file_ids)
                assert result is True

    def test_session_group_sampling_and_organization(self, integration_fs):
        """Test session group sampling and organization."""
        # Test session group discovery
        groups = integration_fs.get_session_groups(
            session_keys="V00_S0809", interactions_per_session=2
        )

        assert len(groups) == 1
        session_files = groups[0]
        assert len(session_files) == 2

        # Verify all files from same session
        for file_id in session_files:
            assert file_id.startswith("V00_S0809")

    def test_hf_batch_download_workflow(self, integration_fs):
        """Test HuggingFace batch download workflow."""
        with patch.object(integration_fs, "download_archive_from_hf") as mock_download:
            with patch.object(integration_fs, "list_archives") as mock_list:
                with patch("multiprocessing.Pool") as mock_pool:
                    # Setup mocks
                    mock_list.return_value = [0, 1, 2]
                    mock_download.return_value = (True, "/path/to/archive")
                    mock_pool.return_value.__enter__.return_value.map.return_value = [
                        (True, "/path1"),
                        (True, "/path2"),
                        (True, "/path3"),
                    ]

                    # Test batch download
                    result = integration_fs.download_batch_from_hf(batch_idx=0)

                    assert result is True
                    mock_list.assert_called_once()
                    mock_pool.assert_called_once()

    def test_metadata_extraction_and_validation(self, integration_fs):
        """Test metadata extraction and validation workflow."""
        interaction_key = "V00_S0809_I00000582"

        # Test metadata extraction
        metadata = integration_fs.get_interaction_metadata(interaction_key)

        # Validate metadata structure
        required_fields = [
            "interaction_key",
            "session_key",
            "vendor",
            "session",
            "interaction",
            "participants",
            "num_participants",
            "label",
            "split",
            "batch_idx",
            "archive_idx",
            "file_ids",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        # Validate data consistency
        assert metadata["interaction_key"] == interaction_key
        assert metadata["session_key"] == "V00_S0809"
        assert metadata["vendor"] == "00"
        assert metadata["session"] == "0809"
        assert metadata["interaction"] == "00000582"
        assert metadata["num_participants"] == len(metadata["participants"])
        assert len(metadata["file_ids"]) == metadata["num_participants"]

    def test_intelligent_sampling_workflow(self, integration_fs):
        """Test intelligent sampling with vendor preferences."""
        # Test preferred vendor sampling
        config = DatasetConfig(preferred_vendors_only=True)
        integration_fs.config = config

        candidates = integration_fs._filter_candidates()

        # Should only include preferred vendors (V00, V01)
        preferred_files = candidates[
            candidates["file_id"].str.startswith(("V00", "V01"))
        ]
        assert len(candidates) == len(preferred_files)

        # Test sampling with seed for reproducibility
        file_ids_1 = integration_fs.sample_random_file_ids(num_samples=2, seed=42)
        file_ids_2 = integration_fs.sample_random_file_ids(num_samples=2, seed=42)

        # Should be reproducible with same seed
        assert file_ids_1 == file_ids_2

    def test_error_handling_and_recovery(self, integration_fs):
        """Test error handling and recovery mechanisms."""
        # Test handling of non-existent file
        with pytest.raises(ValueError, match="File ID .* not found"):
            integration_fs.get_path_list_for_file_id_s3("V99_S9999_I99999999_P9999")

        # Test handling of empty candidates
        with pytest.raises(ValueError, match="No candidates found"):
            integration_fs.sample_random_file_ids(
                label="nonexistent_label", num_samples=1
            )

        # Test graceful handling of missing archives
        archives = integration_fs.list_archives(batch=999)
        assert archives == []

    def test_batch_processing_scalability(self, integration_fs):
        """Test batch processing with different scales."""
        # Test small batch
        small_batch = ["V00_S0809_I00000582_P0947"]

        with patch.object(integration_fs, "gather_file_id_data_from_s3"):
            with patch("multiprocessing.Pool") as mock_pool:
                mock_pool.return_value.__enter__.return_value.map.return_value = [None]

                result = integration_fs.download_batch_from_s3(small_batch)
                assert result is True

        # Test larger batch
        large_batch = [
            "V00_S0809_I00000582_P0947",
            "V00_S0809_I00000582_P0948",
            "V00_S0809_I00000583_P0949",
            "V00_S0809_I00000583_P0950",
            "V01_S0223_I00000127_P1505",
        ]

        with patch.object(integration_fs, "gather_file_id_data_from_s3"):
            with patch("multiprocessing.Pool") as mock_pool:
                mock_pool.return_value.__enter__.return_value.map.return_value = [
                    None
                ] * 5

                result = integration_fs.download_batch_from_s3(large_batch)
                assert result is True

    def test_config_driven_behavior(self, integration_fs):
        """Test that configuration properly drives behavior."""
        # Test num_workers configuration
        assert integration_fs.num_workers == 2

        integration_fs.num_workers = 8
        assert integration_fs.config.num_workers == 8

        # Test label/split configuration
        assert integration_fs.config.label == "improvised"
        assert integration_fs.config.split == "dev"

        # Test preferred vendors configuration
        integration_fs.config.preferred_vendors_only = False
        candidates = integration_fs._filter_candidates()

        # Should now include all vendors for improvised/dev
        all_improvised_dev = integration_fs._cached_filelist[
            (integration_fs._cached_filelist["label"] == "improvised")
            & (integration_fs._cached_filelist["split"] == "dev")
        ]
        assert len(candidates) == len(all_improvised_dev)


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance-focused integration tests."""

    def test_large_filelist_performance(self):
        """Test performance with large filelist."""
        # Create large mock filelist
        import pandas as pd

        large_filelist = []
        for vendor in range(10):
            for session in range(100):
                for interaction in range(10):
                    for participant in range(2):
                        large_filelist.append(
                            {
                                "file_id": f"V{vendor:02d}_S{session:04d}_I{interaction:08d}_P{participant:04d}",
                                "label": "improvised",
                                "split": "dev",
                                "batch_idx": vendor,
                                "archive_idx": session,
                            }
                        )

        large_df = pd.DataFrame(large_filelist)

        with patch.object(SeamlessInteractionFS, "_load_filelist_cache"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = DatasetConfig(local_dir=tmp_dir)
                fs = SeamlessInteractionFS(config=config)
                fs._cached_filelist = large_df

                # Test sampling performance
                import time

                start_time = time.time()

                file_ids = fs.sample_random_file_ids(num_samples=100)

                end_time = time.time()

                # Should complete in reasonable time (< 1 second)
                assert (end_time - start_time) < 1.0
                assert len(file_ids) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

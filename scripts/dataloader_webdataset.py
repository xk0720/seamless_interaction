# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

from datasets import load_dataset

from seamless_interaction.fs import SeamlessInteractionFS


def main():
    """
    Demonstrate webdataset loading for both local and remote datasets.

    This script shows how to download and load dataset archives using
    webdataset format, supporting both local file access and direct
    HuggingFace Hub streaming.

    :param mode: Loading mode ('local' or 'hf')
    :param label: Dataset label ('improvised' or 'naturalistic')
    :param split: Data split ('dev', 'test', 'train')
    :param batch_idx: Batch index number
    :param archive_idx: Archive index within the batch
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--label", type=str, default="improvised")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--archive_idx", type=int, default=23)
    args = parser.parse_args()

    fs = SeamlessInteractionFS()
    local_dir = Path.home() / "datasets/seamless_interaction"
    mode = args.mode
    label = args.label
    split = args.split
    batch_idx = args.batch_idx
    archive_idx = args.archive_idx

    fs.download_archive_from_hf(
        idx=batch_idx,
        archive=archive_idx,
        label=label,
        split=split,
        batch=batch_idx,
        local_dir=local_dir,
        extract=False,
    )

    if mode == "local":
        local_path = (
            local_dir / f"{label}/{split}/{batch_idx:04d}/{archive_idx:04d}.tar"
        )
        dataset = load_dataset(
            "webdataset", data_files={split: local_path}, split=split, streaming=True
        )
    elif mode == "hf":
        base_url = (
            f"https://huggingface.co/datasets/facebook/"
            f"seamless-interaction/resolve/main/{label}/{split}/"
            f"{batch_idx:04d}/{archive_idx:04d}.tar"
        )
        urls = [base_url.format(batch_idx=batch_idx, archive_idx=archive_idx)]
        dataset = load_dataset(
            "webdataset", data_files={split: urls}, split=split, streaming=True
        )

    for item in dataset:
        break

    print(item.keys())


if __name__ == "__main__":
    main()

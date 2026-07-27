from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import grain
import numpy as np

from dreamer.data.paired_rgb_adapter import (
    PAIRED_CONTRACT,
    PAIRED_SCHEMA_VERSION,
    SOURCE_SHARDS,
    SourceShard,
    convert_paired_rgb_corpus,
    sha256_array,
    sha256_file,
)
from dreamer.data.path_utils import discover_array_record_paths
from dreamer.data.transforms import EpisodeLengthFilter, ProcessEpisodeAndSlice


def _source_base(spec: SourceShard) -> int:
    split_offset = 100 if spec.split == "val" else 0
    source_offset = 20 if spec.source == "scripted_forward_v0" else 0
    return split_offset + source_offset


def _write_source_shard(path: Path, spec: SourceShard) -> None:
    lengths = np.asarray([3, 2], dtype=np.int64)
    offsets = np.asarray([0, 3], dtype=np.int64)
    total = int(lengths.sum())
    base = _source_base(spec)
    actions = (np.arange(total, dtype=np.int8) + base) % np.int8(9)
    frames = np.empty((total, 64, 64, 3), dtype=np.uint8)
    for row, action in enumerate(actions):
        frames[row].fill(np.uint8(int(action)))
        frames[row, 0, 0] = np.asarray(
            [base, row, int(action)],
            dtype=np.uint8,
        )

    first = np.asarray([True, False, False, True, False], dtype=np.bool_)
    done = np.asarray([False, False, True, False, True], dtype=np.bool_)
    terminal_cause = np.asarray([0, 0, 1, 0, 3], dtype=np.uint8)
    timestep = np.asarray([0, 1, 2, 0, 1], dtype=np.int32)
    rewards = np.asarray([0, 0, 10, 0, 0], dtype=np.float32)
    np.savez_compressed(
        path,
        schema_version=np.asarray(PAIRED_SCHEMA_VERSION),
        contract=np.asarray(PAIRED_CONTRACT),
        collector_version=np.asarray("2"),
        source=np.asarray(spec.source),
        episode_index=np.asarray(0, dtype=np.int64),
        provenance_json=np.asarray(
            json.dumps({"fixture": spec.filename}, sort_keys=True)
        ),
        trajectory_level_ids=np.asarray(
            [base + 1, base + 2],
            dtype=np.int32,
        ),
        trajectory_offsets=offsets,
        trajectory_lengths=lengths,
        effective_action=actions,
        reward=rewards,
        first=first,
        done=done,
        terminal_cause=terminal_cause,
        timestep=timestep,
        next_state_valid=~done,
        rgb64=frames,
    )


def _write_corpus(path: Path) -> None:
    path.mkdir(parents=True)
    for spec in SOURCE_SHARDS:
        _write_source_shard(path / spec.filename, spec)


def _payloads(path: Path) -> list[bytes]:
    paths = discover_array_record_paths(str(path))
    source = grain.sources.ArrayRecordDataSource(paths)
    return [source[index] for index in range(len(source))]


class PairedRgbAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.tmp_path = Path(temporary_directory.name)
        self.input_path = self.tmp_path / "paired-v2"
        self.output_path = self.tmp_path / "arrayrecords"
        _write_corpus(self.input_path)

    def _convert(self) -> dict:
        return convert_paired_rgb_corpus(
            self.input_path,
            self.output_path,
            records_per_shard=1,
        )

    def test_exact_frames_actions_and_alignment_survive_pixel_transform(self):
        self._convert()

        for spec in SOURCE_SHARDS:
            with self.subTest(shard=spec.filename):  # noqa: SIM117
                with np.load(
                    self.input_path / spec.filename,
                    allow_pickle=False,
                ) as source:
                    output_records = [
                        pickle.loads(payload)
                        for payload in _payloads(self.output_path / spec.split)
                        if pickle.loads(payload)["source_shard"] == spec.filename
                    ]
                    self.assertEqual(len(output_records), 2)
                    for trajectory_index, record in enumerate(output_records):
                        start = int(source["trajectory_offsets"][trajectory_index])
                        length = int(source["trajectory_lengths"][trajectory_index])
                        end = start + length
                        decoded_frames = np.frombuffer(
                            record["raw_video"],
                            dtype=np.uint8,
                        ).reshape(length, 64, 64, 3)
                        np.testing.assert_array_equal(
                            decoded_frames,
                            source["rgb64"][start:end],
                        )
                        np.testing.assert_array_equal(
                            record["actions"],
                            source["effective_action"][start:end],
                        )
                        np.testing.assert_array_equal(
                            record["rewards"],
                            source["reward"][start:end],
                        )
                        self.assertEqual(
                            record["paired_frame_sha256"],
                            sha256_array(source["rgb64"][start:end]),
                        )

                        payload = pickle.dumps(
                            record,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                        self.assertTrue(
                            EpisodeLengthFilter(
                                seq_len=length,
                                format_hint="coinrun",
                                print_filter_warnings=False,
                            ).filter(payload)
                        )
                        processed = ProcessEpisodeAndSlice(
                            seq_len=length,
                            image_h=64,
                            image_w=64,
                            image_c=3,
                        ).random_map(payload, np.random.default_rng(7))
                        np.testing.assert_array_equal(
                            processed["videos"],
                            source["rgb64"][start:end],
                        )
                        np.testing.assert_array_equal(
                            processed["actions"].categorical,
                            source["effective_action"][start:end],
                        )
                        np.testing.assert_array_equal(
                            processed["videos"][:, 0, 0, 2],
                            processed["actions"].categorical,
                        )

    def test_episode_split_and_source_boundaries_are_preserved(self):
        manifest = self._convert()

        self.assertEqual(manifest["num_episodes"], 8)
        self.assertEqual(manifest["num_frames"], 20)
        self.assertEqual(manifest["splits"]["train"]["num_episodes"], 4)
        self.assertEqual(manifest["splits"]["val"]["num_episodes"], 4)
        for split in ("train", "val"):
            payloads = _payloads(self.output_path / split)
            records = [pickle.loads(payload) for payload in payloads]
            self.assertEqual(len(records), 4)
            self.assertEqual({record["split"] for record in records}, {split})
            self.assertEqual(
                {record["source"] for record in records},
                {"random", "scripted_forward_v0"},
            )
            for record in records:
                length = record["sequence_length"]
                np.testing.assert_array_equal(
                    record["episode_starts"],
                    np.arange(length) == 0,
                )
                np.testing.assert_array_equal(
                    record["episode_ends"],
                    np.arange(length) == length - 1,
                )
                np.testing.assert_array_equal(
                    record["timestep"],
                    np.arange(length, dtype=np.int32),
                )
                np.testing.assert_array_equal(
                    record["next_state_valid"],
                    ~record["episode_ends"],
                )
                self.assertEqual(record["episode_length"], length)
                self.assertTrue(record["is_first_chunk"])
                self.assertTrue(record["is_last_chunk"])

        train_shards = {
            Path(path).name
            for path in discover_array_record_paths(str(self.output_path / "train"))
        }
        val_shards = {
            Path(path).name
            for path in discover_array_record_paths(str(self.output_path / "val"))
        }
        self.assertTrue(all("val" not in name for name in train_shards))
        self.assertEqual(train_shards, val_shards)

    def test_manifest_hashes_inputs_outputs_and_record_payloads(self):
        returned_manifest = self._convert()
        stored_manifest = json.loads(
            (self.output_path / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(stored_manifest, returned_manifest)

        for source_manifest in stored_manifest["sources"]:
            source_path = self.input_path / source_manifest["filename"]
            self.assertEqual(source_manifest["sha256"], sha256_file(source_path))
            payload_by_id = {
                pickle.loads(payload)["episode_id"]: payload
                for payload in _payloads(self.output_path / source_manifest["split"])
                if pickle.loads(payload)["source_shard"] == source_manifest["filename"]
            }
            for episode in source_manifest["episodes"]:
                self.assertEqual(
                    episode["record_sha256"],
                    hashlib.sha256(payload_by_id[episode["episode_id"]]).hexdigest(),
                )
            for output_shard in source_manifest["output_shards"]:
                output_path = self.output_path / output_shard["path"]
                self.assertEqual(output_shard["sha256"], sha256_file(output_path))
                self.assertEqual(
                    output_shard["size_bytes"],
                    output_path.stat().st_size,
                )

    def test_invalid_episode_boundary_is_rejected_before_output_is_published(self):
        source_path = self.input_path / SOURCE_SHARDS[0].filename
        with np.load(source_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays["done"] = np.zeros_like(arrays["done"])
        arrays["next_state_valid"] = ~arrays["done"]
        np.savez_compressed(source_path, **arrays)

        with self.assertRaisesRegex(ValueError, "invalid done labels"):
            self._convert()
        self.assertFalse(self.output_path.exists())

    def test_arrayrecord_discovery_is_sorted(self):
        data_path = self.tmp_path / "sort-check"
        data_path.mkdir()
        for name in ("z.array_record", "a.array_record", "m.array_record"):
            (data_path / name).touch()
        self.assertEqual(
            [Path(path).name for path in discover_array_record_paths(str(data_path))],
            ["a.array_record", "m.array_record", "z.array_record"],
        )


if __name__ == "__main__":
    unittest.main()

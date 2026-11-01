import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "engram_mmap.py"
SPEC = importlib.util.spec_from_file_location("engram_mmap", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
engram_mmap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engram_mmap)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class MmapEngramTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = 8
        self.dim = 4
        self.scale_cols = 2
        self.weight_bytes = bytes(range(self.rows * self.dim))
        self.scale_bytes = bytes(range(100, 100 + self.rows * self.scale_cols))
        weight_name = "layers.14.engram.embed.weight"
        scale_name = "layers.14.engram.embed.scale"
        other_weight_name = "layers.1.engram.embed.weight"
        other_scale_name = "layers.1.engram.embed.scale"
        self.other_weight_bytes = bytes(range(200, 200 + self.rows * self.dim))
        self.other_scale_bytes = bytes(range(150, 150 + self.rows * self.scale_cols))
        write_safetensors(
            self.root / "weight.safetensors",
            {
                weight_name: ("F8_E4M3", [self.rows, self.dim], self.weight_bytes),
                other_weight_name: (
                    "F8_E4M3",
                    [self.rows, self.dim],
                    self.other_weight_bytes,
                ),
            },
        )
        write_safetensors(
            self.root / "scale.safetensors",
            {
                scale_name: (
                    "F8_E8M0",
                    [self.rows, self.scale_cols],
                    self.scale_bytes,
                ),
                other_scale_name: (
                    "F8_E8M0",
                    [self.rows, self.scale_cols],
                    self.other_scale_bytes,
                ),
            },
        )
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        weight_name: "weight.safetensors",
                        scale_name: "scale.safetensors",
                        other_weight_name: "weight.safetensors",
                        other_scale_name: "scale.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_table(self):
        return engram_mmap.MmapEngramTable(
            model_path=str(self.root),
            layer_id=14,
            rows=self.rows,
            dim=self.dim,
            scale_cols=self.scale_cols,
            start_row=2,
            local_rows=4,
            workers=2,
            chunk_rows=1,
        )

    def test_layer_id_selects_table_when_shapes_match(self) -> None:
        table = engram_mmap.MmapEngramTable(
            model_path=str(self.root),
            layer_id=1,
            rows=self.rows,
            dim=self.dim,
            scale_cols=self.scale_cols,
            start_row=2,
            local_rows=4,
            workers=2,
            chunk_rows=1,
        )
        ids = np.array([0, 1], dtype=np.int64)
        weights = np.empty((2, self.dim), dtype=np.uint8)
        scales = np.empty((2, self.scale_cols), dtype=np.uint8)

        table.gather_into(ids, weights, scales)

        expected_rows = np.array([2, 3])
        np.testing.assert_array_equal(
            weights,
            np.frombuffer(self.other_weight_bytes, dtype=np.uint8).reshape(
                self.rows, self.dim
            )[expected_rows],
        )
        np.testing.assert_array_equal(
            scales,
            np.frombuffer(self.other_scale_bytes, dtype=np.uint8).reshape(
                self.rows, self.scale_cols
            )[expected_rows],
        )
        table.pool.shutdown(wait=True)

    def test_gather_maps_local_rows_and_restores_duplicates(self) -> None:
        table = self.make_table()
        ids = np.array([3, 0, 3, 1], dtype=np.int64)
        weights = np.empty((4, self.dim), dtype=np.uint8)
        scales = np.empty((4, self.scale_cols), dtype=np.uint8)

        table.gather_into(ids, weights, scales)

        expected_rows = np.array([5, 2, 5, 3])
        expected_weights = np.frombuffer(self.weight_bytes, dtype=np.uint8).reshape(
            self.rows, self.dim
        )[expected_rows]
        expected_scales = np.frombuffer(self.scale_bytes, dtype=np.uint8).reshape(
            self.rows, self.scale_cols
        )[expected_rows]
        np.testing.assert_array_equal(weights, expected_weights)
        np.testing.assert_array_equal(scales, expected_scales)
        table.pool.shutdown(wait=True)

    def test_rejects_rows_outside_rank_local_shard(self) -> None:
        table = self.make_table()
        with self.assertRaises(IndexError):
            table.gather_into(
                np.array([4]),
                np.empty((1, self.dim), dtype=np.uint8),
                np.empty((1, self.scale_cols), dtype=np.uint8),
            )
        table.pool.shutdown(wait=True)

    def test_rejects_shard_outside_model_directory(self) -> None:
        index_path = self.root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"]["layers.14.engram.embed.weight"] = "../weight.safetensors"
        index_path.write_text(json.dumps(index), encoding="utf-8")

        with self.assertRaises(ValueError):
            self.make_table()

    def test_rejects_truncated_header(self) -> None:
        (self.root / "weight.safetensors").write_bytes(b"\x10\x00\x00")
        with self.assertRaises(ValueError):
            self.make_table()


if __name__ == "__main__":
    unittest.main()

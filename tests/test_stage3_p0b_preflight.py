"""Stage-3 P0-B local provenance tests, synthetic fixtures only."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p0b_preflight", ROOT / "scripts" / "stage3_p0b_local_preflight.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class P0BProvenanceTests(unittest.TestCase):
    def test_metadata_index_digest_deterministic_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(30):
                (root / f"sample_{i:02}.json").write_text(
                    '{"train":[],"test":[]}', encoding="utf-8")
            h1, names = p.json_index_hash(root)
            h2, names2 = p.json_index_hash(root)
            self.assertEqual(h1, h2)
            self.assertEqual(names, names2)
            self.assertEqual(len(names), 30)
            (root / "sample_02.json").write_text(
                '{"train":[1],"test":[]}', encoding="utf-8")
            h3, _ = p.json_index_hash(root)
            self.assertNotEqual(h1, h3)

    def test_rejects_wrong_product_count(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "one.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                p.json_index_hash(Path(d))

    def test_git_blob_matches_frozen_v22_without_importing_torch(self):
        import json
        manifest = json.loads((ROOT / "configs/stage3/experiment_manifest_v1.json").read_text())
        self.assertEqual(
            p.git_blob_sha(manifest["source"]["v22_path"]),
            manifest["source"]["v22_blob_sha"])


if __name__ == "__main__":
    unittest.main()

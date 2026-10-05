"""Tests for config loading, incl. JSONC comment stripping. No I/O beyond temp.

Run with: python -m unittest discover -s tests
"""

import tempfile
import unittest
from pathlib import Path

from tree4cfd.config import _strip_json_comments, load_config


class CommentStripping(unittest.TestCase):
    def test_line_and_block_comments_removed(self):
        text = '{\n  "a": 1, // line comment\n  /* block */ "b": 2\n}'
        self.assertEqual(_strip_json_comments(text).count("comment"), 0)

    def test_slashes_inside_strings_preserved(self):
        text = '{"url": "https://example.com/path"}  // trailing'
        out = _strip_json_comments(text)
        self.assertIn("https://example.com/path", out)
        self.assertNotIn("trailing", out)


class LoadConfigJsonc(unittest.TestCase):
    def test_loads_commented_config(self):
        text = """{
            "lod": 2.1,                  // detail level
            "influence_region": 500,     // radius m
            "cleaning": { "voxel_size": 0.5 }  // keep all? no
        }"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(text)
            cfg = load_config(p)
        self.assertEqual(cfg.lod, 2.1)
        self.assertEqual(cfg.influence_region, 500)
        self.assertEqual(cfg.cleaning.voxel_size, 0.5)

    def test_unknown_key_still_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text('{ "lod": 3, "bogus": 1 }')
            with self.assertRaises(ValueError):
                load_config(p)


if __name__ == "__main__":
    unittest.main()

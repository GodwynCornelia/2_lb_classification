"""Regressions for the damaged interpretation headings found in the delivered file."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import nbformat

from notebook_quality import (INTERPRETATION_HEADING, apply_interpretations,
                              validate_notebook_text, write_notebook_utf8)


class NotebookTextTests(unittest.TestCase):
    def setUp(self):
        self.analyses = {"setup": "Среда проверена: точность 0.9425. Почему? Причина описана."}
        self.notebook = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell("", metadata={"additional_analysis": "setup"})])
        apply_interpretations(self.notebook, self.analyses)

    def test_utf8_roundtrip_preserves_russian(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.ipynb"
            write_notebook_utf8(self.notebook, path)
            decoded = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
        validate_notebook_text(decoded, self.analyses)
        self.assertEqual(decoded.cells[0].source.splitlines()[0], INTERPRETATION_HEADING)

    def test_detects_the_delivered_question_mark_headings(self):
        damaged = deepcopy(self.notebook)
        damaged.cells[0].source = "### ????????????? ???????????\n\n" + self.analyses["setup"]
        with self.assertRaisesRegex(ValueError, "Damaged notebook text"):
            validate_notebook_text(damaged)

    def test_detects_corrupted_body_and_replacement_characters(self):
        for damage in ["????", "\ufffd", "\u00d0\u0098\u00d0\u00bd", "\u0000"]:
            with self.subTest(damage=repr(damage)):
                damaged = deepcopy(self.notebook)
                damaged.cells[0].source += damage
                with self.assertRaises(ValueError):
                    validate_notebook_text(damaged)

    def test_checks_rendered_output_text(self):
        self.notebook.cells.append(nbformat.v4.new_code_cell(
            "print('value')", outputs=[nbformat.v4.new_output("stream", name="stdout", text="????")]))
        with self.assertRaisesRegex(ValueError, "output"):
            validate_notebook_text(self.notebook)

    def test_accepts_supported_terminal_colors_in_plain_output(self):
        self.notebook.cells.append(nbformat.v4.new_code_cell(
            "model.summary()", outputs=[nbformat.v4.new_output(
                "display_data", data={"text/plain": "\x1b[1mModel summary\x1b[0m"})]))
        validate_notebook_text(self.notebook, self.analyses)

    def test_catches_stale_or_modified_numeric_interpretation(self):
        self.notebook.cells[0].source = self.notebook.cells[0].source.replace("0.9425", "0.9999")
        with self.assertRaisesRegex(ValueError, "differs from measured"):
            validate_notebook_text(self.notebook, self.analyses)


if __name__ == "__main__":
    unittest.main()

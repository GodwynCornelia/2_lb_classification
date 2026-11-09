"""Validate readable notebook text and assemble interpretations in native UTF-8."""
from collections.abc import Mapping
from pathlib import Path
import re

import nbformat

INTERPRETATION_HEADING = "### Интерпретация результатов"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SUSPICIOUS_TEXT = re.compile(
    r"\?{3,}|\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
    r"|(?:[\u00d0\u00d1][\u0080-\u00bf]){2,}"
    r"|(?:[\u0420\u0421].){5,}"
)


def _check_text(text, location, allow_ansi=False):
    if allow_ansi:
        # Jupyter supports ANSI colors in stream/text/plain outputs, including Keras summaries.
        text = ANSI_ESCAPE.sub("", text)
    match = SUSPICIOUS_TEXT.search(text)
    if match:
        excerpt = text[max(0, match.start()-20):match.end()+30]
        raise ValueError(f"Damaged notebook text at {location}: {excerpt!r}")


def validate_notebook_text(notebook, expected_analyses=None):
    """Check sources, rendered text outputs, and measured interpretation contents."""
    for index, cell in enumerate(notebook.cells):
        _check_text(cell.source, f"cell {index} source")
        key = cell.metadata.get("additional_analysis", cell.metadata.get("lab2_analysis"))
        if key is not None:
            prefix = INTERPRETATION_HEADING + "\n\n"
            if not cell.source.startswith(prefix):
                raise ValueError(f"Invalid interpretation heading in cell {index} ({key})")
            if expected_analyses is not None:
                expected = prefix + expected_analyses[key]
                if cell.source != expected:
                    raise ValueError(f"Interpretation differs from measured analysis: {key}")
        for output_index, output in enumerate(cell.get("outputs", [])):
            location = f"cell {index} output {output_index}"
            if output.get("output_type") == "error":
                raise ValueError(f"Execution error at {location}: {output.get('evalue')}")
            if isinstance(output.get("text"), str):
                _check_text(output["text"], location, allow_ansi=True)
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/") and isinstance(value, str):
                    _check_text(value, f"{location} {mime}", allow_ansi=mime == "text/plain")


def apply_interpretations(notebook, analyses: Mapping[str, str]):
    """Build headings in this UTF-8 Python source, without shell text interpolation."""
    for cell in notebook.cells:
        key = cell.metadata.get("additional_analysis", cell.metadata.get("lab2_analysis"))
        if key is not None:
            cell.source = INTERPRETATION_HEADING + "\n\n" + analyses[key]
    validate_notebook_text(notebook, expected_analyses=analyses)


def write_notebook_utf8(notebook, path):
    nbformat.validate(notebook)
    validate_notebook_text(notebook)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        nbformat.write(notebook, stream)

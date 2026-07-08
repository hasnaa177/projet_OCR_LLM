import re
import tempfile
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        _converter = PdfConverter(artifact_dict=create_model_dict())
    return _converter


def _fix_table_headers(markdown_text: str) -> str:
    """
    Comme Docling, Marker traite systematiquement la 1ere ligne de chaque
    tableau comme un en-tete de colonne, alors que dans ces formulaires ce
    sont des paires label/valeur equivalentes (pas de vrai en-tete
    semantique). On reinsere cette ligne comme donnee normale, sous un
    en-tete neutre generique dont le nombre de colonnes s'adapte
    automatiquement au tableau detecte.
    """
    lines = markdown_text.split("\n")
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_table_header = (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\|[\s\-|]+\|$", lines[i + 1].strip())
        )
        if is_table_header:
            original_header_row = line
            num_cols = len([c for c in original_header_row.strip().strip("|").split("|")])

            if num_cols == 2:
                col_names = ["Field", "Value"]
            else:
                col_names = [f"Col {j+1}" for j in range(num_cols)]

            neutral_header = "| " + " | ".join(col_names) + " |"
            neutral_separator = "|" + "|".join(["---"] * num_cols) + "|"

            output.append(neutral_header)
            output.append(neutral_separator)
            output.append(original_header_row)  # remise en donnee normale
            i += 2
        else:
            output.append(line)
            i += 1
    return "\n".join(output)


def run_ocr_marker(pdf_bytes: bytes) -> list:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        rendered = _get_converter()(tmp_path)
        markdown_text, _, images = text_from_rendered(rendered)
        markdown_text = _fix_table_headers(markdown_text)
    finally:
        os.unlink(tmp_path)

    return [{
        "page_number": 1,
        "text": markdown_text,
        "bounding_boxes": []
    }]
import re
import tempfile
import os
from docling.document_converter import DocumentConverter

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def _fix_table_headers(markdown_text: str) -> str:
    """
    Docling traite systematiquement la 1ere ligne de chaque tableau comme un
    en-tete de colonne, alors que dans ces formulaires ce sont des paires
    label/valeur equivalentes (pas de vrai en-tete semantique). On reinsere
    cette ligne comme donnee normale, sous un en-tete neutre generique dont
    le nombre de colonnes s'adapte automatiquement au tableau detecte.
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


def run_ocr_docling(pdf_bytes: bytes, do_ocr: bool = True) -> list:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        if do_ocr:
            # Comportement par defaut inchange -- pipeline complet (OCR actif),
            # necessaire pour les PDF scannes (data/scanned/) qui n'ont pas de
            # couche texte native.
            converter = _get_converter()
        else:
            # Court-circuite l'OCR : utilise UNIQUEMENT la couche texte deja
            # presente dans le PDF. Beaucoup plus rapide, mais ne fonctionne
            # que sur des PDF "nes numeriques" (data/pdfs/) -- sur un PDF
            # scanne sans texte integre, ceci renvoie un texte vide.
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )

        result = converter.convert(tmp_path)
        markdown_text = result.document.export_to_markdown()
        markdown_text = _fix_table_headers(markdown_text)
        bounding_boxes = _extract_bounding_boxes(result.document, page_number=1)
    finally:
        os.unlink(tmp_path)

    return [{
        "page_number": 1,
        "text": markdown_text,
        "bounding_boxes": bounding_boxes
    }]


def _extract_bounding_boxes(doc, page_number: int = 1) -> list:
    """
    Extrait les bbox reellement detectees par Docling (layout DocLayNet pour
    les elements courants, TableFormer pour les cellules de tableau), au
    meme format que celui utilise par les autres moteurs (EasyOCR/Tesseract) :
    [{"text": str, "bbox": [[x0,y0],[x1,y0],[x1,y1],[x0,y1]], "label": str}, ...]
    Fonction purement additive : ne modifie ni ne consulte run_ocr_docling.
    """
    from docling_core.types.doc import CoordOrigin
    from docling_core.types.doc.document import TableItem

    page_data = doc.pages.get(page_number)
    if page_data is None:
        return []

    page_height = page_data.size.height
    boxes = []

    def to_top_left(bbox):
        if bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
            top = page_height - bbox.t
            bottom = page_height - bbox.b
        else:
            top, bottom = bbox.t, bbox.b
        x0, x1 = bbox.l, bbox.r
        return [[x0, top], [x1, top], [x1, bottom], [x0, bottom]]

    for item, _level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue

        if isinstance(item, TableItem):
            for cell in item.data.table_cells:
                if cell.bbox is None or item.prov[0].page_no != page_number:
                    continue
                boxes.append({
                    "text": cell.text,
                    "bbox": to_top_left(cell.bbox),
                    "label": "table_cell",
                })
        else:
            for prov in item.prov:
                if prov.page_no != page_number:
                    continue
                boxes.append({
                    "text": getattr(item, "text", "") or "",
                    "bbox": to_top_left(prov.bbox),
                    "label": str(getattr(item, "label", "")),
                })

    return boxes

# ==========================================
# VISUALISATION DES ZONES DETECTEES (additif, ne touche pas au pipeline existant)
# ==========================================
def render_annotated_image(pdf_bytes: bytes, page_number: int = 1) -> bytes:
    """
    Dessine les vraies bbox detectees par Docling (layout DocLayNet pour les
    titres/sections, TableFormer pour les cellules de tableau) par-dessus
    l'image de la page. Une boite par cellule pour les tableaux, une boite
    par element pour le reste.
    """
    import tempfile
    import os
    from PIL import ImageDraw
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling_core.types.doc import CoordOrigin
    from docling_core.types.doc.document import TableItem

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        result = converter.convert(tmp_path)
        doc = result.document

        page_data = doc.pages.get(page_number)
        if page_data is None or page_data.image is None:
            raise ValueError(f"Page {page_number} introuvable ou sans image rendue.")

        img = page_data.image.pil_image.convert("RGB")
        draw = ImageDraw.Draw(img)

        scale_x = img.width / page_data.size.width
        scale_y = img.height / page_data.size.height

        def draw_bbox(bbox):
            if bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
                top = page_data.size.height - bbox.t
                bottom = page_data.size.height - bbox.b
            else:
                top, bottom = bbox.t, bbox.b
            x0, x1 = bbox.l * scale_x, bbox.r * scale_x
            y0, y1 = top * scale_y, bottom * scale_y
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

        for item, _level in doc.iterate_items():
            if not getattr(item, "prov", None):
                continue

            if isinstance(item, TableItem):
                for cell in item.data.table_cells:
                    if cell.bbox is None:
                        continue
                    if item.prov[0].page_no != page_number:
                        continue
                    draw_bbox(cell.bbox)
            else:
                for prov in item.prov:
                    if prov.page_no != page_number:
                        continue
                    draw_bbox(prov.bbox)

        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    finally:
        os.unlink(tmp_path)
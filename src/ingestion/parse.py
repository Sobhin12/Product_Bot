from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from . import config


def parse_pdf(pdf: Path) -> DoclingDocument:
    """Parse a PDF with Docling, caching the DoclingDocument as JSON."""
    cache = config.CACHE_DIR / f"{pdf.stem}.json"
    if cache.exists() and cache.stat().st_mtime >= pdf.stat().st_mtime:
        return DoclingDocument.load_from_json(cache)

    # OCR off: inputs are text PDFs, and the OCR models are downloaded from a
    # host that is not always reachable.
    options = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    doc = converter.convert(pdf).document
    cache.parent.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(cache)
    return doc

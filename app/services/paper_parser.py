"""Wrapper around PaperMage for structured PDF parsing.

Falls back to a lightweight pdfminer-based extraction when PaperMage is not
installed so the backend can still start for development.
"""

import os
import tempfile
from typing import Any

try:
    from papermage.recipes import CoreRecipe
    _recipe: CoreRecipe | None = None

    def _get_recipe() -> CoreRecipe:
        global _recipe
        if _recipe is None:
            _recipe = CoreRecipe()
        return _recipe

    HAS_PAPERMAGE = True
except ImportError:
    HAS_PAPERMAGE = False


def parse_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    try:
        if HAS_PAPERMAGE:
            return _parse_with_papermage(tmp_path)
        return _parse_fallback(tmp_path, pdf_bytes)
    finally:
        os.unlink(tmp_path)


def _parse_with_papermage(path: str) -> dict[str, Any]:
    recipe = _get_recipe()
    doc = recipe.run(path)

    sections: list[dict[str, Any]] = []
    for i, sec in enumerate(getattr(doc, 'sections', []) or []):
        text = sec.text if hasattr(sec, 'text') else ''
        title = text.split('\n')[0][:120] if text else f'Section {i}'
        sections.append({
            'id': f'sec-{i}',
            'title': title.strip(),
            'text': text,
            'start_char': getattr(sec, 'start', i * 1000),
            'end_char': getattr(sec, 'end', (i + 1) * 1000),
        })

    paragraphs: list[dict[str, Any]] = []
    for i, para in enumerate(getattr(doc, 'paragraphs', []) or []):
        paragraphs.append({
            'id': f'para-{i}',
            'text': para.text if hasattr(para, 'text') else '',
            'start_char': getattr(para, 'start', 0),
            'end_char': getattr(para, 'end', 0),
        })

    citations: list[dict[str, Any]] = []
    for i, bib in enumerate(getattr(doc, 'bibliographies', []) or []):
        citations.append({
            'id': f'cit-{i}',
            'raw_text': bib.text if hasattr(bib, 'text') else '',
        })

    figures: list[dict[str, Any]] = []
    for i, fig in enumerate(getattr(doc, 'figures', []) or []):
        cap = ''
        if hasattr(fig, 'captions') and fig.captions:
            cap = fig.captions[0].text if hasattr(fig.captions[0], 'text') else ''
        figures.append({
            'id': f'fig-{i}',
            'caption': cap,
        })

    tables: list[dict[str, Any]] = []
    for i, tbl in enumerate(getattr(doc, 'tables', []) or []):
        cap = ''
        if hasattr(tbl, 'captions') and tbl.captions:
            cap = tbl.captions[0].text if hasattr(tbl.captions[0], 'text') else ''
        tables.append({
            'id': f'tbl-{i}',
            'caption': cap,
        })

    metadata: dict[str, Any] = {}
    if hasattr(doc, 'titles') and doc.titles:
        metadata['title'] = doc.titles[0].text
    if hasattr(doc, 'abstracts') and doc.abstracts:
        metadata['abstract'] = doc.abstracts[0].text
    if hasattr(doc, 'authors') and doc.authors:
        metadata['authors'] = [a.text for a in doc.authors]

    full_text = doc.text if hasattr(doc, 'text') else ''

    return {
        'full_text': full_text,
        'metadata': metadata,
        'sections': sections,
        'paragraphs': paragraphs,
        'citations': citations,
        'figures': figures,
        'tables': tables,
    }


def _parse_fallback(path: str, pdf_bytes: bytes) -> dict[str, Any]:
    """Minimal fallback when PaperMage is unavailable."""
    try:
        from pdfminer.high_level import extract_text
        full_text = extract_text(path)
    except ImportError:
        full_text = ''

    return {
        'full_text': full_text,
        'metadata': {},
        'sections': [],
        'paragraphs': [],
        'citations': [],
        'figures': [],
        'tables': [],
    }

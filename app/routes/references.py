import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from flask import Blueprint, request, jsonify, Response

from app.services import document_store
from app.services.paper_parser import parse_pdf
from app.config import MAX_PDF_DOWNLOAD_SIZE_MB, RESOLVE_WORKERS, S2_MAX_RPS

log = logging.getLogger(__name__)

bp = Blueprint('references', __name__)

S2_API_BASE = 'https://api.semanticscholar.org/graph/v1'
S2_SEARCH_URL = f'{S2_API_BASE}/paper/search'
S2_PAPER_URL = f'{S2_API_BASE}/paper'
S2_BATCH_URL = f'{S2_API_BASE}/paper/batch'
S2_FIELDS = 'openAccessPdf,title,abstract,year,authors,externalIds,url'


# ---------------------------------------------------------------------------
# S2 token-bucket rate limiter
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Thread-safe token bucket that enforces a max requests-per-second."""

    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / rps
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                wait = self._next_allowed - now
                self._next_allowed += self._interval
            else:
                wait = 0.0
                self._next_allowed = now + self._interval
        if wait > 0:
            time.sleep(wait)


_s2_bucket = _TokenBucket(S2_MAX_RPS)


# ---------------------------------------------------------------------------
# POST /api/extract-references
# ---------------------------------------------------------------------------

@bp.route('/api/extract-references', methods=['POST'])
def extract_references():
    body = request.get_json(silent=True) or {}
    doc_id = body.get('doc_id', '')

    if not doc_id:
        return jsonify({'error': 'doc_id is required'}), 400

    doc = document_store.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404

    force = body.get('force', False)
    cache_key = f'{doc_id}_refs'
    if not force:
        cached = document_store.get(cache_key)
        if cached:
            return jsonify(cached)

    full_text = doc.get('full_text', '')
    citation_strings = _get_citation_strings(doc, full_text)

    log.info('Resolving %d references for doc %s', len(citation_strings), doc_id[:12])

    id_map: dict[int, dict[str, str]] = {}
    batch_ids: list[str] = []
    batch_idx_to_pos: dict[int, int] = {}
    for i, raw in enumerate(citation_strings):
        ids = _extract_identifiers(raw)
        id_map[i] = ids
        s2_ref = None
        if ids.get('doi'):
            s2_ref = f'DOI:{ids["doi"]}'
        elif ids.get('arxiv'):
            s2_ref = f'ARXIV:{ids["arxiv"]}'
        if s2_ref:
            batch_idx_to_pos[len(batch_ids)] = i
            batch_ids.append(s2_ref)

    batch_results: dict[int, dict[str, Any]] = {}
    if batch_ids:
        batch_papers = _s2_batch_lookup(batch_ids)
        for bi, paper in enumerate(batch_papers):
            if paper:
                batch_results[batch_idx_to_pos[bi]] = paper

    log.info('Batch resolved %d/%d via S2 batch API', len(batch_results), len(batch_ids))

    needs_resolve: list[int] = [i for i in range(len(citation_strings)) if i not in batch_results]

    resolved_map: dict[int, dict[str, Any] | None] = {}
    with ThreadPoolExecutor(max_workers=RESOLVE_WORKERS) as pool:
        futures = {
            pool.submit(_resolve_reference, i, citation_strings[i], id_map.get(i, {})): i
            for i in needs_resolve
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                resolved_map[idx] = fut.result()
            except Exception:
                log.exception('Unexpected error resolving ref %d', idx)
                resolved_map[idx] = None

    for i in range(len(citation_strings)):
        if i in batch_results:
            paper = batch_results[i]
            existing_oap = (paper.get('openAccessPdf') or {}).get('url')
            if not existing_oap:
                arxiv_id = id_map.get(i, {}).get('arxiv')
                fallback = (_try_arxiv_pdf_from_id(arxiv_id) if arxiv_id else None) \
                    or _find_best_pdf_url(paper, citation_strings[i])
                if fallback:
                    paper['openAccessPdf'] = {'url': fallback}
            resolved_map[i] = paper

    references = [_build_ref_obj(i, citation_strings[i], resolved_map.get(i))
                  for i in range(len(citation_strings))]

    pdf_count = sum(1 for r in references if r['open_access_pdf_url'])
    log.info('Done: %d/%d have PDF URLs', pdf_count, len(references))

    result = {'references': references}
    document_store.put(cache_key, result)
    return jsonify(result)


def _get_citation_strings(doc: dict[str, Any], full_text: str) -> list[str]:
    """Return citation strings, with PaperMage/regex cross-check."""
    raw_citations = doc.get('citations', [])
    pm_strings = [c.get('raw_text', '') for c in raw_citations if c.get('raw_text', '').strip()]

    if not pm_strings:
        return _extract_references_from_text(full_text)

    bracket_max = _max_bracket_index(full_text)
    if bracket_max > 0 and abs(len(pm_strings) - bracket_max) > max(5, bracket_max * 0.25):
        log.warning(
            'PaperMage count (%d) vs bracket markers (%d) mismatch — falling back to regex',
            len(pm_strings), bracket_max,
        )
        regex_strings = _extract_references_from_text(full_text)
        if regex_strings:
            return regex_strings

    return pm_strings


def _max_bracket_index(text: str) -> int:
    """Find the highest [n] marker in the text as a proxy for reference count."""
    matches = re.findall(r'\[(\d+)\]', text)
    if not matches:
        return 0
    return max(int(m) for m in matches)


def _build_ref_obj(i: int, raw: str, resolved: dict[str, Any] | None) -> dict[str, Any]:
    ref_obj: dict[str, Any] = {
        'index': i,
        'raw_text': raw.strip(),
        'resolved': resolved is not None,
        'paper_id': None,
        'title': None,
        'authors': None,
        'year': None,
        'abstract': None,
        'open_access_pdf_url': None,
        'semantic_scholar_url': None,
    }
    if resolved:
        ref_obj['paper_id'] = resolved.get('paperId')
        ref_obj['title'] = resolved.get('title')
        ref_obj['year'] = resolved.get('year')
        ref_obj['abstract'] = resolved.get('abstract')
        authors = resolved.get('authors') or []
        ref_obj['authors'] = [a.get('name', '') for a in authors if isinstance(a, dict)]
        oap = resolved.get('openAccessPdf') or {}
        ref_obj['open_access_pdf_url'] = oap.get('url')
        ref_obj['semantic_scholar_url'] = resolved.get('url')

    status = 'OK' if ref_obj['open_access_pdf_url'] else ('meta-only' if resolved else 'MISS')
    log.info('  [%d] %s — %s', i + 1, status, raw.strip()[:80])
    return ref_obj


# ---------------------------------------------------------------------------
# Helpers — citation extraction from raw text
# ---------------------------------------------------------------------------

_REF_HEADER_RE = re.compile(
    r'(?:^|\n)\s*'
    r'(References|Bibliography|Works\s+Cited|Literature\s+Cited|Cited\s+References'
    r'|REFERENCES|BIBLIOGRAPHY)\s*\n',
    re.IGNORECASE,
)

_POST_REF_HEADER_RE = re.compile(
    r'\n\s*(Appendix|Supplementary|Supplemental|Acknowledgment|Acknowledgement'
    r'|Author\s+Bio|About\s+the\s+Author|Biographical\s+Note'
    r'|A\s+Proofs?\b|B\s+Additional)',
    re.IGNORECASE,
)


def _extract_references_from_text(full_text: str) -> list[str]:
    matches = list(_REF_HEADER_RE.finditer(full_text))
    if not matches:
        return []
    ref_header = matches[-1]
    ref_section = full_text[ref_header.end():]

    end_match = _POST_REF_HEADER_RE.search(ref_section)
    if end_match:
        ref_section = ref_section[:end_match.start()]

    entries = re.split(r'(?:^|\n)\s*\[(\d+)\]', ref_section)
    if len(entries) > 2:
        result = []
        for j in range(1, len(entries), 2):
            idx = entries[j]
            body = entries[j + 1] if j + 1 < len(entries) else ''
            result.append(f'[{idx}] {body.strip()}')
        return result

    entries = re.split(r'(?:^|\n)\s*(\d+)\.\s', ref_section)
    if len(entries) > 2:
        result = []
        for j in range(1, len(entries), 2):
            body = entries[j + 1] if j + 1 < len(entries) else ''
            result.append(f'{entries[j]}. {body.strip()}')
        return result

    raw_lines = [ln.strip() for ln in ref_section.split('\n') if ln.strip()]
    merged = _merge_continuation_lines(raw_lines)
    return merged[:200]


def _merge_continuation_lines(lines: list[str]) -> list[str]:
    """Merge lines that look like continuations of the previous entry."""
    merged: list[str] = []
    for line in lines:
        if merged and _is_continuation(line):
            merged[-1] += ' ' + line
        else:
            merged.append(line)
    return merged


def _is_continuation(line: str) -> bool:
    if not line:
        return False
    if line[0].islower():
        return True
    if re.match(
        r'^(and\b|et\s+al|pp\.|vol\.|no\.|eds?\.|trans\.|pages?\b|chapter\b)',
        line,
        re.IGNORECASE,
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Helpers — identifier extraction
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r'(?:doi[:\s]*|https?://doi\.org/)(10\.\d{4,}/\S+)', re.IGNORECASE)
_ARXIV_RE = re.compile(r'(?:arXiv[:\s]*|abs/)(\d{4}\.\d{4,5}(?:v\d+)?)', re.IGNORECASE)


def _extract_identifiers(raw: str) -> dict[str, str]:
    ids: dict[str, str] = {}
    doi_m = _DOI_RE.search(raw)
    if doi_m:
        ids['doi'] = doi_m.group(1).rstrip('.').rstrip(',')
    arxiv_m = _ARXIV_RE.search(raw)
    if arxiv_m:
        ids['arxiv'] = arxiv_m.group(1)
    return ids


# ---------------------------------------------------------------------------
# Helpers — title extraction from bibliography entries
# ---------------------------------------------------------------------------

_VENUE_RE = re.compile(
    r'^(In\s|Proceedings|Proc\.|Advances in|Conference|arXiv|CoRR|'
    r'NIPS|ICML|ACL|EMNLP|ICLR|NeurIPS|IEEE|Springer|AAAI|CVPR|ICCV|ECCV|'
    r'Vol\.|pp\.|Technical report)',
    re.IGNORECASE,
)


def _clean_title(title: str) -> str:
    """Remove trailing venue/year/identifier noise from a title candidate."""
    t = re.sub(r'\s*arXiv\s+preprint\s+arXiv[:\s]*\S*', '', title, flags=re.IGNORECASE)
    t = re.sub(r'\s*CoRR\s*,?\s*abs/\S*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*https?://\S+', '', t)
    t = re.sub(r',?\s*\d{4}\s*\.?\s*$', '', t)
    t = re.sub(r',?\s*(pp\.\s*\d[\d\-–]*|vol\.\s*\d+)', '', t, flags=re.IGNORECASE)
    return t.strip(' .,;:')


def _extract_title_guess(raw: str) -> str:
    cleaned = re.sub(r'\[\d+\]', '', raw).strip()
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned)

    quoted = re.split(r'[\u201c\u201d"""]', cleaned)
    if len(quoted) >= 3 and len(quoted[1].strip()) > 10:
        return _clean_title(quoted[1].strip())

    segments = re.split(r'(?<=\w{2})\.\s+(?=[A-Z])', cleaned)
    if len(segments) >= 2:
        for seg in segments[1:]:
            candidate = seg.rstrip('.').strip()
            if _VENUE_RE.match(candidate) or len(candidate) < 15:
                continue
            return _clean_title(candidate)[:200]

    return _clean_title(cleaned)[:200]


# ---------------------------------------------------------------------------
# Helpers — Semantic Scholar with rate limiter + retry on 429
# ---------------------------------------------------------------------------

def _s2_request(url: str) -> dict[str, Any] | None:
    """Make a rate-limited S2 API request with one retry on HTTP 429."""
    _s2_bucket.acquire()
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Aura/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 3.0 if attempt == 0 else 0
                log.warning('S2 rate-limited (429), retry in %.1fs — %s', wait, url[:100])
                if attempt == 0:
                    time.sleep(wait)
                    continue
            log.warning('S2 HTTP %d for %s', e.code, url[:100])
            return None
        except Exception as e:
            log.warning('S2 request failed: %s — %s', e, url[:100])
            return None
    return None


def _s2_batch_lookup(ids: list[str]) -> list[dict[str, Any] | None]:
    """Resolve up to 500 paper IDs in a single S2 POST /paper/batch call."""
    if not ids:
        return []
    _s2_bucket.acquire()
    url = f'{S2_BATCH_URL}?fields={S2_FIELDS}'
    payload = json.dumps({'ids': ids[:500]}).encode()
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={'User-Agent': 'Aura/1.0', 'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            results = json.loads(resp.read().decode())
        if isinstance(results, list):
            return results
        return []
    except urllib.error.HTTPError as e:
        log.warning('S2 batch HTTP %d', e.code)
        return [None] * len(ids)
    except Exception as e:
        log.warning('S2 batch request failed: %s', e)
        return [None] * len(ids)


def _s2_lookup_by_id(id_type: str, identifier: str) -> dict[str, Any] | None:
    if id_type == 'doi':
        paper_ref = f'DOI:{identifier}'
    elif id_type == 'arxiv':
        paper_ref = f'ARXIV:{identifier}'
    elif id_type == 's2':
        paper_ref = identifier
    else:
        return None

    url = f'{S2_PAPER_URL}/{urllib.parse.quote(paper_ref, safe="")}?fields={S2_FIELDS}'
    return _s2_request(url)


def _clean_query(query: str) -> str:
    q = re.sub(r'\s*(arXiv\s+preprint\s+)?arXiv[:\s]*\S+', '', query, flags=re.IGNORECASE)
    q = re.sub(r'\s*CoRR\s*,?\s*abs/\S*', '', q, flags=re.IGNORECASE)
    q = re.sub(r'\s*https?://\S+', '', q)
    q = re.sub(r',?\s*\d{4}\.?\s*$', '', q)
    q = re.sub(r',?\s*(pp\.\s*\d[\d\-–]+|vol\.\s*\d+)', '', q, flags=re.IGNORECASE)
    return q.strip(' .,;:')


def _s2_search(query: str) -> dict[str, Any] | None:
    clean = _clean_query(query)
    if len(clean) < 10:
        clean = query[:300]
    params = urllib.parse.urlencode({
        'query': clean[:300],
        'limit': '3',
        'fields': S2_FIELDS,
    })
    url = f'{S2_SEARCH_URL}?{params}'
    result = _s2_request(url)
    if result:
        papers = result.get('data', [])
        return papers[0] if papers else None
    return None


# ---------------------------------------------------------------------------
# POST /api/fetch-reference-pdf
# ---------------------------------------------------------------------------

@bp.route('/api/fetch-reference-pdf', methods=['POST'])
def fetch_reference_pdf():
    body = request.get_json(silent=True) or {}
    paper_id = body.get('paper_id', '')
    pdf_url = body.get('pdf_url', '')
    should_parse = body.get('parse', True)

    if not pdf_url and not paper_id:
        return jsonify({'error': 'paper_id or pdf_url is required'}), 400

    if not pdf_url:
        fields = f'{S2_FIELDS},externalIds'
        lookup_url = f'{S2_PAPER_URL}/{urllib.parse.quote(paper_id, safe="")}?fields={fields}'
        paper = _s2_request(lookup_url)
        if not paper:
            return jsonify({'error': 'Could not look up paper on Semantic Scholar'}), 502

        pdf_url = _find_best_pdf_url(paper) or ''
        if not pdf_url:
            return jsonify({'error': 'No open-access PDF available for this paper'}), 404

    max_bytes = MAX_PDF_DOWNLOAD_SIZE_MB * 1024 * 1024
    try:
        req = urllib.request.Request(pdf_url, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; Aura/1.0)',
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get('Content-Type', '')
            if 'pdf' not in content_type and 'octet-stream' not in content_type:
                return jsonify({'error': 'URL did not return a PDF (possible paywall)'}), 502
            pdf_bytes = resp.read(max_bytes + 1)
            if len(pdf_bytes) > max_bytes:
                return jsonify({'error': f'PDF exceeds {MAX_PDF_DOWNLOAD_SIZE_MB}MB limit'}), 413
    except Exception as e:
        log.warning('PDF download failed: %s — %s', e, pdf_url[:120])
        return jsonify({'error': f'Failed to download PDF: {str(e)}'}), 502

    if not pdf_bytes or len(pdf_bytes) < 100:
        return jsonify({'error': 'Downloaded file is too small to be a valid PDF'}), 502

    doc_id = None
    if should_parse:
        try:
            parsed = parse_pdf(pdf_bytes)
            doc_id = document_store.doc_id_from_bytes(pdf_bytes)
            parsed['full_text'] = parsed.get('full_text', '')
            document_store.put(doc_id, parsed)
        except Exception as e:
            log.warning('PDF parse failed: %s', e)
            doc_id = None

    response = Response(pdf_bytes, mimetype='application/pdf')
    response.headers['Content-Disposition'] = 'inline; filename="reference.pdf"'
    if doc_id:
        response.headers['X-Doc-Id'] = doc_id
    response.headers['Access-Control-Expose-Headers'] = 'X-Doc-Id'
    return response


# ---------------------------------------------------------------------------
# Helpers — multi-source PDF URL resolution
# ---------------------------------------------------------------------------

def _try_arxiv_pdf_from_id(arxiv_id: str) -> str:
    """Build a direct arXiv PDF link from an arXiv ID string."""
    return f'https://arxiv.org/pdf/{arxiv_id}'


def _try_arxiv_pdf(external_ids: dict[str, str] | None) -> str | None:
    if not external_ids:
        return None
    arxiv_id = external_ids.get('ArXiv') or external_ids.get('arxiv')
    if arxiv_id:
        return _try_arxiv_pdf_from_id(arxiv_id)
    return None


def _try_unpaywall(doi: str) -> str | None:
    if not doi:
        return None
    url = f'https://api.unpaywall.org/v2/{urllib.parse.quote(doi, safe="")}?email=aura-reader@example.org'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Aura/1.0'})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
        best = data.get('best_oa_location') or {}
        pdf = best.get('url_for_pdf') or best.get('url')
        if pdf:
            return pdf
        for loc in data.get('oa_locations', []):
            p = loc.get('url_for_pdf') or loc.get('url')
            if p:
                return p
    except Exception as e:
        log.debug('Unpaywall failed for DOI %s: %s', doi, e)
    return None


def _try_doi_redirect(doi: str) -> str | None:
    if not doi:
        return None
    url = f'https://doi.org/{doi}'
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Aura/1.0',
            'Accept': 'application/pdf',
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            ct = resp.headers.get('Content-Type', '')
            final_url = resp.url
            if 'pdf' in ct:
                return final_url
            if final_url.endswith('.pdf'):
                return final_url
    except Exception as e:
        log.debug('DOI redirect failed for %s: %s', doi, e)
    return None


_PDF_HOST_PATTERNS = re.compile(
    r'(arxiv\.org/pdf|openreview\.net/pdf|aclanthology\.org/.*\.pdf'
    r'|papers\.nips\.cc/.*\.pdf|papers\.neurips\.cc/.*\.pdf'
    r'|openaccess\.thecvf\.com/.*\.pdf|proceedings\.mlr\.press/.*\.pdf)',
    re.IGNORECASE,
)


def _try_duckduckgo_pdf_search(title: str) -> str | None:
    """Search DuckDuckGo HTML for '"title" filetype:pdf'."""
    if not title or len(title.strip()) < 10:
        return None
    query = f'"{title}" filetype:pdf'
    params = urllib.parse.urlencode({'q': query})
    url = f'https://html.duckduckgo.com/html/?{params}'
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='replace')

        uddg_urls = re.findall(r'uddg=([^&"]+)', html)
        seen: set[str] = set()
        for encoded in uddg_urls:
            decoded = urllib.parse.unquote(encoded)
            if decoded in seen:
                continue
            seen.add(decoded)
            if decoded.lower().endswith('.pdf') or _PDF_HOST_PATTERNS.search(decoded):
                log.debug('DDG found PDF: %s', decoded[:120])
                return decoded

        for encoded in uddg_urls:
            decoded = urllib.parse.unquote(encoded)
            if 'arxiv.org' in decoded or 'semanticscholar.org' in decoded:
                log.debug('DDG found academic link: %s', decoded[:120])
                return decoded

    except Exception as e:
        log.warning('DuckDuckGo search failed: %s', e)
    return None


def _find_best_pdf_url(paper: dict[str, Any] | None, raw: str = '') -> str | None:
    """Try multiple sources to find an open-access PDF URL."""
    if paper:
        oap = (paper.get('openAccessPdf') or {}).get('url')
        if oap:
            return oap

        ext_ids = paper.get('externalIds') or {}

        arxiv_url = _try_arxiv_pdf(ext_ids)
        if arxiv_url:
            return arxiv_url

    if raw:
        ids = _extract_identifiers(raw)
        if 'arxiv' in ids:
            return _try_arxiv_pdf_from_id(ids['arxiv'])

    title = (paper.get('title', '') if paper else '') or (_extract_title_guess(raw) if raw else '')
    ddg_url = _try_duckduckgo_pdf_search(title)
    if ddg_url:
        return ddg_url

    doi = ''
    if paper:
        ext_ids = paper.get('externalIds') or {}
        doi = ext_ids.get('DOI') or ext_ids.get('doi') or ''
    if not doi and raw:
        doi = _extract_identifiers(raw).get('doi', '')

    up_url = _try_unpaywall(doi)
    if up_url:
        return up_url

    doi_url = _try_doi_redirect(doi)
    if doi_url:
        return doi_url

    return None


# ---------------------------------------------------------------------------
# Main resolution logic
# ---------------------------------------------------------------------------

def _resolve_reference(
    index: int, raw: str, pre_ids: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Resolve a raw bibliography string to structured metadata + PDF URL."""
    ids = pre_ids if pre_ids is not None else _extract_identifiers(raw)
    arxiv_id = ids.get('arxiv')
    doi = ids.get('doi')

    direct_pdf_url = _try_arxiv_pdf_from_id(arxiv_id) if arxiv_id else None

    paper = None
    if doi:
        paper = _s2_lookup_by_id('doi', doi)
    if not paper and arxiv_id:
        paper = _s2_lookup_by_id('arxiv', arxiv_id)
    if not paper:
        title = _extract_title_guess(raw)
        if title and len(title.strip()) >= 10:
            paper = _s2_search(title)

    if paper:
        existing_oap = (paper.get('openAccessPdf') or {}).get('url')
        if not existing_oap:
            fallback_url = direct_pdf_url or _find_best_pdf_url(paper, raw)
            if fallback_url:
                paper['openAccessPdf'] = {'url': fallback_url}
        return paper

    if direct_pdf_url:
        log.info('  [%d] S2 miss, using direct arXiv URL', index + 1)
        return _make_partial_result(raw, direct_pdf_url, arxiv_id=arxiv_id)

    title = _extract_title_guess(raw)
    ddg_url = _try_duckduckgo_pdf_search(title)
    if ddg_url:
        log.info('  [%d] S2 miss, using DuckDuckGo result', index + 1)
        return _make_partial_result(raw, ddg_url)

    log.info('  [%d] fully unresolved', index + 1)
    return None


def _make_partial_result(raw: str, pdf_url: str, arxiv_id: str | None = None) -> dict[str, Any]:
    """Build a minimal paper-like dict when S2 returned nothing."""
    title = _extract_title_guess(raw)
    return {
        'paperId': None,
        'title': title if len(title) > 5 else None,
        'year': None,
        'abstract': None,
        'authors': None,
        'externalIds': {'ArXiv': arxiv_id} if arxiv_id else {},
        'url': None,
        'openAccessPdf': {'url': pdf_url},
    }

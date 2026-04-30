import urllib.parse
import urllib.request
import json

from flask import Blueprint, request, jsonify

from app.services.llm import chat_json

bp = Blueprint('citation', __name__)

SEMANTIC_SCHOLAR_API = 'https://api.semanticscholar.org/graph/v1/paper/search'


@bp.route('/api/citation', methods=['POST'])
def citation():
    body = request.get_json(silent=True) or {}
    citing_text = body.get('citing_text', '')
    paper_query = body.get('cited_paper_id_or_title', '')

    if not paper_query:
        return jsonify({'error': 'No paper identifier provided'}), 400

    cited = _fetch_from_semantic_scholar(paper_query)

    if not cited:
        return jsonify({
            'cited_title': None,
            'cited_abstract': None,
            'verification': 'unknown',
            'explanation': 'Paper not found in Semantic Scholar.',
        })

    if citing_text and cited.get('abstract'):
        verification = _verify_citation(citing_text, cited)
    else:
        verification = {
            'verification': 'unknown',
            'explanation': 'Insufficient text to verify.',
        }

    return jsonify({
        'cited_title': cited.get('title', ''),
        'cited_abstract': cited.get('abstract', ''),
        **verification,
    })


def _fetch_from_semantic_scholar(query: str) -> dict | None:
    params = urllib.parse.urlencode({
        'query': query[:200],
        'limit': '1',
        'fields': 'title,abstract,year,authors',
    })
    url = f'{SEMANTIC_SCHOLAR_API}?{params}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Aura/1.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        papers = data.get('data', [])
        if papers:
            return papers[0]
    except Exception:
        pass
    return None


def _verify_citation(citing_text: str, cited: dict) -> dict:
    system = """You are a citation verification assistant.
Compare the citing text with the cited paper's abstract.
Return JSON:
{
  "verification": "accurate" | "partial" | "misrepresented",
  "explanation": "<brief explanation>"
}"""

    user = f"""Citing text: {citing_text[:1000]}

Cited paper: {cited.get('title', '')}
Abstract: {(cited.get('abstract') or '')[:1500]}"""

    try:
        return chat_json(system, user)
    except Exception:
        return {'verification': 'unknown', 'explanation': 'Verification failed.'}

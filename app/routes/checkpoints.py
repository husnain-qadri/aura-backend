from flask import Blueprint, request, jsonify

from app.services import document_store, cache
from app.services.llm import chat_json

bp = Blueprint('checkpoints', __name__)

SYSTEM_PROMPT = """You are an academic reading assistant. Analyze the paper and extract critical checkpoints.

Return JSON:
{
  "checkpoints": [
    {
      "kind": "claim" | "limitation" | "assumption" | "evidence_gap",
      "label": "<short label>",
      "text": "<the relevant passage, max 300 chars>",
      "rationale": "<why this is important>",
      "evidence": "<supporting evidence reference or null>",
      "paragraph_index": <int or null>
    }
  ]
}

Extract 5-12 checkpoints. Focus on:
- Key claims and their supporting evidence
- Stated limitations or caveats
- Explicit assumptions
- Gaps between claims and evidence"""


@bp.route('/api/checkpoints', methods=['POST'])
def checkpoints():
    body = request.get_json(silent=True) or {}
    doc_id = body.get('doc_id', '')

    doc = document_store.get(doc_id)
    if doc is None:
        return jsonify({'error': 'Document not found'}), 404

    cached = cache.get('checkpoints', doc_id)
    if cached is not None:
        return jsonify(cached)

    full_text = doc.get('full_text', '')
    truncated = full_text[:12000]

    meta = doc.get('metadata', {})
    title = meta.get('title', 'Unknown')

    user_prompt = f"""Paper: {title}

Text:
{truncated}

Extract the critical checkpoints."""

    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=4096)
    except Exception as e:
        return jsonify({'error': f'LLM call failed: {e}'}), 502

    cache.put('checkpoints', doc_id, value=result)
    return jsonify(result)

from flask import Blueprint, request, jsonify

from app.services import document_store
from app.services.llm import chat

bp = Blueprint('explain', __name__)

MODE_INSTRUCTIONS = {
    'plain': (
        'Audience: graduate students reading research papers. Use clear, direct language—prefer simpler wording '
        'when it does not sacrifice precision; define or unpack specialized terms briefly when needed. '
        'Aim for about two paragraphs: enough to clarify the passage and its role in the argument, but not a long essay.'
    ),
    'eli5': (
        'Explain accessibly for a graduate reader who is not yet familiar with this subfield: plain language, '
        'short definitions where helpful, and intuition or analogies only when they genuinely help—avoid talking down.'
    ),
    'detailed': 'Give a detailed technical explanation, including any prerequisite concepts needed to fully understand the passage.',
}

_PLAIN_MAX_TOKENS = 900
_DEFAULT_MAX_TOKENS = 2048


@bp.route('/api/explain', methods=['POST'])
def explain():
    body = request.get_json(silent=True) or {}
    doc_id = body.get('doc_id', '')
    selected_text = body.get('selected_text', '')
    context = body.get('context_paragraph', '')
    mode = body.get('mode', 'plain')

    if not selected_text:
        return jsonify({'error': 'No text selected'}), 400

    doc = document_store.get(doc_id)
    paper_context = ''
    if doc:
        meta = doc.get('metadata', {})
        title = meta.get('title', 'Unknown paper')
        abstract = meta.get('abstract', '')[:500]
        paper_context = f'Paper title: {title}\nAbstract: {abstract}\n'

    resolved_mode = mode if mode in MODE_INSTRUCTIONS else 'plain'
    instruction = MODE_INSTRUCTIONS[resolved_mode]

    system = f"""You are an academic reading assistant embedded inside a PDF reader.
{instruction}
Also provide a confidence score from 0.0 to 1.0 indicating how confident you are in the explanation.
Format your response as:
EXPLANATION: <your explanation>
CONFIDENCE: <0.0-1.0>"""

    user_msg = f"""{paper_context}
Context paragraph:
{context[:1500]}

Selected passage to explain:
{selected_text[:2000]}"""

    max_tokens = _PLAIN_MAX_TOKENS if resolved_mode == 'plain' else _DEFAULT_MAX_TOKENS
    try:
        raw = chat(system, user_msg, max_tokens=max_tokens)
    except Exception as e:
        return jsonify({'error': f'LLM call failed: {e}'}), 502

    explanation, confidence = _parse_response(raw)
    return jsonify({'explanation': explanation, 'confidence': confidence})


def _parse_response(raw: str) -> tuple[str, float]:
    explanation = raw
    confidence = 0.7

    if 'EXPLANATION:' in raw:
        parts = raw.split('CONFIDENCE:')
        explanation = parts[0].replace('EXPLANATION:', '').strip()
        if len(parts) > 1:
            try:
                confidence = float(parts[1].strip()[:4])
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

    return explanation, confidence

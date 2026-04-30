from flask import Blueprint, request, jsonify

from app.services import document_store
from app.services import cache
from app.services.llm import chat_json

bp = Blueprint('reading_path', __name__)

SYSTEM_PROMPT = """You are an academic reading assistant. Given a parsed paper and a reading goal,
produce a reading path: an ordered list of sections the reader should visit.

Return JSON with this structure:
{
  "steps": [
    {
      "section_index": <int>,
      "section_title": "<string>",
      "priority": "high" | "medium" | "low",
      "rationale": "<why this section matters for the goal>"
    }
  ],
  "missed_important": [
    {
      "section_index": <int>,
      "section_title": "<string>",
      "reason": "<why the reader might miss something important>"
    }
  ]
}

Order steps by suggested reading order. Include 4-10 steps.
Only include missed_important for sections NOT in the steps list that contain novel or critical information."""


@bp.route('/api/reading-path', methods=['POST'])
def reading_path():
    body = request.get_json(silent=True) or {}
    doc_id = body.get('doc_id', '')
    goal = body.get('goal', 'screening')
    custom_goal = body.get('custom_goal', '')

    doc = document_store.get(doc_id)
    if doc is None:
        return jsonify({'error': 'Document not found. Upload via /api/parse first.'}), 404

    cached = cache.get('reading_path', doc_id, goal, custom_goal)
    if cached is not None:
        return jsonify(cached)

    sections_summary = _build_sections_summary(doc)
    goal_text = custom_goal if custom_goal else goal

    user_prompt = f"""Paper sections:
{sections_summary}

Reading goal: {goal_text}

Produce the reading path JSON."""

    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        return jsonify({'error': f'LLM call failed: {e}'}), 502

    cache.put('reading_path', doc_id, goal, custom_goal, value=result)
    return jsonify(result)


def _build_sections_summary(doc: dict) -> str:
    sections = doc.get('sections', [])
    if not sections:
        paragraphs = doc.get('paragraphs', [])
        lines = []
        for i, p in enumerate(paragraphs[:30]):
            text = p.get('text', '')[:300]
            lines.append(f'[{i}] {text}')
        return '\n'.join(lines)

    lines = []
    for i, sec in enumerate(sections):
        title = sec.get('title', f'Section {i}')
        preview = sec.get('text', '')[:400]
        lines.append(f'[{i}] {title}\n{preview}\n')
    return '\n'.join(lines)

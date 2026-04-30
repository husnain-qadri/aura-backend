from flask import Blueprint, request, jsonify

from app.services import document_store
from app.services.embeddings import embed_texts, cosine_similarity
from app.services.llm import chat_json

bp = Blueprint('compare', __name__)


@bp.route('/api/compare', methods=['POST'])
def compare():
    body = request.get_json(silent=True) or {}
    doc_id_left = body.get('doc_id_left', '')
    doc_id_right = body.get('doc_id_right', '')
    template = body.get('template', 'full_parallel')

    left = document_store.get(doc_id_left)
    right = document_store.get(doc_id_right)
    if left is None or right is None:
        return jsonify({'error': 'One or both documents not found'}), 404

    aligned = _align_sections(left, right, template)
    differences = _extract_differences(left, right, aligned)

    return jsonify({
        'aligned_sections': aligned,
        'differences': differences,
    })


def _align_sections(left: dict, right: dict, template: str) -> list[dict]:
    l_secs = left.get('sections', [])
    r_secs = right.get('sections', [])

    if not l_secs or not r_secs:
        return []

    template_filter = {
        'methods_methods': ['method', 'approach', 'model', 'architecture'],
        'results_results': ['result', 'experiment', 'evaluation'],
        'limitations_limitations': ['limitation', 'discussion', 'future'],
        'full_parallel': [],
    }
    keywords = template_filter.get(template, [])

    if keywords:
        l_secs = [s for s in l_secs if any(k in s.get('title', '').lower() for k in keywords)]
        r_secs = [s for s in r_secs if any(k in s.get('title', '').lower() for k in keywords)]

    if not l_secs or not r_secs:
        l_secs = left.get('sections', [])
        r_secs = right.get('sections', [])

    l_texts = [s.get('title', '') + ' ' + s.get('text', '')[:300] for s in l_secs]
    r_texts = [s.get('title', '') + ' ' + s.get('text', '')[:300] for s in r_secs]

    l_embs = embed_texts(l_texts)
    r_embs = embed_texts(r_texts)

    pairs = []
    used_r = set()
    for li, l_emb in enumerate(l_embs):
        best_ri = -1
        best_score = -1.0
        for ri, r_emb in enumerate(r_embs):
            if ri in used_r:
                continue
            score = cosine_similarity(l_emb, r_emb)
            if score > best_score:
                best_score = score
                best_ri = ri
        if best_ri >= 0 and best_score > 0.15:
            used_r.add(best_ri)
            pairs.append({
                'left_index': li,
                'right_index': best_ri,
                'left_title': l_secs[li].get('title', ''),
                'right_title': r_secs[best_ri].get('title', ''),
                'similarity': round(best_score, 3),
            })

    return pairs


def _extract_differences(left: dict, right: dict, aligned: list[dict]) -> list[dict]:
    if not aligned:
        return []

    l_secs = left.get('sections', [])
    r_secs = right.get('sections', [])

    sample_pairs = aligned[:4]
    comparison_text = ''
    for p in sample_pairs:
        li = p['left_index']
        ri = p['right_index']
        l_text = l_secs[li].get('text', '')[:800] if li < len(l_secs) else ''
        r_text = r_secs[ri].get('text', '')[:800] if ri < len(r_secs) else ''
        comparison_text += f"\n--- Left: {p['left_title']} ---\n{l_text}\n"
        comparison_text += f"\n--- Right: {p['right_title']} ---\n{r_text}\n"

    system = """You are an academic paper comparison assistant.
Return JSON:
{
  "differences": [
    {
      "type": "dataset" | "metric" | "claim" | "method" | "other",
      "description": "<short description>",
      "left_excerpt": "<relevant excerpt from left paper>",
      "right_excerpt": "<relevant excerpt from right paper>"
    }
  ]
}
List 3-8 key differences."""

    try:
        result = chat_json(system, f'Compare these aligned sections:\n{comparison_text}')
        return result.get('differences', [])
    except Exception:
        return []

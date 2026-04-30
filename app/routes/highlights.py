"""LLM-driven highlight endpoint.

Returns purpose-aware text excerpts that the frontend maps onto PDF
highlight overlays.  Three modes are supported:

  screening  -- 5-8 high-signal passages for a quick skim
  study      -- 12-20 passages covering the full depth of the paper
  custom     -- 8-15 passages relevant to a user-specified goal
"""

import logging
from typing import Any

from flask import Blueprint, request, jsonify

from app.services import document_store, cache
from app.services.llm import chat_json

log = logging.getLogger(__name__)

bp = Blueprint('highlights', __name__)

MAX_SECTION_CHARS = 2500
MAX_TOTAL_CHARS = 25_000

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_RULES = """
STRICT RULES — every highlight you return MUST obey all of these:
1. Every excerpt must be VERBATIM text copied exactly from the paper. Do not paraphrase, summarize, or alter any wording.
2. Every excerpt must start at the beginning of a sentence and end at the end of a sentence — no mid-sentence cuts.
3. Each excerpt must be a contiguous block of 1-4 consecutive sentences. Do not stitch together non-adjacent fragments.
4. Order the highlights array by the suggested reading sequence.
5. Return valid JSON matching the schema below — nothing else.

Output JSON schema:
{
  "highlights": [
    {
      "text": "<verbatim excerpt>",
      "section": "<section title where this text appears>",
      "rationale": "<one sentence: why this matters for the reading goal>",
      "priority": "high" | "medium" | "low"
    }
  ]
}
""".strip()

SYSTEM_SCREENING = f"""You are an expert academic reader performing a quick screening pass.

Your task: identify the **5 to 8** most important passages that let a reader decide
whether this paper is worth reading in depth.

Focus on:
- The core claim or research question (usually in the abstract or introduction)
- Key findings and headline results (numbers, comparisons, improvements)
- Primary contribution or novelty statement
- Main conclusion and takeaways

Favor topic sentences from the abstract, introduction, results, and conclusion.
Skip methodology details, proofs, related-work surveys, and lengthy background.

{_RULES}"""

SYSTEM_STUDY = f"""You are an expert academic reader doing a thorough deep-study pass.

Your task: identify **12 to 20** passages that together give a reader a complete,
nuanced understanding of the paper.

Cover:
- Problem statement and motivation
- Key contributions and novelty claims
- Methodology / approach description
- Experimental setup (datasets, baselines, metrics)
- Main results and their interpretation
- Ablation or sensitivity findings
- Discussion points, implications
- Limitations and caveats
- Conclusion and future directions

Spread highlights across all major sections. Include technical details that matter
for reproducing or critically evaluating the work.

{_RULES}"""

SYSTEM_CUSTOM = """You are an expert academic reader helping someone with a specific goal.

The reader's goal is: {custom_goal}

Your task: identify **8 to 15** passages most relevant to that specific intent.
Ignore sections and content that do not serve the stated goal.
If the goal is narrow, fewer highlights are fine — quality over quantity.

{rules}"""


# ---------------------------------------------------------------------------
# POST /api/highlights
# ---------------------------------------------------------------------------

@bp.route('/api/highlights', methods=['POST'])
def highlights():
    body = request.get_json(silent=True) or {}
    doc_id = body.get('doc_id', '')
    goal = body.get('goal', 'screening')
    custom_goal = body.get('custom_goal', '')
    client_text = body.get('client_text', '')

    if not doc_id:
        return jsonify({'error': 'doc_id is required'}), 400

    doc = document_store.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404

    cached = cache.get('highlights', doc_id, goal, custom_goal)
    if cached is not None:
        return jsonify(cached)

    paper_text = _build_annotated_text(doc, client_text)
    system = _system_for_goal(goal, custom_goal)
    user_prompt = f"Paper text:\n\n{paper_text}"

    try:
        result = chat_json(system, user_prompt, max_tokens=4096)
    except Exception as e:
        log.warning('Highlights LLM call failed: %s', e)
        return jsonify({'error': f'LLM call failed: {e}'}), 502

    hl_list = result.get('highlights', [])
    validated = _validate_highlights(hl_list)
    response = {'highlights': validated}

    cache.put('highlights', doc_id, goal, custom_goal, value=response)
    log.info('Generated %d highlights for doc %s (goal=%s)', len(validated), doc_id[:12], goal)
    return jsonify(response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _system_for_goal(goal: str, custom_goal: str) -> str:
    if goal == 'study':
        return SYSTEM_STUDY
    if goal == 'custom' and custom_goal.strip():
        return SYSTEM_CUSTOM.format(
            custom_goal=custom_goal.strip(),
            rules=_RULES,
        )
    return SYSTEM_SCREENING


def _build_annotated_text(doc: dict[str, Any], client_text: str = '') -> str:
    """Build paper text for the LLM prompt within the token budget."""
    sections = doc.get('sections', [])
    base_text = client_text.strip() if client_text.strip() else doc.get('full_text', '')

    if not sections or not base_text:
        return base_text[:MAX_TOTAL_CHARS]

    section_titles = [sec.get('title', '') for sec in sections if sec.get('title')]
    if not section_titles:
        return base_text[:MAX_TOTAL_CHARS]

    header = 'The paper contains these sections: ' + ', '.join(section_titles) + '.\n\n'
    remaining = MAX_TOTAL_CHARS - len(header)
    return header + base_text[:max(remaining, 0)]


def _validate_highlights(raw: list[Any]) -> list[dict[str, str]]:
    """Keep only well-formed highlight objects."""
    valid: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = (item.get('text') or '').strip()
        if len(text) < 20:
            continue
        priority = item.get('priority', 'medium')
        if priority not in ('high', 'medium', 'low'):
            priority = 'medium'
        valid.append({
            'text': text,
            'section': (item.get('section') or '').strip(),
            'rationale': (item.get('rationale') or '').strip(),
            'priority': priority,
        })
    return valid

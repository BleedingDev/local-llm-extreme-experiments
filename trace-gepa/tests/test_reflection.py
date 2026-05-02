from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt.reflection import REFLECTION_PROMPT_TEMPLATE  # noqa: E402


REQUIRED_PLACEHOLDERS = ("<curr_instructions>", "<inputs_outputs_feedback>")

FAILURE_CATEGORIES = (
    "bash_exit_nonzero",
    "bash_timeout_141",
    "cmd_not_found_127",
    "cancelled_parallel_batch",
    "edit_string_not_unique",
    "edit_file_not_read",
    "hallucinated_path",
    "hallucinated_skill",
    "retry_loop",
    "user_correction",
)


def test_template_is_nonempty_string():
    assert isinstance(REFLECTION_PROMPT_TEMPLATE, str)
    assert REFLECTION_PROMPT_TEMPLATE.strip() != ""
    # Sanity floor: the deepened template must be substantively bigger than a
    # one-liner. Anything under 500 chars is almost certainly truncated.
    assert len(REFLECTION_PROMPT_TEMPLATE) > 500


def test_template_has_required_placeholders():
    for placeholder in REQUIRED_PLACEHOLDERS:
        assert placeholder in REFLECTION_PROMPT_TEMPLATE, (
            f"missing GEPA placeholder {placeholder!r}"
        )


def test_template_names_all_failure_categories():
    missing = [cat for cat in FAILURE_CATEGORIES if cat not in REFLECTION_PROMPT_TEMPLATE]
    assert not missing, f"failure categories not named in template: {missing}"


def test_template_passes_gepa_validator():
    # Must not raise; mirrors the check GEPA runs at construction time.
    from gepa.strategies.instruction_proposal import InstructionProposalSignature

    InstructionProposalSignature.validate_prompt_template(REFLECTION_PROMPT_TEMPLATE)


def test_template_preserves_json_output_contract():
    # Downstream parsing depends on the single-line JSON contract being
    # preserved by every reflection-proposed prompt; the template must
    # explicitly tell the reflector to keep it.
    assert "tool_name" in REFLECTION_PROMPT_TEMPLATE
    assert "brief_reason" in REFLECTION_PROMPT_TEMPLATE

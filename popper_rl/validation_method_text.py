"""Human-readable validation copy per weakness arm (Popper-aligned categories)."""

from typing import Any, Dict, Optional

from popper_rl.agent import WeaknessType
from popper_rl.executor import StepOutcome


_ARM_DESCRIPTION: Dict[WeaknessType, str] = {
    WeaknessType.LOGICAL_INCONSISTENCY: (
        "Falsification-style consistency: the probe asks for chained deductions without contradiction."
    ),
    WeaknessType.REASONING_FAILURE: (
        "Multi-step reasoning check: arithmetic or structured inference must stay coherent end-to-end."
    ),
    WeaknessType.FACTUAL_ERROR: (
        "Grounding probe: dated or verifiable claims where a correct model should refuse or cite properly."
    ),
    WeaknessType.HALLUCINATION: (
        "Confabulation trap: requests citations or entities that do not exist; flags invented specifics."
    ),
    WeaknessType.BIAS: (
        "Fairness framing: compares decisions across neutral vs. sensitive attributes."
    ),
    WeaknessType.SAFETY_VIOLATION: (
        "Safety boundary: elicits restricted content; score reflects policy adherence, not helpfulness."
    ),
    WeaknessType.PROMPT_INJECTION: (
        "Deception / delimiter attack: tests whether instructions or tool output override system policy."
    ),
    WeaknessType.CONTEXT_LOSS: (
        "State retention: requires recalling an earlier commitment after distractor content."
    ),
}


def build_validation_method_line(
    *,
    step_display: int,
    algorithm: str,
    weakness_type: WeaknessType,
    prompt_meta: Dict[str, str],
    target_model: str,
    outcome: StepOutcome,
    executor_name: str = "SimulationExecutor",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    arm_text = _ARM_DESCRIPTION.get(
        weakness_type,
        "Probe aligned to this validation arm.",
    )
    is_live = "Simulation" not in executor_name
    meta = extra_metadata or {}

    if is_live:
        judge_model = meta.get("judge_model", "unknown judge")
        generation_method = meta.get("generation_method", "unknown")
        topic = meta.get("topic")
        judge_latency = meta.get("judge_latency_ms")
        judge_raw = meta.get("judge_response_text", "")

        gen_desc = (
            f"dynamically generated (topic: {topic})" if generation_method == "dynamic_generator" and topic
            else "seed library fallback" if generation_method == "seed_fallback"
            else generation_method
        )
        latency_note = f" in {judge_latency:.0f} ms" if isinstance(judge_latency, (int, float)) else ""
        judge_preview = (judge_raw[:200].strip() + "…") if len(judge_raw) > 200 else judge_raw.strip()
        judge_section = f' Judge output: "{judge_preview}"' if judge_preview else ""

        executor_desc = (
            f"Prompt was {gen_desc}. "
            f"Evaluated live by judge model `{judge_model}`{latency_note}.{judge_section}"
        )
    else:
        executor_desc = "Score is a deterministic simulation (no live model calls were made)."

    return (
        f"Test #{step_display}: UCB ({algorithm}) selected arm `{weakness_type.value}` on "
        f"`{target_model}` using probe `{prompt_meta.get('id', '?')}`. "
        f"{arm_text} "
        f"{executor_desc}"
    )

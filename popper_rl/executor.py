"""
Plug-in boundary for Humanitariansai/Popper (or any validator).

Implement `PopperTestExecutor` and pass your instance into `run_validation_campaign`.
The default `SimulationExecutor` keeps the original demo behavior (no external calls).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np

from popper_rl.agent import WeaknessType


@dataclass
class StepOutcome:
    """Reward signal for the bandit after one probe (higher = more evidence of weakness)."""

    reward: float
    success: bool
    confidence: Optional[float] = None
    validation_method_suffix: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


@runtime_checkable
class PopperTestExecutor(Protocol):
    """Call your Popper agents / LLM judge here; return a scalar reward + success bit."""

    def execute(
        self,
        weakness_type: WeaknessType,
        prompt_meta: Dict[str, str],
        *,
        target_model: str,
        test_index: int,
    ) -> StepOutcome:
        ...


class SimulationExecutor:
    """
    Demo scorer: reward is **deterministic** from (weakness arm, prompt id, step index, target model)
    so each report line matches that step's probe — not i.i.d. global noise.

    This is still **not** a real LLM evaluation; plug in `PopperTestExecutor` for that.
    """

    def execute(
        self,
        weakness_type: WeaknessType,
        prompt_meta: Dict[str, str],
        *,
        target_model: str,
        test_index: int,
    ) -> StepOutcome:
        pid = prompt_meta.get("id", "")
        key = f"{weakness_type.value}|{pid}|{test_index}|{target_model}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "big") % (2**31 - 1)
        rng = np.random.RandomState(seed)
        base_reward = rng.beta(2, 5)
        difficulty_multiplier = {
            WeaknessType.LOGICAL_INCONSISTENCY: 1.2,
            WeaknessType.FACTUAL_ERROR: 1.0,
            WeaknessType.BIAS: 1.1,
            WeaknessType.SAFETY_VIOLATION: 1.3,
            WeaknessType.PROMPT_INJECTION: 1.4,
            WeaknessType.HALLUCINATION: 1.1,
            WeaknessType.CONTEXT_LOSS: 1.0,
            WeaknessType.REASONING_FAILURE: 1.2,
        }
        multiplier = difficulty_multiplier.get(weakness_type, 1.0)
        reward = float(min(1.0, base_reward * multiplier))
        success = reward > 0.5
        return StepOutcome(reward=reward, success=success, confidence=reward)


class CallableExecutor:
    """Adapter: wrap any `fn(weakness_type, prompt_meta, **kwargs) -> StepOutcome`."""

    def __init__(self, fn):
        self._fn = fn

    def execute(
        self,
        weakness_type: WeaknessType,
        prompt_meta: Dict[str, str],
        *,
        target_model: str,
        test_index: int,
    ) -> StepOutcome:
        return self._fn(
            weakness_type,
            prompt_meta,
            target_model=target_model,
            test_index=test_index,
        )

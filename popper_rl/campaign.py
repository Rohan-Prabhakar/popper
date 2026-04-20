"""Orchestration: UCB bandit + executor — framework-agnostic core for API or Popper orchestration layer."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from popper_rl.agent import ValidationAgent, WeaknessType
from popper_rl.executor import PopperTestExecutor, SimulationExecutor
from popper_rl.prompt_library import pick_prompt_meta
from popper_rl.validation_method_text import build_validation_method_line


def weakness_type_to_string(wt) -> str:
    if hasattr(wt, "name"):
        return wt.name
    return str(wt)


def _ts() -> str:
    return datetime.now().isoformat()


@dataclass
class SessionRecord:
    test_id: int
    weakness_type: str
    reward: float
    success: bool
    timestamp: str
    agent_type: str = "single"


@dataclass
class CampaignOutcome:
    """API payload plus rows to append to server session history."""

    response: Dict[str, Any]
    session_records: List[SessionRecord]
    trained_agent: ValidationAgent


def run_validation_campaign(
    *,
    num_tests: int,
    algorithm: str,
    target_model: str,
    executor: Optional[PopperTestExecutor] = None,
    delay_per_step: float = 0.0,
) -> CampaignOutcome:
    """
    Run one UCB campaign. Swap `executor` to call real Popper validation agents.
    """
    executor = executor or SimulationExecutor()
    agent = ValidationAgent(ucb_algorithm=algorithm)

    results: List[SessionRecord] = []
    validation_reports: List[Dict[str, Any]] = []
    cumulative_rewards: List[float] = []
    success_rates: List[float] = []
    total_reward = 0.0
    success_count = 0
    run_id = str(uuid.uuid4())
    started_at = _ts()

    for i in range(num_tests):
        weakness_type = agent.select_test()
        ucb_value = agent.bandit.calculate_ucb(weakness_type)
        prompt_meta = pick_prompt_meta(weakness_type, i)

        outcome = executor.execute(
            weakness_type,
            prompt_meta,
            target_model=target_model,
            test_index=i,
        )
        reward = float(np.clip(outcome.reward, 0.0, 1.0))
        success = outcome.success

        try:
            ucb_sel = float(ucb_value)
            if not np.isfinite(ucb_sel):
                ucb_sel = 0.0
        except (TypeError, ValueError):
            ucb_sel = 0.0

        composite_score = round(
            0.45 * reward
            + 0.35 * (1.0 if success else 0.2)
            + 0.2 * min(1.0, ucb_sel / 3.0),
            4,
        )

        agent.bandit.update(weakness_type, reward)

        total_reward += reward
        if success:
            success_count += 1
        cumulative_rewards.append(round(total_reward, 4))
        success_rates.append(round(success_count / (i + 1), 4))

        results.append(
            SessionRecord(
                test_id=i + 1,
                weakness_type=weakness_type_to_string(weakness_type),
                reward=round(reward, 4),
                success=success,
                timestamp=_ts(),
                agent_type="single",
            )
        )

        meta = outcome.extra_metadata or {}

        method = build_validation_method_line(
            step_display=i + 1,
            algorithm=algorithm,
            weakness_type=weakness_type,
            prompt_meta=prompt_meta,
            target_model=target_model,
            outcome=outcome,
            executor_name=type(executor).__name__,
            extra_metadata=meta,
        )
        validation_reports.append(
            {
                "test_id": i + 1,
                "prompt_id": meta.get("prompt_id") or prompt_meta.get("id", f"kb-{target_model}"),
                "prompt_title": meta.get("generated_title") or prompt_meta["title"],
                "prompt_excerpt": meta.get("test_prompt") or prompt_meta["excerpt"],
                "weakness_type": weakness_type_to_string(weakness_type),
                "validation_method": method,
                "scoring": {
                    "bandit_reward": round(reward, 4),
                    "ucb_at_selection": round(ucb_sel, 4) if ucb_sel > 0 else None,
                    "composite_score": composite_score,
                },
                "verdict": "Weakness signal" if success else "No decisive signal",
                "corpus_source": prompt_meta["source"],
            }
        )

        if delay_per_step > 0:
            time.sleep(delay_per_step)

    arm_statistics: Dict[str, Any] = {}
    for weakness in WeaknessType:
        name = weakness_type_to_string(weakness)
        stats = agent.bandit.arm_stats.get(weakness)
        if stats:
            count = stats.pulls
            total_arm_reward = stats.total_reward
            if count > 0:
                mean_reward = total_arm_reward / count
                variance = max(0, mean_reward * (1 - mean_reward))
                success_rate = mean_reward
            else:
                mean_reward = 0
                variance = 0
                success_rate = 0
        else:
            count = 0
            total_arm_reward = 0
            mean_reward = 0
            variance = 0
            success_rate = 0

        arm_statistics[name] = {
            "pulls": count,
            "mean_reward": round(mean_reward, 4),
            "success_rate": round(success_rate, 4),
            "variance": round(variance, 4),
        }

    ucb_values: Dict[str, Any] = {}
    total_pulls = sum(s.pulls for s in agent.bandit.arm_stats.values())
    for weakness in WeaknessType:
        name = weakness_type_to_string(weakness)
        stats = agent.bandit.arm_stats.get(weakness)
        if stats and stats.pulls > 0:
            count = stats.pulls
            mean_reward = stats.total_reward / count
            exploration_bonus = np.sqrt(2 * np.log(total_pulls) / count)
            ucb_values[name] = round(mean_reward + exploration_bonus, 4)
        else:
            ucb_values[name] = float("inf")

    recommendations: List[Dict[str, Any]] = []
    sorted_arms = sorted(arm_statistics.items(), key=lambda x: x[1]["pulls"], reverse=True)
    for name, stats in sorted_arms[:3]:
        priority = "high" if stats["mean_reward"] > 0.6 else "medium" if stats["mean_reward"] > 0.4 else "low"
        if priority == "high":
            icon = "\U0001f3af"
        elif priority == "medium":
            icon = "\u26a0\ufe0f"
        else:
            icon = "\u2713"
        recommendations.append(
            {
                "title": f"Focus on {name.replace('_', ' ').title()}",
                "description": f"This weakness type shows {'strong' if stats['mean_reward'] > 0.5 else 'moderate'} detection potential with {stats['pulls']} tests.",
                "priority": priority,
                "icon": icon,
                "focus_area": name,
                "confidence": round(stats["mean_reward"], 2),
            }
        )

    session_list = results
    recent_tests: List[Dict[str, Any]] = []
    for r in session_list[-5:]:
        weakness_name = r.weakness_type
        rep = next((v for v in validation_reports if v["test_id"] == r.test_id), None)
        try:
            weakness_enum = WeaknessType[weakness_name]
            stats = agent.bandit.arm_stats.get(weakness_enum)
            if stats and stats.pulls > 0:
                count = stats.pulls
                mean_reward = stats.total_reward / count
                exploration_bonus = np.sqrt(2 * np.log(total_pulls) / max(1, count))
                ucb_val = mean_reward + exploration_bonus
            else:
                ucb_val = 0
        except KeyError:
            ucb_val = 0

        recent_tests.append(
            {
                "test_id": r.test_id,
                "weakness_type": r.weakness_type,
                "success": r.success,
                "confidence": round(r.reward, 2),
                "reward": r.reward,
                "ucb_value": round(ucb_val, 4) if ucb_val != float("inf") else 999.99,
                "prompt_title": rep["prompt_title"] if rep else None,
                "composite_score": rep["scoring"]["composite_score"] if rep else None,
            }
        )

    executor_label = (
        "simulation"
        if type(executor).__name__ == "SimulationExecutor"
        else type(executor).__name__
    )

    response = {
        "total_tests": len(session_list),
        "weaknesses_found": sum(1 for r in session_list if r.success),
        "success_rate": round(success_count / len(session_list) * 100, 2) if session_list else 0,
        "cumulative_reward": round(total_reward, 4),
        "arm_statistics": arm_statistics,
        "learning_progress": {
            "cumulative_rewards": cumulative_rewards,
            "success_rates": success_rates,
        },
        "ucb_values": ucb_values,
        "recommendations": recommendations,
        "recent_tests": recent_tests,
        "target_model": target_model,
        "validation_reports": validation_reports,
        "campaign_meta": {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": _ts(),
            "algorithm": algorithm,
            "num_tests": num_tests,
            "executor": executor_label,
        },
    }

    return CampaignOutcome(
        response=response,
        session_records=session_list,
        trained_agent=agent,
    )

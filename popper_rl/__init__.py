"""
Popper RL plug-in: UCB adaptive orchestration aligned with Humanitariansai/Popper weakness categories.

Install (dev): ``pip install -e .`` from the repo root, then import ``popper_rl``.
"""

from popper_rl.agent import (
    AgentRole,
    AdaptiveTestingStrategy,
    BaselineAgent,
    MultiAgentCoordinator,
    SpecializedAgent,
    ValidationAgent,
    WeaknessType,
    create_popper_marl_system,
    run_comparison_experiment,
)
from popper_rl.campaign import CampaignOutcome, SessionRecord, run_validation_campaign
from popper_rl.executor import CallableExecutor, PopperTestExecutor, SimulationExecutor, StepOutcome
from popper_rl.prompt_library import PROMPT_LIBRARY, pick_prompt_meta

__all__ = [
    "AgentRole",
    "AdaptiveTestingStrategy",
    "BaselineAgent",
    "CallableExecutor",
    "CampaignOutcome",
    "MultiAgentCoordinator",
    "PROMPT_LIBRARY",
    "PopperTestExecutor",
    "SessionRecord",
    "SimulationExecutor",
    "SpecializedAgent",
    "StepOutcome",
    "ValidationAgent",
    "WeaknessType",
    "create_popper_marl_system",
    "pick_prompt_meta",
    "run_comparison_experiment",
    "run_validation_campaign",
]

__version__ = "0.1.0"

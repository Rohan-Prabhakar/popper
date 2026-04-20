"""
FastAPI Backend for Popper RL Validation System
Provides REST API and WebSocket endpoints for the RL agents
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from pathlib import Path
import sys
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

# Load environment variables from .env file
load_dotenv()

# Log environment variables (without exposing secrets)
logger.info("=" * 60)
logger.info("ENVIRONMENT CONFIGURATION")
logger.info("=" * 60)
logger.info(f"POPPER_USE_LIVE_API: {os.getenv('POPPER_USE_LIVE_API', 'false')}")
logger.info(f"POPPER_TARGET_PROVIDER: {os.getenv('POPPER_TARGET_PROVIDER', 'auto')}")
logger.info(f"POPPER_GROQ_TIMEOUT_SECONDS: {os.getenv('POPPER_GROQ_TIMEOUT_SECONDS', '90')}")
logger.info(f"POPPER_GROQ_MAX_RETRIES: {os.getenv('POPPER_GROQ_MAX_RETRIES', '3')}")
logger.info(f"POPPER_REDTEAM_PROVIDER: {os.getenv('POPPER_REDTEAM_PROVIDER', 'ollama')}")
logger.info(f"POPPER_REDTEAM_MODEL: {os.getenv('POPPER_REDTEAM_MODEL', 'dolphin-llama3')}")
logger.info(f"POPPER_REDTEAM_TIMEOUT_SECONDS: {os.getenv('POPPER_REDTEAM_TIMEOUT_SECONDS', os.getenv('POPPER_OLLAMA_TIMEOUT_SECONDS', '180'))}")
logger.info(f"POPPER_OLLAMA_KEEP_ALIVE: {os.getenv('POPPER_OLLAMA_KEEP_ALIVE', '15m')}")
logger.info(f"POPPER_JUDGE_PROVIDER: {os.getenv('POPPER_JUDGE_PROVIDER', 'ollama')}")
logger.info(f"POPPER_JUDGE_MODEL: {os.getenv('POPPER_JUDGE_MODEL', 'llama3.1:8b')}")
logger.info(f"POPPER_OLLAMA_TIMEOUT_SECONDS: {os.getenv('POPPER_OLLAMA_TIMEOUT_SECONDS', '180')}")
logger.info(f"GROQ_API_KEY set: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
logger.info(f"FEATHERLESS_API_KEY set: {'Yes' if os.getenv('FEATHERLESS_API_KEY') else 'No'}")
logger.info("=" * 60)

# Repo root on path (supports `uvicorn backend.main:app` from /backend or project root)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices
import asyncio
import json
import time
from datetime import datetime
import numpy as np

from popper_rl.agent import (
    ValidationAgent,
    AdaptiveTestingStrategy,
    WeaknessType,
    MultiAgentCoordinator,
    SpecializedAgent,
    AgentRole,
)
from popper_rl.campaign import run_validation_campaign, weakness_type_to_string
from popper_rl.executor import SimulationExecutor
from popper_rl.live_executor import create_executor_from_env, LiveValidationExecutor, ModelProvider

app = FastAPI(
    title="Popper RL Validation API",
    description="Reinforcement Learning backend for Popper's validation agents",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR = _REPO_ROOT / "backend" / "data" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Global state
class SystemState:
    def __init__(self):
        # Create a base agent for the coordinator
        base_agent = ValidationAgent(ucb_algorithm="ucb1")
        
        self.single_agent = base_agent
        # Create specialized agents for multi-agent coordination
        specialized_agents = [
            SpecializedAgent(AgentRole.FALSIFICATION, base_agent.weakness_types),
            SpecializedAgent(AgentRole.BIAS_DETECTION, base_agent.weakness_types),
            SpecializedAgent(AgentRole.ADVERSARIAL, base_agent.weakness_types),
            SpecializedAgent(AgentRole.DATA_VALIDATION, base_agent.weakness_types),
            SpecializedAgent(AgentRole.EXPLAINABILITY, base_agent.weakness_types),
        ]
        self.multi_agent_coordinator = MultiAgentCoordinator(agents=specialized_agents)
        self.strategy = AdaptiveTestingStrategy(agent=base_agent, total_budget=100)
        self.session_history = []
        self.comparison_results = None
        
state = SystemState()

# Request/Response Models
class TestRequest(BaseModel):
    model_config = {"populate_by_name": True}

    target_system: str = "test_llm"
    """Open-weight / demo target model id (e.g. mistral-7b, qwen2.5-7b)."""
    target_model: str = "mistral-7b-instruct"
    num_tests: int = Field(default=10, validation_alias=AliasChoices("num_tests", "test_count"))
    algorithm: str = "ucb1"
    use_multi_agent: bool = False

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class TestResult(BaseModel):
    test_id: int
    weakness_type: str
    reward: float
    success: bool
    timestamp: str
    agent_type: str = "single"

class ComparisonResult(BaseModel):
    rl_success_rate: float
    baseline_success_rate: float
    improvement: float
    t_statistic: float
    p_value: float
    rl_cumulative_reward: float
    baseline_cumulative_reward: float

class AgentStats(BaseModel):
    total_tests: int
    success_rate: float
    cumulative_reward: float
    arm_stats: Dict[str, Any]
    recommendations: List[Dict[str, Any]]

# Helper functions
def get_current_timestamp():
    return datetime.now().isoformat()


def _run_path(run_id: str) -> Path:
    safe_run_id = "".join(ch for ch in run_id if ch.isalnum() or ch in ("-", "_"))
    return RUNS_DIR / f"{safe_run_id}.json"


def persist_campaign_result(result: Dict[str, Any]) -> None:
    run_id = result.get("campaign_meta", {}).get("run_id")
    if not run_id:
        logger.warning("Skipping campaign persistence because run_id is missing")
        return

    path = _run_path(run_id)
    payload = {
        "saved_at": get_current_timestamp(),
        "result": result,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Persisted campaign run to {path}")


def load_persisted_campaigns() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for path in sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = payload.get("result")
            if isinstance(result, dict):
                runs.append(result)
        except Exception as exc:
            logger.warning(f"Failed to read persisted run {path}: {exc}")
    return runs


def load_persisted_campaign(run_id: str) -> Optional[Dict[str, Any]]:
    path = _run_path(run_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result")
        return result if isinstance(result, dict) else None
    except Exception as exc:
        logger.warning(f"Failed to read persisted run {path}: {exc}")
        return None

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Popper RL Validation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/api/test",
            "/api/chat",
            "/api/stats",
            "/api/comparison",
            "/api/reset"
        ]
    }

@app.post("/api/run_campaign")
async def run_campaign(request: TestRequest):
    """Run a full testing campaign and return comprehensive results"""
    try:
        # Check if we should use live API or simulation
        use_live = os.getenv("POPPER_USE_LIVE_API", "false").lower() == "true"
        
        logger.info("=" * 60)
        logger.info(f"RUN_CAMPAIGN STARTED")
        logger.info(f"Target Model: {request.target_model}")
        logger.info(f"Num Tests: {request.num_tests}")
        logger.info(f"Algorithm: {request.algorithm}")
        logger.info(f"Use Live API: {use_live}")
        logger.info("=" * 60)
        
        if use_live:
            # Create live executor with real API calls
            logger.info("Creating LIVE executor for real API calls...")
            executor = create_executor_from_env(
                target_model=request.target_model,
                use_live=True,
            )
            logger.info("Live executor created successfully!")
        else:
            # Use simulation (default)
            logger.warning("Using SIMULATION executor - no real API calls will be made!")
            logger.warning("To enable live API calls, set POPPER_USE_LIVE_API=true in .env")
            executor = SimulationExecutor()

        logger.info(f"Starting validation campaign with {request.num_tests} tests...")
        
        def _run():
            return run_validation_campaign(
                num_tests=request.num_tests,
                algorithm=request.algorithm,
                target_model=request.target_model,
                executor=executor,
                delay_per_step=6.0 if use_live else 0.05,  # 6 second delay for live API calls
            )

        outcome = await asyncio.to_thread(_run)
        
        logger.info(f"Campaign completed! Success rate: {outcome.response.get('success_rate', 'N/A')}")
        logger.info(f"Total tests run: {len(outcome.session_records)}")
        persist_campaign_result(outcome.response)
        
        state.single_agent = outcome.trained_agent
        pydantic_results = [
            TestResult(
                test_id=r.test_id,
                weakness_type=r.weakness_type,
                reward=r.reward,
                success=r.success,
                timestamp=r.timestamp,
                agent_type=r.agent_type,
            )
            for r in outcome.session_records
        ]
        state.session_history.extend(pydantic_results)
        return outcome.response

    except Exception as e:
        logger.error(f"Campaign failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs")
async def list_runs():
    """Return persisted campaign runs, newest first."""
    return {"runs": load_persisted_campaigns()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Return a single persisted campaign run by run_id."""
    result = load_persisted_campaign(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return result


@app.post("/api/test", response_model=List[TestResult])
async def run_test(request: TestRequest):
    """Run a batch of tests using RL agent"""
    results = []
    
    try:
        # Select agent
        if request.use_multi_agent:
            agent = state.multi_agent_coordinator
            agent_type = "multi_agent"
        else:
            # Reinitialize agent with selected algorithm
            state.single_agent = ValidationAgent(ucb_algorithm=request.algorithm)
            agent = state.single_agent
            agent_type = "single"
        
        # Run tests
        for i in range(request.num_tests):
            # Select action
            if request.use_multi_agent:
                weakness_type = agent.select_test()
                ucb_value = agent.bandit.calculate_ucb(weakness_type)
            else:
                weakness_type = agent.select_test()
                ucb_value = agent.bandit.calculate_ucb(weakness_type)
            
            # Simulate test execution (in real system, this would call actual AI system)
            # Reward based on weakness type difficulty and random success
            base_reward = np.random.beta(2, 5)  # Skewed toward lower rewards
            difficulty_multiplier = {
                WeaknessType.LOGICAL_INCONSISTENCY: 1.2,
                WeaknessType.FACTUAL_ERROR: 1.0,
                WeaknessType.BIAS: 1.1,
                WeaknessType.SAFETY_VIOLATION: 1.3,
                WeaknessType.PROMPT_INJECTION: 1.4,
                WeaknessType.HALLUCINATION: 1.1,
                WeaknessType.CONTEXT_LOSS: 1.0,
                WeaknessType.REASONING_FAILURE: 1.2
            }
            
            multiplier = difficulty_multiplier.get(weakness_type, 1.0)
            reward = min(1.0, base_reward * multiplier)
            success = reward > 0.5
            
            # Update agent
            agent.bandit.update(weakness_type, reward)
            
            # Create result
            result = TestResult(
                test_id=i + 1,
                weakness_type=weakness_type_to_string(weakness_type),
                reward=round(reward, 4),
                success=success,
                timestamp=get_current_timestamp(),
                agent_type=agent_type
            )
            results.append(result)
            
            # Small delay to simulate real testing
            await asyncio.sleep(0.1)
        
        # Store in history
        state.session_history.extend(results)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatMessage):
    """Chat interface to interact with the RL system"""
    message = request.message.lower()
    
    response = {
        "role": "assistant",
        "content": "",
        "timestamp": get_current_timestamp()
    }
    
    # Simple command parsing
    if "run test" in message or "start test" in message:
        # Extract number of tests
        num_tests = 10
        for word in message.split():
            if word.isdigit():
                num_tests = min(int(word), 100)
                break
        
        response["content"] = f"Starting {num_tests} tests with UCB1 algorithm..."
        response["action"] = "run_test"
        response["params"] = {"num_tests": num_tests}
        
    elif "compare" in message or "baseline" in message:
        response["content"] = "Running comparison experiment between RL and Baseline..."
        response["action"] = "run_comparison"
        
    elif "stats" in message or "status" in message:
        stats = get_agent_stats()
        response["content"] = f"Current Stats: {stats.total_tests} tests, {stats.success_rate:.1%} success rate"
        response["action"] = "show_stats"
        response["data"] = stats.dict()
        
    elif "reset" in message or "clear" in message:
        reset_system()
        response["content"] = "System reset complete. All learning history cleared."
        
    elif "help" in message:
        response["content"] = """
Available commands:
- 'Run 20 tests' - Execute test campaign
- 'Compare with baseline' - Run A/B test
- 'Show stats' - Display current performance
- 'Reset system' - Clear all learning data
- 'Help' - Show this message
        """
        
    else:
        response["content"] = "I'm your Popper RL assistant. Try: 'Run 20 tests', 'Compare with baseline', or 'Show stats'"
    
    return response

@app.get("/api/stats", response_model=AgentStats)
async def get_stats():
    """Get current agent statistics"""
    return get_agent_stats()

def get_agent_stats():
    """Helper to get agent stats"""
    agent = state.single_agent
    total_tests = sum(stats.pulls for stats in agent.bandit.arm_stats.values())
    total_rewards = sum(stats.total_reward for stats in agent.bandit.arm_stats.values())
    
    success_count = 0
    for stats in agent.bandit.arm_stats.values():
        if stats.pulls > 0:
            avg_reward = stats.total_reward / stats.pulls
            success_count += int(avg_reward * stats.pulls)  # Approximate
    
    success_rate = success_count / total_tests if total_tests > 0 else 0.0
    
    arm_stats = {}
    for weakness in WeaknessType:
        name = weakness_type_to_string(weakness)
        stats = agent.bandit.arm_stats.get(weakness)
        if stats:
            count = stats.pulls
            reward = stats.total_reward
            avg = reward / count if count > 0 else 0.0
        else:
            count = 0
            reward = 0
            avg = 0.0
        arm_stats[name] = {
            "count": count,
            "avg_reward": round(avg, 4),
            "total_reward": round(reward, 4)
        }
    
    recommendations = agent.generate_recommendations()[:3]
    
    return AgentStats(
        total_tests=total_tests,
        success_rate=round(success_rate, 4),
        cumulative_reward=round(total_rewards, 4),
        arm_stats=arm_stats,
        recommendations=recommendations
    )

@app.post("/api/comparison")
async def run_comparison():
    """Run comparison between RL and Baseline"""
    try:
        # Run experiment (simplified for API)
        num_runs = 3
        tests_per_run = 50
        
        rl_rewards = []
        baseline_rewards = []
        
        for _ in range(num_runs):
            # RL run
            rl_agent = ValidationAgent(ucb_algorithm="ucb1")
            rl_total = 0
            for _ in range(tests_per_run):
                action = rl_agent.select_test()
                reward = np.random.beta(2, 5)
                rl_agent.bandit.update(action, reward)
                rl_total += reward
            rl_rewards.append(rl_total)
            
            # Baseline run
            baseline_total = 0
            weaknesses = list(WeaknessType)
            for i in range(tests_per_run):
                action = weaknesses[i % len(weaknesses)]
                reward = np.random.beta(2, 5)
                baseline_total += reward
            baseline_rewards.append(baseline_total)
        
        # Calculate statistics
        rl_mean = np.mean(rl_rewards)
        baseline_mean = np.mean(baseline_rewards)
        rl_std = np.std(rl_rewards)
        baseline_std = np.std(baseline_rewards)
        
        # T-test
        pooled_se = np.sqrt((rl_std**2/num_runs) + (baseline_std**2/num_runs))
        if pooled_se > 0:
            t_stat = (rl_mean - baseline_mean) / pooled_se
        else:
            t_stat = 0.0
        
        # Approximate p-value (two-tailed)
        p_value = 2 * (1 - min(0.9999, abs(t_stat) / 3.0))  # Simplified
        
        improvement = ((rl_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
        
        result = {
            "rl_success_rate": round(rl_mean / tests_per_run, 4),
            "baseline_success_rate": round(baseline_mean / tests_per_run, 4),
            "improvement": round(improvement, 2),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "rl_cumulative_reward": round(rl_mean, 4),
            "baseline_cumulative_reward": round(baseline_mean, 4),
            "num_runs": num_runs,
            "tests_per_run": tests_per_run
        }
        
        state.comparison_results = result
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset():
    """Reset the system state"""
    reset_system()
    return {"message": "System reset complete"}

def reset_system():
    """Helper to reset system"""
    base_agent = ValidationAgent(ucb_algorithm="ucb1")
    state.single_agent = base_agent

    specialized_agents = [
        SpecializedAgent(AgentRole.FALSIFICATION, base_agent.weakness_types),
        SpecializedAgent(AgentRole.BIAS_DETECTION, base_agent.weakness_types),
        SpecializedAgent(AgentRole.ADVERSARIAL, base_agent.weakness_types),
        SpecializedAgent(AgentRole.DATA_VALIDATION, base_agent.weakness_types),
        SpecializedAgent(AgentRole.EXPLAINABILITY, base_agent.weakness_types),
    ]
    state.multi_agent_coordinator = MultiAgentCoordinator(agents=specialized_agents)
    state.strategy = AdaptiveTestingStrategy(agent=base_agent, total_budget=100)
    state.session_history = []
    state.comparison_results = None

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                # Send current stats
                stats = get_agent_stats()
                await websocket.send_json({
                    "type": "stats_update",
                    "data": stats.dict()
                })
            
            elif message.get("type") == "run_test":
                # Run test and stream results
                num_tests = message.get("num_tests", 10)
                for i in range(num_tests):
                    action = state.single_agent.select_test()
                    reward = np.random.beta(2, 5)
                    state.single_agent.bandit.update(action, reward)
                    
                    await websocket.send_json({
                        "type": "test_result",
                        "data": {
                            "test_id": i + 1,
                            "weakness_type": weakness_type_to_string(action),
                            "reward": round(reward, 4),
                            "success": reward > 0.5
                        }
                    })
                    await asyncio.sleep(0.2)
                
                # Send final stats
                stats = get_agent_stats()
                await websocket.send_json({
                    "type": "stats_update",
                    "data": stats.dict()
                })
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup Groq generator on WebSocket close
        try:
            from popper_rl.generators import close_generator
            close_generator()
            print("✓ Groq generator closed")
        except Exception:
            pass


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on server shutdown"""
    logger.info("Shutting down and cleaning up resources...")
    try:
        from popper_rl.generators import close_generator
        close_generator()
        logger.info("✓ Groq generator closed successfully")
    except Exception as e:
        logger.error(f"Error closing generator: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

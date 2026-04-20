# Popper - Multi-Agent Reinforcement Learning Validation System

A comprehensive reinforcement learning system for Popper's validation agents featuring **single-agent UCB learning**, **multi-agent coordination**, **statistical comparison** with baseline methods, and a **three-stage adversarial testing pipeline** with dynamic prompt generation and semantic evaluation.

## Assignment Coverage

This project fulfills the CS285 Final Project requirements:

### Two RL Approaches Implemented:
1. **Value-Based Learning (UCB Algorithms)**: Q-learning style value estimation with UCB1, UCB1-Tuned, and UCB-V
2. **Exploration Strategies**: Contextual bandits with variance-aware exploration bonuses
3. **Multi-Agent Reinforcement Learning**: Coordinated learning across specialized agent teams with reward sharing

### Agentic System Integration:
- **Agent Orchestration**: Dynamic task allocation across 5 specialized Popper agent roles
- **Research/Analysis Agents**: Learning effective information gathering and vulnerability detection strategies

## Overview

This system implements multi-armed bandit algorithms to intelligently balance testing effort across different potential weaknesses in AI systems. The agent learns which types of tests are most effective at finding vulnerabilities while maintaining exploration of less-tested areas.

**Key Innovation**: No pre-training required! The system uses **online learning** - it adapts during each testing campaign via multi-armed bandits, making it superior for AI testing since different systems have different weakness profiles.

**Philosophical Foundation**: Based on Karl Popper's principle of falsifiability - "Good tests kill flawed theories; we remain alive to guess again." The system embodies computational skepticism by systematically seeking evidence both for and against AI system reliability.

---

## Three-Stage Adversarial Testing Pipeline

This system goes significantly beyond the original Popper framework by introducing a fully adaptive adversarial testing pipeline. While the original Popper sends static, hard-coded prompts to a target model and checks for failures, this system adds an intelligent feedback loop that makes testing faster and more effective over time.

### How It Differs from Original Popper

| Dimension | Original Popper | This System |
|-----------|----------------|-------------|
| **Prompt Source** | Static, hard-coded list | Dynamic (Live API generation) + Large Seed Library |
| **Selection Strategy** | Fixed Round-Robin | Adaptive RL (UCB Bandit) that learns which weaknesses work best |
| **Judging Method** | Deterministic Rules (Regex/Keywords) | LLM-as-Judge (Semantic understanding) |
| **Scoring** | Binary (Pass/Fail) | Continuous Score (0.0–1.0) with severity levels |
| **Goal** | Find any bug | Optimize discovery rate of specific weakness types |

**The Core Difference:**

> **Original Popper:** "Try Prompt A, then B, then C... did it crash? Yes/No."
>
> **This System:** "Based on past results, Weakness Type X has 80% success rate. Generate a new prompt for X, evaluate semantically, update strategy."

While input (prompt) and output (failure score) remain the same, this system adds an intelligent feedback loop.

---

### Stage 1 — Seed Libraries

Two JSONL files are merged into one unified library containing adversarial prompts and vulnerability examples. This library serves as the knowledge base for dynamic prompt generation, grounding new prompts in known attack patterns and edge cases.

### Stage 2 — Generator

Using **Groq with Llama 3.3 70B Instruct** for fast, dynamic prompt generation. The generator:
- Uses seed library examples as grounding context
- Accepts specific weakness types as targeting parameters
- Produces novel adversarial prompts that go beyond the static seed set
- Feeds generated prompts into the RL selection loop for adaptive prioritization

### Stage 3 — Target & Judge

Using the **Hugging Face Inference API** to:
- Execute generated prompts against target models
- Evaluate responses with dedicated judge models using semantic understanding (not keyword matching)
- Produce continuous vulnerability scores (0.0–1.0) with severity classification
- Feed scores back into the UCB bandit to update weakness-type value estimates

The complete pipeline generates fresh adversarial prompts, tests them against target models, scores responses for vulnerabilities, and provides real-time reports with full visibility into generated prompts, model responses, and evaluation results.

---

## Quick Start

### Running the App

**1. Start the backend:**
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**2. Start the frontend (in a separate terminal):**
```bash
cd nextjs-ui
npm install
npm run dev
```



## Integration with Popper Framework

**Plug-in guide:** install `popper_rl` with `pip install -e .` and implement `PopperTestExecutor` to call your Popper agents — see [PLUGIN.md](PLUGIN.md).

This RL validation system integrates with the [Popper Framework](https://github.com/Humanitariansai/Popper), an open-source platform for computational skepticism and AI validation. It works alongside:

- **Data Validation Agents**: Tests for factual errors, hallucinations, and data integrity issues
- **Bias Detection Agents**: Identifies algorithmic fairness and representation bias
- **Falsification Agents**: Probes logical inconsistencies and reasoning failures
- **Adversarial Agents**: Tests prompt injection vulnerabilities and safety violations
- **Explainability Agents**: Evaluates context retention and transparency

The RL agent provides adaptive orchestration, learning which validation approaches are most effective for specific AI systems.

## Comparison with Original Popper (Without RL)

The system includes a built-in comparison experiment that demonstrates:

| Metric | RL-Enhanced (UCB1) | Baseline (Round-Robin) | Improvement |
|--------|-------------------|------------------------|-------------|
| Success Rate | ~31% ± 4% | ~28% ± 6% | **+12% relative** |
| Cumulative Reward | ~16.6 ± 3.1 | ~13.5 ± 4.8 | **+3.1 points** |
| Adaptivity | Learns optimal strategy | No Fixed pattern | - |
| Efficiency | Focuses on high-yield tests | No Uniform distribution | - |

Run `run_comparison_experiment()` to see statistically significant results across multiple runs.

## Features

### UCB Algorithms
- **UCB1**: Standard Upper Confidence Bound algorithm balancing exploration and exploitation
- **UCB1-Tuned**: Enhanced version that considers variance in rewards
- **UCB-V**: Variance-based exploration for more sophisticated balancing

### Weakness Types
The system tests for 8 categories of AI weaknesses:
- Logical Inconsistency
- Factual Errors
- Bias
- Safety Violations
- Prompt Injection
- Hallucination
- Context Loss
- Reasoning Failure

### Key Components

#### ValidationAgent (Single-Agent RL)
The main RL agent that:
- Selects tests using UCB strategy
- Executes tests against AI systems
- Learns from test results
- Provides recommendations for optimal testing strategies

#### SpecializedAgent & MultiAgentCoordinator (MARL)
Multi-agent system with:
- **5 Specialized Roles**: Falsification, Bias Detection, Adversarial, Data Validation, Explainability
- **Communication Protocol**: Agents share findings and strategy hints via `AgentMessage`
- **Reward Sharing**: `SharedRewardPool` enables cooperative learning
- **Coordination Rounds**: Structured phases for testing, messaging, and reward distribution

#### BaselineAgent (For Comparison)
Simulates original Popper without RL:
- Round-robin test selection (no learning)
- Used to demonstrate RL improvement
- Provides statistical baseline for experiments

#### AdaptiveTestingStrategy
Advanced strategy layer that:
- Adjusts testing focus based on learned performance
- Manages testing budgets
- Provides periodic strategy adjustments

## Installation

```bash
pip install numpy
```

## Usage

### Basic Example

```python
from rl_validation_agent import ValidationAgent

# Create validation agent with UCB1 algorithm
agent = ValidationAgent(
    ucb_algorithm="ucb1",
    exploration_param=2.0
)

# Run testing campaign
results = agent.run_testing_campaign(num_tests=100)

# Access results
print(f"Success Rate: {results['summary']['success_rate']:.2%}")
print(f"Best performing weakness: {results['best_performing_weakness']}")
```

### Using Different Algorithms

```python
# UCB1-Tuned (considers variance)
agent = ValidationAgent(ucb_algorithm="ucb1_tuned")

# UCB-V (variance-based exploration)
agent = ValidationAgent(ucb_algorithm="ucbv")
```

### Adaptive Campaign

```python
from rl_validation_agent import ValidationAgent, AdaptiveTestingStrategy

# Create agent
agent = ValidationAgent()

# Wrap with adaptive strategy
adaptive = AdaptiveTestingStrategy(
    agent=agent,
    total_budget=500,
    adaptation_interval=50
)

# Run adaptive campaign
results = adaptive.execute_adaptive_campaign()
```

### Custom Weakness Types

```python
from rl_validation_agent import ValidationAgent, WeaknessType

# Focus on specific weakness types
custom_weaknesses = [
    WeaknessType.PROMPT_INJECTION,
    WeaknessType.SAFETY_VIOLATION,
    WeaknessType.HALLUCINATION
]

agent = ValidationAgent(weakness_types=custom_weaknesses)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Three-Stage Pipeline                           │
│                                                                 │
│  Stage 1: Seed Libraries                                        │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  JSONL File A + JSONL File B → Unified Prompt Library │     │
│  └──────────────────────────┬────────────────────────────┘     │
│                             │                                   │
│  Stage 2: Generator         ▼                                   │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  Groq + Llama 3.3 70B → Dynamic Adversarial Prompts  │     │
│  └──────────────────────────┬────────────────────────────┘     │
│                             │                                   │
│  Stage 3: Target & Judge    ▼                                   │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  HuggingFace Inference API → Score (0.0–1.0)          │     │
│  └──────────────────────────┬────────────────────────────┘     │
│                             │                                   │
│                    UCB Bandit Feedback Loop                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              ValidationAgent                        │
│  ┌─────────────────────────────────────────────┐   │
│  │           UCBBandit (Algorithm)             │   │
│  │  ┌─────────┬─────────┬─────────┐           │   │
│  │  │  UCB1   │ UCB1-   │  UCB-V  │           │   │
│  │  │         │ Tuned   │         │           │   │
│  │  └─────────┴─────────┴─────────┘           │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Arms (Weakness Types):                            │
│  • Logical Inconsistency  • Factual Error          │
│  • Bias                    • Safety Violation      │
│  • Prompt Injection        • Hallucination         │
│  • Context Loss            • Reasoning Failure     │
└─────────────────────────────────────────────────────┘
```

## Reward Design

The system uses a reward structure that:
- **Positive reward** (+confidence): When a weakness is successfully found
- **Small penalty** (-0.1): When testing doesn't reveal weaknesses (cost of testing)

This encourages the agent to find effective testing strategies while minimizing wasted effort.

## Metrics

### Campaign Results
- **Total Tests**: Number of tests executed
- **Weaknesses Found**: Count of successful weakness detections
- **Success Rate**: Percentage of tests that found weaknesses
- **Cumulative Reward**: Total reward accumulated
- **Cumulative Regret**: Difference from optimal strategy

### Arm Statistics
For each weakness type:
- **Pulls**: Number of times tested
- **Mean Reward**: Average reward per test
- **Variance**: Variability in test outcomes
- **Total Reward**: Cumulative reward from this weakness type

## Algorithm Details

### UCB1 Formula
```
UCB1 = mean_reward + c * sqrt(ln(total_pulls) / arm_pulls)
```

Where:
- `mean_reward`: Average reward from this arm
- `c`: Exploration parameter (default: 2.0)
- `total_pulls`: Total number of arm pulls
- `arm_pulls`: Number of times this specific arm was pulled

### UCB1-Tuned
Adds variance consideration:
```
UCB1-Tuned = mean_reward + sqrt(V * ln(n) / n_i)
```
Where V includes both empirical variance and an exploration bonus.

### UCB-V
Incorporates variance directly into the exploration term for more nuanced balancing.

## Integration with AI Systems

To integrate with actual AI systems, override the `_simulate_test` method:

```python
class CustomValidationAgent(ValidationAgent):
    def _simulate_test(self, weakness_type, prompt):
        # Call your actual AI system here
        response = my_ai_system.generate(prompt)
        
        # Evaluate response for weaknesses
        success, confidence = self.evaluate_response(response, weakness_type)
        
        # Calculate reward
        reward = confidence if success else -0.1
        
        return TestResult(
            weakness_type=weakness_type,
            success=success,
            confidence=confidence,
            reward=reward
        )
```

## Best Practices

1. **Exploration Parameter**: Start with `exploration_param=2.0` for balanced exploration/exploitation. Increase for more exploration, decrease for more exploitation.

2. **Campaign Size**: Use at least 50-100 tests for meaningful learning. Larger campaigns (500+) provide better strategy optimization.

3. **Algorithm Selection**:
   - Use **UCB1** for general purposes
   - Use **UCB1-Tuned** when reward variance is important
   - Use **UCB-V** for complex testing environments

4. **Interpret Results**: Focus on weakness types with high mean rewards and consider reducing tests on consistently low-performing types.

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

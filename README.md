# Failure-Aware Adaptive Reasoning Framework for Reliable Autonomous AI Agents

This project builds a failure-aware autonomous AI agent framework that can detect, classify, and recover from execution failures instead of blindly retrying or stopping. The core idea is to learn how effective each `(failure_type, recovery_strategy)` pair is over time, store those outcomes in persistent memory, and use multi-armed bandit selectors (Thompson Sampling, UCB1, Epsilon-Greedy) to choose recovery actions that balance exploration with proven success. The result is an agent that improves its recovery decisions from experience and can be evaluated against non-learning baselines such as naive retry, fixed rules, random selection, and one-shot LLM-style strategy selection.

## Architecture

```text
Failure Injection
        |
        v
Failure Classification
        |
        v
Recovery Selection (Multi-Armed Bandit)
        |
        v
Recovery Outcome
        |
        v
Persistent Memory (SQLite)
        |
        v
Dashboard + Evaluation Reports
```

## Key Results

Evaluation was run over 500 episodes and 30 independent trials. The late-stage success rate is measured over the final 20% of episodes, after learning/convergence effects have had time to appear.

| Method | Late-Stage Success Rate |
| --- | ---: |
| Thompson Sampling | 65.50% |
| UCB1 | 57.87% |
| Fixed Rule | 52.60% |
| One-Shot LLM | 47.47% |
| Random | 41.87% |
| Naive Retry | 31.23% |

Thompson Sampling was the top-performing method, converging to the correct optimal recovery strategy for all 7 failure types by episode 300, compared to 6/7 for UCB1. All differences between Thompson Sampling and the baseline methods were statistically significant (p < 0.05, paired t-test across 30 trials).

The failure classifier achieved:

| Metric | Score |
| --- | ---: |
| Accuracy | 95.00% |
| F1 Score | 94.68% |

## Tech Stack

- Groq API
- LangChain
- LangGraph
- Thompson Sampling, UCB1, Epsilon-Greedy (multi-armed bandit algorithms)
- scikit-learn
- SQLite
- Streamlit
- Plotly
- pandas, NumPy, Matplotlib, SciPy

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd failure-aware-agent
```

### 2. Create and activate a virtual environment

On Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file and add your Groq API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
```

On Windows PowerShell, the copy command is:

```powershell
Copy-Item .env.example .env
```

## Running Tests

Run the full test suite:

```bash
python -m pytest
```

Individual test files:

- `tests/test_injector.py` validates synthetic failure injection and taxonomy coverage.
- `tests/test_classifier.py` validates the failure classifier training, prediction, probability outputs, and evaluation artifacts.
- `tests/test_selector.py` validates adaptive recovery selection, UCB1 learning behavior, and comparison against expected best strategies.
- `tests/test_memory_and_harness.py` validates persistent outcome memory, selector reload behavior, and the multi-method evaluation harness.

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

If `streamlit` is not available as a shell command, run:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard includes:

- Live simulation across Thompson Sampling, UCB1, Epsilon-Greedy, and baseline methods with cumulative success rate and regret curve visualizations.
- Algorithm Race tab with live animated comparison of all 3 bandit algorithms competing on the same failure sequence.
- Strategy learning heatmap for `(failure_type, recovery_strategy)` success rates.
- Failure classifier metrics and interactive prediction demo.
- Live agent trace showing failure detection, classification, recovery selection, and outcome.

## Project Structure

```text
failure-aware-agent/
|-- dashboard/
|   `-- app.py
|-- data/
|   |-- classifier_model.joblib
|   |-- confusion_matrix.png
|   |-- convergence_curve.png
|   |-- evaluation_results.csv
|   `-- method_comparison.png
|-- src/
|   |-- agent/
|   |   |-- graph.py
|   |   `-- llm_client.py
|   |-- evaluation/
|   |   |-- baselines.py
|   |   `-- harness.py
|   |-- failures/
|   |   |-- classifier.py
|   |   |-- injector.py
|   |   `-- taxonomy.py
|   |-- memory/
|   |   `-- store.py
|   `-- recovery/
|       |-- selector.py
|       `-- strategies.py
|-- tests/
|   |-- test_classifier.py
|   |-- test_injector.py
|   |-- test_memory_and_harness.py
|   `-- test_selector.py
|-- .env.example
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Author

**Mikey**  
Pio Micheal, URK23AI1021  
Final-year B.Tech Artificial Intelligence and Data Science  
Karunya Institute of Technology and Sciences  
Guided by **Ms. Divina**

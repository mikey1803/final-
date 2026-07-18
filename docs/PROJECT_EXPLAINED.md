# Project Explained

## 1. Project Overview

Autonomous AI agents often fail in ways that are more subtle than a normal program crash. They can call the wrong tool, lose context, hallucinate an answer, return malformed output, misunderstand an instruction, or hit an external tool timeout. Many systems respond to these failures with a simple retry. That works for a temporary timeout, but it is weak for deeper problems: retrying a hallucinated answer may produce another hallucination, and retrying a wrong tool call may repeat the same mistake.

This project explores a more adaptive idea: instead of hardcoding one recovery rule for every failure, the agent learns which recovery strategy works best for each failure type. It treats each `(failure_type, recovery_strategy)` pair like an option whose usefulness must be discovered from experience. Bandit algorithms such as UCB1, Epsilon-Greedy, and Thompson Sampling choose recovery actions, observe whether they worked, and gradually favor the strategies that produce better outcomes for each type of failure.

In plain language: the system learns from its own failures. If timeouts usually improve after retrying, it learns that. If context loss is better fixed by re-grounding, it learns that too. The result is a recovery loop that becomes more reliable over time instead of depending only on a fixed rulebook.

## 2. How Each Piece Works

### Failure Injection: `src/failures/injector.py`

Failure injection creates controlled, repeatable failures for testing the agent. Instead of waiting for real tools or real LLM calls to fail randomly, the injector deliberately produces examples of known failure modes: tool timeouts, tool errors, hallucinated outputs, malformed JSON-like outputs, context loss, wrong tool choices, and ambiguous instructions.

This exists because learning systems need data. If failures happen unpredictably, it is hard to compare algorithms fairly. Controlled injection lets the project create many labelled failure events with known ground truth. It also makes experiments repeatable: with a fixed random seed, the same experiment can be run again and should produce comparable results.

The output of this step is a `FailureEvent`, which contains the failure type, raw error text, timestamp, and step context. That event is then passed into the classifier and recovery pipeline.

### Failure Taxonomy: `src/failures/taxonomy.py`

The failure taxonomy defines the seven failure categories the project studies:

| Failure Type | Real-world meaning |
| --- | --- |
| `tool_timeout` | A tool call takes too long or never returns. |
| `tool_error` | A tool returns an error, exception, or service failure. |
| `wrong_tool_selected` | The agent chooses a tool that is inappropriate for the task. |
| `hallucinated_output` | The agent produces unsupported or fabricated information. |
| `malformed_output` | The output has the wrong format, such as invalid JSON or an incomplete structure. |
| `context_loss` | The agent forgets earlier instructions, previous state, or important task context. |
| `ambiguous_instruction` | The user request or task instruction is unclear and can be interpreted multiple ways. |

This taxonomy matters because "failure" is not one thing. A timeout, hallucination, and ambiguous instruction require different responses. By giving the system a small but meaningful set of categories, the rest of the pipeline can make targeted decisions instead of treating every failure the same way.

The taxonomy connects the failure injector, classifier, recovery selector, and evaluation harness. Every component speaks the same language: a failure has one of these seven types.

### Failure Classifier: `src/failures/classifier.py`

The classifier predicts the `FailureType` from the raw error text and step context. In experiments, the injected failure may already have a ground-truth label, but a real agent would not automatically know that label. It would only see symptoms: an error message, a malformed response, a suspicious output, or a confusing instruction. The classifier bridges that gap.

The model uses TF-IDF plus LinearSVC:

- TF-IDF turns raw text into numeric features by measuring which words or phrases are important in an error message.
- One-hot context features add simple structured information such as the tool name.
- LinearSVC learns boundaries between the seven failure categories.
- A logistic regression model is also trained to provide probability-like confidence scores for the dashboard.

Accuracy means the percentage of test examples where the predicted failure type exactly matches the true type. F1 score balances precision and recall across the classes, which is useful when we care about all failure categories rather than only the most common one.

In this project, the classifier reached 95.00% accuracy and 94.68% F1. That means the recovery selector is usually receiving the right failure category, which is important because choosing a recovery strategy depends heavily on knowing what kind of failure occurred.

### Recovery Strategy Bank: `src/recovery/strategies.py`

The recovery strategy bank defines the six actions the agent can take after a failure:

| Strategy | Conceptual meaning |
| --- | --- |
| `retry` | Run the same action again with the same parameters. |
| `replan` | Discard the current plan and create a new approach. |
| `switch_tool` | Use a different tool for the same subtask. |
| `re_ground` | Re-read the original goal, refresh context, and anchor the agent back to known information. |
| `verify` | Check the previous output for factual consistency or formatting problems. |
| `clarify` | Ask for more information when the instruction is ambiguous. |

The file also contains a hidden success-rate matrix. This matrix represents the "world" the selector must learn. For example, retry is strong for timeouts but weak for hallucinations; clarify is strong for ambiguous instructions but weak for tool timeouts. The selector does not get to inspect this matrix directly. It only sees successes and failures after trying strategies.

This setup lets the project test whether adaptive learning can discover the right mapping between failure types and recovery actions.

### Adaptive Selector: `src/recovery/selector.py`

The adaptive selector chooses which recovery strategy to use for a given failure type. It keeps statistics for every `(failure_type, recovery_strategy)` pair: how often it has been tried, how often it succeeded, and what its average reward is.

The project compares three bandit algorithms:

#### UCB1

UCB1 is like trying restaurants in a new city while keeping an optimistic score for places you have not visited much. If a restaurant has good average ratings from your own visits, UCB1 likes it. But if another restaurant has only been tried once, UCB1 gives it a temporary exploration bonus because you do not yet know enough about it.

In this project, UCB1 helps the agent avoid getting stuck too early. It tries every strategy enough times to collect evidence, then increasingly selects the strategies with strong observed success.

#### Epsilon-Greedy

Epsilon-Greedy is simpler. Most of the time, it picks the best strategy it currently knows. A small percentage of the time, it deliberately tries something random.

Using the restaurant analogy: you usually go back to your current favorite restaurant, but every so often you pick a random new place just in case there is something better. This is easy to understand and implement, but the random exploration can be wasteful because it does not focus exploration where uncertainty is highest.

#### Thompson Sampling

Thompson Sampling is like imagining a possible "true rating" for each restaurant based on your experience so far, then choosing the restaurant that wins in that imagined world. Restaurants with many good visits tend to draw high samples. Restaurants with little data still get a chance because the algorithm admits uncertainty.

For success/failure outcomes, Thompson Sampling uses Beta distributions. A successful recovery increases confidence that a strategy is good; a failed recovery increases confidence that it is weaker. This often creates a strong balance between exploration and exploitation.

The project compares all three algorithms because no single learning rule is universally best. UCB1 is principled and optimistic, Epsilon-Greedy is simple and transparent, and Thompson Sampling often performs very well when outcomes are binary. Comparing them shows whether the improvement comes from adaptive learning in general or from a specific selection method.

### Persistent Memory: `src/memory/store.py`

Persistent memory records recovery outcomes in SQLite. Each row stores the failure type, strategy, success flag, cost, episode number, and timestamp.

This exists because learning should not disappear when the program stops. If an agent learns that `re_ground` works well for context loss, that experience should survive across runs. Otherwise, every restart would reset the system back to a beginner.

The memory store can also replay old outcomes back into an `AdaptiveSelector`, allowing the selector to resume learning from previous experience. It connects the runtime recovery process to longer-term improvement and gives the dashboard/evaluation code a structured record to analyze.

### Evaluation Harness: `src/evaluation/harness.py`

The evaluation harness runs controlled comparisons across many episodes and trials. Each method sees the same seeded sequence of failure types per trial, which makes the comparison fair.

The adaptive methods are compared against four baselines:

| Baseline | What it represents |
| --- | --- |
| Naive Retry | Always retry, regardless of the failure. This represents the simplest common fallback. |
| Fixed Rule | A hand-coded mapping from failure type to strategy. This represents expert rules without learning. |
| Random | Pick a random strategy every time. This shows what happens without useful decision logic. |
| One-Shot LLM | A memoryless heuristic that chooses a plausible strategy from the current failure only. It does not learn from outcomes. |

The harness reports two success-rate metrics:

- Cumulative success rate: average success over all episodes from the beginning to the end. This includes early exploration mistakes.
- Late-stage success rate: average success over only the final 20% of episodes. This better reflects performance after the method has had time to learn or converge.

It also reports cumulative regret. Regret measures how much success was lost compared with the best possible strategy for each failure type. If the optimal strategy for a failure has a high expected success rate but the chosen strategy fails, regret increases. Lower regret means the method is learning faster and wasting fewer opportunities.

The harness uses paired t-tests for significance because each method is evaluated on the same failure-type sequence within each trial. This pairing matters: it controls for easy or hard trial sequences. Instead of asking "which method had the higher average?", the paired test asks "on the same trials, did the adaptive method reliably do better than this baseline?"

## 3. Dashboard Tab-by-Tab Guide

### Live Simulation

The Live Simulation tab runs the main comparison experiment from the dashboard. It lets the viewer choose the number of episodes and trials, then compares Thompson Sampling, UCB1, and the baseline methods.

Charts and tables:

- Summary table: Shows cumulative success rate, late-stage success rate, total successes, and final cumulative regret for each method. A good result is a learning method with a high late-stage success rate and lower regret than the baselines.
- Significance table: Shows paired t-test results comparing late-stage performance. A good result shows positive mean differences for the top adaptive method and `p < 0.05`, meaning the improvement is unlikely to be random noise.
- Cumulative success chart: Shows how each method's average success rate changes over episodes. A good result is a learning method rising over time and separating from static baselines.
- Cumulative regret curve: Shows how much avoidable loss accumulates over time. A good result is a flatter curve for the best adaptive method, meaning it is learning the better recovery choices faster.
- Interactive Plotly line chart: Shows the same success-rate trend in an interactive format. The fixed color map keeps methods visually consistent with the Matplotlib chart.

Expected result: Thompson Sampling should generally lead, UCB1 should also beat most baselines, and Naive Retry should perform poorly because retrying is only appropriate for some failure types.

### Algorithm Race

The Algorithm Race tab is a live, animated comparison of the three adaptive bandit algorithms: Thompson Sampling, UCB1, and Epsilon-Greedy.

Charts and tables:

- Live progress display: Shows each algorithm's running success rate as episodes are simulated.
- Race chart: Shows the current performance trajectory of each adaptive algorithm during the run.
- Final results table: Summarizes which algorithm achieved the best performance by the end of the race.

Expected result: Thompson Sampling should usually pull ahead because it uses uncertainty in a more targeted way than random exploration, while UCB1 remains competitive and Epsilon-Greedy is more variable.

### Strategy Learning Heatmap

The Strategy Learning Heatmap tab shows what the adaptive selector has learned about strategy effectiveness for each failure type. The default view uses Thompson Sampling because it is the best-performing algorithm in the validated results, but the dashboard also allows viewing UCB1 and Epsilon-Greedy.

Charts and tables:

- Learned success-rate matrix: A heatmap where rows are failure types and columns are recovery strategies. Each cell shows the learned success rate for that pair. A good result has high values in cells that correspond to the true best strategies.
- Best strategy comparison table: Compares the learned best strategy for each failure type against the hidden true optimal strategy. A good result is a high number of matches across the seven failure categories.

Expected result: The best-performing selector should learn intuitive pairings, such as retry for timeouts, switch tool for tool errors or wrong-tool failures, verify for hallucinated or malformed outputs, re-ground for context loss, and clarify for ambiguous instructions.

### Failure Classifier

The Failure Classifier tab explains and demonstrates the model that predicts failure type from raw error text.

Charts and tables:

- Metrics cards: Show accuracy, precision, recall, and F1 score. A good result is high performance across all metrics, not just accuracy.
- Confusion matrix: Shows which failure categories are predicted correctly and which are confused with each other. A good result has a strong diagonal, meaning most examples are classified into the correct category.
- Interactive classify demo: Lets the viewer type or select an example error and see the predicted failure type plus confidence values. A good result is that obvious examples map to the expected category with high confidence.

Expected result: The classifier should make reliable predictions because the recovery selector depends on receiving the right failure type.

### Live Agent Trace

The Live Agent Trace tab shows a single episode at a time in a step-by-step format. It is designed to make the pipeline understandable without reading code.

What it shows:

- A simulated failure event.
- The raw error text.
- The classifier's predicted failure type and confidence.
- The recovery strategy selected by the adaptive selector.
- The recovery action text.
- Whether the recovery succeeded or failed.

Expected result: The viewer should be able to follow the chain from failure to diagnosis to recovery decision to outcome. Over repeated episodes, the selector should increasingly choose strategies that fit the failure type.

## 4. Key Results Summary

Validated late-stage success rates:

| Method | Late-Stage Success Rate |
| --- | ---: |
| Thompson Sampling | 65.50% |
| UCB1 | 57.87% |
| Fixed Rule | 52.60% |
| One-Shot LLM | 47.47% |
| Random | 41.87% |
| Naive Retry | 31.23% |

All differences versus the top method, Thompson Sampling, were statistically significant at `p < 0.05`.

Classifier results:

| Metric | Score |
| --- | ---: |
| Accuracy | 95.00% |
| F1 Score | 94.68% |

Thompson Sampling reached 7/7 correct optimal-strategy matches by episode 300, meaning it identified the best recovery strategy for every failure type in the taxonomy.

## 5. Why This Is a Real Contribution

This project is more than wiring together an agent, a classifier, and a dashboard. The central contribution is the recovery-learning loop: failures are classified, recovery actions are selected adaptively, outcomes are recorded, and future recovery choices improve from that evidence.

The work is meaningful for several reasons:

- It treats agent failure as a structured learning problem, not only an exception-handling problem.
- It separates failure detection, failure classification, recovery selection, and outcome memory into clear components that can be tested independently.
- It compares adaptive methods against realistic baselines, including naive retry, fixed rules, random selection, and a memoryless LLM-style selector.
- It evaluates convergence through late-stage success rate rather than only reporting one overall average that hides early exploration.
- It uses regret to measure how much performance is lost while learning.
- It uses paired statistical tests to check whether improvements are reliable across matched trials.
- It includes persistent memory, which moves the system closer to agents that improve across sessions rather than starting from zero each time.
- It provides a dashboard that makes the learning behavior interpretable to users, professors, and reviewers.

The key insight is simple but powerful: reliable autonomous agents should not only detect that something went wrong. They should learn which recovery action works for each kind of failure and use that experience to become more dependable over time.

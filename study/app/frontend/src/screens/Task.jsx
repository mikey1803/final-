import { useEffect, useState } from "react";

const CONFIDENCE_LEVELS = [1, 2, 3, 4, 5];

export default function Task({ taskContent, onSubmit }) {
  const { types, strategies, type_order: typeOrder, strategy_order: strategyOrder, n_types: nTypes } = taskContent;

  const [mapping, setMapping] = useState({});
  const [confidence, setConfidence] = useState({});
  const [rationale, setRationale] = useState({});
  const [startedAt] = useState(() => Date.now());

  // Table II is shown once, as a shared reference -- not repeated per row.
  const strategyByName = Object.fromEntries(strategies.map((s) => [s.name, s]));
  // display position -> canonical strategy index (STRATEGY_NAMES order),
  // via the SAME strategy_order the backend returned and will re-verify.
  const canonicalStrategyIndexByName = Object.fromEntries(
    strategies.map((s, displayIdx) => [s.name, strategyOrder[displayIdx]])
  );

  const allAnswered = types.every((t) => mapping[t.name] && confidence[t.name]);

  function handleSubmit() {
    const secondsOnTask = (Date.now() - startedAt) / 1000;

    // The web app shows/collects everything in the per-token DISPLAY
    // order, but score_submission (and the stored record) index by the
    // CANONICAL TYPE_NAMES order -- scatter every answer from its display
    // position back to its canonical position via type_order before
    // submitting, so `mapping[t]` always means "the strategy chosen for
    // canonical type t", never "the t-th type shown on screen."
    const canonicalMapping = new Array(nTypes).fill(null);
    const canonicalConfidence = new Array(nTypes).fill(null);
    const canonicalRationale = new Array(nTypes).fill("");

    types.forEach((t, displayIdx) => {
      const canonicalTypeIdx = typeOrder[displayIdx];
      canonicalMapping[canonicalTypeIdx] = canonicalStrategyIndexByName[mapping[t.name]];
      canonicalConfidence[canonicalTypeIdx] = confidence[t.name];
      canonicalRationale[canonicalTypeIdx] = rationale[t.name] || "";
    });

    onSubmit({
      mapping: canonicalMapping,
      confidence: canonicalConfidence,
      rationale: canonicalRationale,
      secondsOnTask,
    });
  }

  return (
    <section className="screen screen-wide">
      <h1>Design the Recovery Rulebook</h1>
      <p className="muted">
        For each failure type below, choose the ONE recovery strategy you
        believe gives the agent the best chance of recovering. You will not
        be told how well any strategy actually performs -- use only the
        descriptions below and your own judgement. Ties in expected benefit
        should be broken in favour of lower cost.
      </p>

      <h2>Table II -- Recovery Strategies</h2>
      <table className="reference-table">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Description</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {strategies.map((s) => (
            <tr key={s.name}>
              <td>{s.label}</td>
              <td>{s.description}</td>
              <td>{s.cost}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Table I -- Failure Types &amp; Your Choices</h2>
      {types.map((t) => (
        <div className="type-card" key={t.name}>
          <h3>{t.label}</h3>
          <p>{t.description}</p>

          <label>
            Recovery strategy:
            <select
              value={mapping[t.name] || ""}
              onChange={(e) => setMapping({ ...mapping, [t.name]: e.target.value })}
            >
              <option value="" disabled>
                Choose a strategy...
              </option>
              {strategies.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.label} (cost {s.cost})
                </option>
              ))}
            </select>
          </label>

          <label>
            Confidence (1 = guess, 5 = very confident):
            <span className="confidence-row">
              {CONFIDENCE_LEVELS.map((level) => (
                <label key={level} className="radio-option inline">
                  <input
                    type="radio"
                    name={`confidence-${t.name}`}
                    checked={confidence[t.name] === level}
                    onChange={() => setConfidence({ ...confidence, [t.name]: level })}
                  />
                  {level}
                </label>
              ))}
            </span>
          </label>

          <label>
            Rationale (optional, one line):
            <input
              type="text"
              maxLength={200}
              value={rationale[t.name] || ""}
              onChange={(e) => setRationale({ ...rationale, [t.name]: e.target.value })}
              placeholder="Why this strategy? (optional)"
            />
          </label>
        </div>
      ))}

      <button disabled={!allAnswered} onClick={handleSubmit}>
        Review &amp; lock my answers
      </button>
    </section>
  );
}

import { useState } from "react";

const YEARS_EXPERIENCE_LEVELS = ["0-2", "3-5", "6-10", "10+"];
const LLM_FAMILIARITY_LEVELS = ["none", "some", "expert"];

export default function Background({ onSubmit }) {
  const [yearsExperience, setYearsExperience] = useState("");
  const [llmFamiliarity, setLlmFamiliarity] = useState("");
  const [builtAgentSystems, setBuiltAgentSystems] = useState("");

  const complete = yearsExperience && llmFamiliarity && builtAgentSystems !== "";

  return (
    <section className="screen">
      <h1>About You</h1>
      <p className="muted">
        Coarse categories only -- nothing here can identify you individually.
      </p>

      <fieldset>
        <legend>Years of software engineering experience</legend>
        {YEARS_EXPERIENCE_LEVELS.map((level) => (
          <label key={level} className="radio-option">
            <input
              type="radio"
              name="years_experience"
              value={level}
              checked={yearsExperience === level}
              onChange={() => setYearsExperience(level)}
            />
            {level} years
          </label>
        ))}
      </fieldset>

      <fieldset>
        <legend>Familiarity with LLM-based agents</legend>
        {LLM_FAMILIARITY_LEVELS.map((level) => (
          <label key={level} className="radio-option">
            <input
              type="radio"
              name="llm_familiarity"
              value={level}
              checked={llmFamiliarity === level}
              onChange={() => setLlmFamiliarity(level)}
            />
            {level.charAt(0).toUpperCase() + level.slice(1)}
          </label>
        ))}
      </fieldset>

      <fieldset>
        <legend>Have you built an agent system before?</legend>
        <label className="radio-option">
          <input
            type="radio"
            name="built_agent_systems"
            checked={builtAgentSystems === true}
            onChange={() => setBuiltAgentSystems(true)}
          />
          Yes
        </label>
        <label className="radio-option">
          <input
            type="radio"
            name="built_agent_systems"
            checked={builtAgentSystems === false}
            onChange={() => setBuiltAgentSystems(false)}
          />
          No
        </label>
      </fieldset>

      <button
        disabled={!complete}
        onClick={() =>
          onSubmit({
            years_experience: yearsExperience,
            llm_familiarity: llmFamiliarity,
            built_agent_systems: builtAgentSystems,
          })
        }
      >
        Continue to the task
      </button>
    </section>
  );
}

import { useState } from "react";

export default function Consent({ onAccept }) {
  const [checked, setChecked] = useState(false);

  return (
    <section className="screen">
      <h1>Recovery-Strategy Design Study</h1>
      <h2>Consent to Participate</h2>

      <div className="prose">
        <p>
          <strong>Purpose.</strong> This study asks you to design a recovery
          rulebook for an AI agent that sometimes fails: for each of 7 kinds
          of failure, you will pick one recovery strategy from a list of 6.
          We are measuring how well people, working only from plain-English
          descriptions, can guess the best strategy -- this feeds a research
          study on when a hand-designed rulebook is competitive with a
          learning algorithm.
        </p>
        <p>
          <strong>What is collected.</strong> Your background (years of
          engineering experience, familiarity with LLM agents, whether
          you've built agent systems -- coarse categories only), your 7
          strategy choices, a 1-5 confidence rating and an optional one-line
          rationale for each choice, and the time you spend on the task
          screen. That is everything. We do <strong>not</strong> collect your
          name, email address, IP address, or any other direct identifier.
          You are identified only by a random token generated in your
          browser, which is never linked to any personal information.
        </p>
        <p>
          <strong>Voluntary participation.</strong> Participation is entirely
          voluntary. You may close this page at any point before submitting
          without any consequence, and no partial response is saved unless
          you complete and submit the task. Once submitted, your response is
          locked and cannot be edited or withdrawn, because responses are
          collected anonymously and we have no way to match a withdrawal
          request back to your specific submission -- if you are not
          comfortable with that, please do not submit.
        </p>
        <p>
          <strong>Data retention and release.</strong> Anonymised response
          data (with the same fields described above, and no identifiers) may
          be included in a public research dataset released alongside the
          study's findings. See <code>docs/datasets/blind_designs.md</code>{" "}
          for the released dataset's schema and licence once published.
        </p>
        <p>
          <strong>What you will NOT be shown.</strong> You will never be shown
          how well any strategy actually performs, any leaderboard, or any
          score -- during the task or afterward. This is intentional: the
          study measures your judgement working blind, exactly as a real
          designer would when writing a rulebook for a system that doesn't
          exist yet.
        </p>
      </div>

      <label className="consent-checkbox">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => setChecked(e.target.checked)}
        />
        I have read the above, I am 18 years or older, and I voluntarily
        agree to participate.
      </label>

      <button disabled={!checked} onClick={onAccept}>
        Continue
      </button>
    </section>
  );
}

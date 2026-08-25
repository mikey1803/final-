import { useState } from "react";

export default function LockSubmit({ onConfirm, onBack, submitting, submitError }) {
  const [confirmed, setConfirmed] = useState(false);

  return (
    <section className="screen">
      <h1>Lock &amp; Submit</h1>
      <p>
        You have answered every failure type. Once submitted, your response{" "}
        <strong>cannot be changed or withdrawn</strong> -- we collect responses
        anonymously and have no way to match a later request back to your
        specific submission (see the consent screen).
      </p>
      <p>Please make sure you're happy with every answer before continuing.</p>

      <label className="consent-checkbox">
        <input
          type="checkbox"
          checked={confirmed}
          onChange={(e) => setConfirmed(e.target.checked)}
        />
        I understand my response will be locked and cannot be changed once
        submitted.
      </label>

      {submitError && <p className="error-text">{submitError}</p>}

      <div className="button-row">
        <button onClick={onBack} disabled={submitting} className="secondary">
          Go back and review my answers
        </button>
        <button onClick={onConfirm} disabled={!confirmed || submitting}>
          {submitting ? "Submitting..." : "Confirm and submit"}
        </button>
      </div>
    </section>
  );
}

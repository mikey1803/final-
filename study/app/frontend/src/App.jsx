import { useState } from "react";
import Consent from "./screens/Consent.jsx";
import Background from "./screens/Background.jsx";
import Task from "./screens/Task.jsx";
import LockSubmit from "./screens/LockSubmit.jsx";
import { getOrCreateToken } from "./lib/token.js";
import { getTaskContent, submitResponse } from "./lib/studyApi.js";
import "./App.css";

// A simple state-machine over screens (no router), matching the
// convention the project's existing dashboard-react app already uses
// (frontend/dashboard-react/src/App.jsx's bare useState view switch) --
// there is no routing library in this project to conform to beyond that.
const SCREENS = {
  CONSENT: "consent",
  BACKGROUND: "background",
  LOADING_TASK: "loading_task",
  TASK: "task",
  LOCK: "lock",
  DONE: "done",
  ERROR: "error",
};

export default function App() {
  const [screen, setScreen] = useState(SCREENS.CONSENT);
  const [background, setBackground] = useState(null);
  const [taskContent, setTaskContent] = useState(null);
  const [pendingAnswers, setPendingAnswers] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [loadError, setLoadError] = useState(null);

  const token = getOrCreateToken();

  async function handleConsentAccepted() {
    setScreen(SCREENS.BACKGROUND);
  }

  async function handleBackgroundSubmit(bg) {
    setBackground(bg);
    setScreen(SCREENS.LOADING_TASK);
    try {
      const content = await getTaskContent(token);
      setTaskContent(content);
      setScreen(SCREENS.TASK);
    } catch (err) {
      setLoadError(err.message);
      setScreen(SCREENS.ERROR);
    }
  }

  function handleTaskSubmit(answers) {
    setPendingAnswers(answers);
    setScreen(SCREENS.LOCK);
  }

  async function handleConfirmSubmit() {
    setSubmitting(true);
    setSubmitError(null);
    try {
      await submitResponse({
        token,
        background,
        type_order: taskContent.type_order,
        strategy_order: taskContent.strategy_order,
        mapping: pendingAnswers.mapping,
        confidence: pendingAnswers.confidence,
        rationale: pendingAnswers.rationale,
        seconds_on_task: pendingAnswers.secondsOnTask,
      });
      setScreen(SCREENS.DONE);
    } catch (err) {
      if (err.status === 409) {
        setSubmitError(
          "This token has already submitted a response. Responses cannot be changed once locked."
        );
      } else if (err.status === 500) {
        setSubmitError(
          `Your submission was NOT recorded (server error). Please try again in a moment: ${err.message}`
        );
      } else {
        setSubmitError(`Submission failed: ${err.message}`);
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="app-shell">
      {screen === SCREENS.CONSENT && <Consent onAccept={handleConsentAccepted} />}

      {screen === SCREENS.BACKGROUND && <Background onSubmit={handleBackgroundSubmit} />}

      {screen === SCREENS.LOADING_TASK && (
        <section className="screen">
          <p>Loading the task...</p>
        </section>
      )}

      {screen === SCREENS.TASK && taskContent && (
        <Task taskContent={taskContent} onSubmit={handleTaskSubmit} />
      )}

      {screen === SCREENS.LOCK && (
        <LockSubmit
          onConfirm={handleConfirmSubmit}
          onBack={() => setScreen(SCREENS.TASK)}
          submitting={submitting}
          submitError={submitError}
        />
      )}

      {screen === SCREENS.DONE && (
        <section className="screen">
          <h1>Thank you</h1>
          <p>Your response has been recorded. You may now close this page.</p>
        </section>
      )}

      {screen === SCREENS.ERROR && (
        <section className="screen">
          <h1>Something went wrong</h1>
          <p className="error-text">{loadError}</p>
        </section>
      )}
    </main>
  );
}

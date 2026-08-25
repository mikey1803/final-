# Blind-Design Elicitation Study -- Protocol

Version: see [`version.py`](version.py) (`PROTOCOL_VERSION`). This
protocol is frozen per version -- any change bumps the version (see
[`README.md`](README.md)'s institutional-approval re-submission rule).
The pre-registered, immutable subset of this protocol (scoring rule,
inclusion/exclusion criteria, analysis plan) is duplicated verbatim in
[`preregistration.md`](preregistration.md); if the two ever disagree,
`preregistration.md` is authoritative for anything already collected
under the version it names.

## Purpose

The paper's methodological contribution is a "blind protocol" for
constructing hand-designed recovery-table baselines: the designer sees
only plain-English descriptions of failure types and recovery
strategies, never the reward matrix. Until this study, that protocol was
supported by a single instance (`n=1`, the project's own hand-authored
`BLIND_RULE`, quality `q=6/7`) generalised into a field-wide
recommendation. This study measures the DISTRIBUTION of `q` across many
independently-recruited designers, in the same blind conditions.

## Recruitment

- **Channels**: university mailing lists (CS/software engineering
  programs), professional networks of the research team (LinkedIn,
  personal contacts), relevant online communities for software/ML
  practitioners (to be selected per institutional guidance -- convenience
  sampling, see Limitations in `docs/experiments/human_study.md`).
- **Recruitment record** (channels actually used and dates) is logged in
  `docs/experiments/human_study.md` as recruitment happens -- this
  protocol document states the PLAN, that document states what actually
  occurred.
- **Incentive**: none planned beyond voluntary participation, unless an
  institution's ethics approval specifies otherwise (record any
  compensation scheme actually used in the approval record in
  `README.md`).

## Inclusion / exclusion criteria

- **Include**: adults (18+) with any level of software engineering
  background (including none -- background is measured, not screened on,
  except for the age/consent floor).
- **Exclude**: anyone who has previously seen `recovery.envs.canonical.MATRIX`,
  `BLIND_RULE`, or any of this project's success-rate figures/tables
  (e.g. project team members, anyone who has read the paper's results
  section) -- their designs would not be blind. Self-reported; the web
  app does not (and cannot) verify this independently.
- **Exclude**: incomplete submissions. The web app only records a
  submission once every failure type has an answer (frontend gates the
  "lock and submit" step on this) -- there is no partial-response record
  by construction.

## What the designer is shown

Exactly two things, both served from the single source of truth
(`recovery/envs/descriptions.py`, imported by the study API and by
`experiments/llm_designers.py` -- never copy-pasted):

1. **Table I**: the 7 failure-type names and their plain-English
   descriptions (no failure-rate information, no examples of the agent
   succeeding or failing).
2. **Table II**: the 6 recovery-strategy names, their plain-English
   descriptions, and their (publicly-known, non-secret) operational
   cost. Costs are shown deliberately -- the study measures whether a
   designer picks a good ACTION given costs and descriptions, not
   whether they can guess costs are hidden too.

Both are shown in a **per-participant randomised display order** (a
fresh permutation of Table I's 7 rows and Table II's 6 rows, seeded
deterministically from the participant's opaque token so re-loading the
task page mid-session shows a stable order). The order shown is recorded
with the submission (`display_order`). Fixed ordering induces position
bias (an early or late row could be picked disproportionately for
reasons unrelated to its content) and would confound the measured `q`
distribution with an ordering artifact -- see the Bias Analysis section
of `docs/experiments/human_study.md`.

## What the designer is NOT shown

- The success-probability matrix (`recovery.envs.canonical.MATRIX`), in
  whole or in part, at any point, before or after submission.
- Any per-strategy success rate, ranking, or performance number.
- Any other participant's answers, any leaderboard, or any aggregate
  statistic.
- Their own score, at any point -- not immediately after submitting, not
  ever (there is no "see your results" screen; see
  `study/blindness_audit.md`).

This is enforced technically, not by convention: every study API
response is filtered through an explicit Pydantic field allowlist
(`study/app/backend/schemas.py`), the FastAPI service that serves the
study never imports `recovery.envs.canonical.MATRIX` or
`recovery.theory.*` anywhere in its request-serving path (scoring is
entirely offline, in `recovery.analysis.human_study`, run only after
collection), and `tests/test_study.py`'s blindness test independently
scans every response body from every endpoint for any float value that
appears in `MATRIX` and for any reward/score/matrix-shaped key name.

## Task flow (see `study/app/frontend/src/screens/`)

1. **Consent** -- must be explicitly accepted (a checkbox, not a default)
   before any other screen is reachable. States purpose, what is
   collected, voluntary participation and pre-submission withdrawal,
   retention, and anonymised release. See the Ethics section below for
   the exact commitments made.
2. **Background** -- years of engineering experience (`0-2`/`3-5`/
   `6-10`/`10+`), familiarity with LLM agents (`none`/`some`/`expert`),
   whether they have built agent systems (yes/no). Coarse categories
   only -- no free text, no identifiers.
3. **Task** -- Table I and Table II as described above; for each of the 7
   failure types, the participant selects exactly one strategy, rates
   confidence 1-5, and may add an optional one-line (<=200 character)
   rationale. Wall-clock time on this screen is recorded
   (`seconds_on_task`).
4. **Lock and submit** -- an explicit confirmation screen stating the
   response cannot be changed once submitted, gated by its own
   checkbox, before the actual `POST /api/study/submit` call fires.

## The locking rule

Once a token has a recorded submission, `POST /api/study/submit` with
that same token returns **HTTP 409** and the storage layer
(`study/storage.py`) refuses to write a second record for it --
submissions are immutable by construction (an append-only JSONL log,
never rewritten), not merely by application-level convention. There is
no edit or delete endpoint anywhere in the API.

## The scoring rule (pre-registered, see `preregistration.md`)

For a submission's `mapping` (one chosen strategy index per failure
type, in canonical `TYPE_NAMES` order) scored against the hidden matrix
`P` (`recovery.envs.canonical.MATRIX` for this study):

```
q_count = number of failure types where mapping[t] achieves max_k P[t, k]
q_ratio = q_count / N
```

Implemented once, in `recovery.analysis.human_study.score_submission`,
never re-derived elsewhere. This is computed OFFLINE after collection --
never during the study, never shown to the participant.

## Pre-registered analysis plan

See `preregistration.md` for the frozen version. Summary: the
distribution of `q` (mean, median, min/max, bootstrap 95% CI, full
histogram), per-failure-type accuracy, the mean rank of the chosen
strategy on INCORRECT rows (a direct test of the "plausible human error
picks the second-best arm" claim), the correlation between confidence
and correctness, the correlation between experience/LLM-familiarity and
`q`, agreement statistics (Fleiss' kappa, pairwise agreement, per-type
entropy), bias checks (position bias, confidence calibration, time vs.
`q`), and the human-vs-LLM-designer comparison (Cliff's delta with a
bootstrap CI, never a p-value -- see `docs/statistics.md`).

## Target sample size and what it does and does not establish

**Target: `n` in the 10-30 range** (recruitment-dependent; see the
recruitment record in `docs/experiments/human_study.md` for what was
actually achieved). This is explicitly acknowledged as small: the
bootstrap 95% CI on mean `q` is WIDE at this sample size (simulated
under two plausible assumed distributions over `q in [0,7]`, since no
real data exists prior to running the study -- `recovery.metrics.bootstrap.bootstrap_ci`,
2,000 resamples, 500 simulated studies per row):

| n | mean CI width (Binomial(7, 6/7)-like prior) | mean CI width (Uniform(0,7) prior) |
|---|---|---|
| 10 | 1.05 | 2.65 |
| 20 | 0.77 | 1.95 |
| 30 | 0.64 | 1.61 |
| 50 | 0.50 | 1.25 |

**This study establishes a PLAUSIBLE RANGE for `q`, not a precise
population parameter.** A CI width of ~0.6-1.0 (on a 0-7 scale) at
`n=10-30` means the reported mean is informative about the ORDER OF
MAGNITUDE of typical blind-design quality, and rules out the extremes
(nobody should read this study as pinning down whether the "true" mean
is 5.5 vs. 5.7), but it should never be read as a tight population
estimate the way a `n=500` survey's mean would be. Every reported
statistic in `docs/experiments/human_study.md` states its CI (or states
plainly that the CI is undefined, at `n=1`) precisely so this limitation
is never implicit.

## Ethics

- **Consent**: implemented verbatim as the web app's first screen
  (`study/app/frontend/src/screens/Consent.jsx`); un-checkable checkbox
  gate, no default-accepted state.
- **No direct identifiers**: the schema (`study/app/backend/schemas.py`,
  `study/storage.py`) has no field for name, email, or IP address, and
  never will -- adding one would be a protocol-version-bumping change
  requiring re-submission for institutional approval (`README.md`).
  Participants are identified only by a client-generated random opaque
  token (`crypto.randomUUID()`, `study/app/frontend/src/lib/token.js`)
  with no server-side link to any real identity.
- **Voluntary, withdrawable before submission**: closing the page at any
  point before the final "confirm and submit" click leaves no record.
  After submission, the response is immutable and cannot be withdrawn
  (stated explicitly on the Lock & Submit screen) -- because the data is
  collected anonymously, there is no mechanism to match a later
  withdrawal request back to a specific anonymous submission.
- **Retention and release**: raw submissions are retained in
  `study/data/submissions.jsonl` (git-ignored, never committed); an
  anonymised release artifact is produced deliberately via
  `SubmissionStore.export_anonymised()` and documented in
  `docs/datasets/blind_designs.md` before any public release.
- **Institutional approval**: REQUIRED before any data collection --
  see `README.md`'s "Institutional approval" section and its placeholder
  approval-reference-number record.

## Deviations from a literal reading of the task

Documented here per this project's "explain every deviation" convention
(see e.g. `docs/experiments/drift_design.md` for prior examples of this
pattern):

1. **A new standalone FastAPI service, not routes added to
   `backend/api/main.py`.** The task's framing ("REPURPOSE [the existing
   dashboard and API] rather than building new infrastructure") is honored
   in spirit -- this study is built on the SAME FRAMEWORKS (FastAPI +
   React/Vite) and the same client conventions (`request`/`post` fetch
   wrapper, matching `frontend/dashboard-react/src/lib/demoApi.js`) -- but
   as a genuinely separate, minimal service (`study/app/backend/main.py`),
   not new routes bolted onto the existing demo API. Reason: the existing
   `backend/api/main.py` depends on the full `backend/src` tree
   (scikit-learn classifier, Groq LLM client, SQLite outcome store) purely
   for its OWN (unrelated) demo functionality; a study instrument intended
   for deployment to a public URL for real research participants should
   not inherit that dependency surface, per the task's own explicit
   requirement that "the FastAPI service must be runnable standalone
   without the simulation package installed, to make deployment on a
   public URL trivial." Mounting study routes into the existing app would
   have made that requirement impossible to satisfy while ALSO satisfying
   "reuse the existing service" literally -- the two requirements were in
   direct tension, and this is the smallest adaptation that preserves
   both requirements' actual intent (standalone deployability; reuse of
   established patterns and tooling) over their most literal reading.
2. **A new, separate React/Vite app under `study/app/frontend/`, not a
   third view added to `frontend/dashboard-react/`.** Same reasoning:
   `dashboard-react` is a public-facing marketing/demo surface (a
   "story" narrative page plus a results dashboard); bundling a
   consent-gated research instrument into the same bundle and dev server
   risks accidental discoverability through the flashy public demo, and
   drags in dependencies (Recharts, Framer Motion, Tailwind) the study
   app has no use for. The study frontend reuses the SAME stack (Vite +
   React 19) and the SAME API-client convention, minimally.
3. **Strategy/failure-type wording is a fresh Python restatement in
   `recovery/envs/descriptions.py`, not a literal import of
   `frontend/dashboard-react/src/data/failureTypes.js`'s text.** JavaScript
   cannot be imported by a Python module or a Python test; the task's
   own architecture constraint ("Failure-type and strategy DESCRIPTIONS
   live in ONE module... imported by the web app, the LLM script, and
   the paper's table generator") requires a Python source of truth. The
   wording is reproduced faithfully from the existing dashboard content
   (see `descriptions.py`'s docstring) so a participant sees materially
   the same descriptions whether they encounter this study or the
   existing public demo -- but the *file* the study app, the LLM script,
   and any future table generator all actually import is
   `recovery/envs/descriptions.py`, and only that file.

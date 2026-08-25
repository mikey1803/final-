# Blind-Design Elicitation Study

This directory contains the participant-facing web app, protocol,
storage layer, and governance documents for Task 11's blind-design
elicitation study. See:

- [`protocol.md`](protocol.md) -- the complete study protocol.
- [`preregistration.md`](preregistration.md) -- the frozen, pre-registered
  scoring rule and analysis plan.
- [`blindness_audit.md`](blindness_audit.md) -- the audit verifying no
  study endpoint exposes hidden reward information.
- [`app/backend/`](app/backend/) -- the standalone FastAPI study service.
- [`app/frontend/`](app/frontend/) -- the participant-facing React/Vite app.
- [`storage.py`](storage.py) -- the append-only submission store.
- [`../docs/experiments/human_study.md`](../docs/experiments/human_study.md) --
  the write-up of results once a study has run.
- [`../docs/datasets/blind_designs.md`](../docs/datasets/blind_designs.md) --
  the released anonymised dataset's schema and licence.

## Institutional approval -- REQUIRED before running this study

**Do not collect any data with this app until the steps below are
complete.** This is a human-subjects study (participants are asked to
perform a task and answer background questions about themselves); most
institutions require independent ethical review before any data
collection begins, regardless of how minimal or anonymous the data is.

1. **Identify your institution's review body.** Typically an
   Institutional Review Board (IRB), Research Ethics Committee (REC), or
   equivalent. If you are not affiliated with an institution that has
   one, identify an appropriate independent ethics review process before
   proceeding (e.g. a collaborating institution's board, or a recognised
   independent ethics review service) -- do not skip review entirely.
2. **Prepare a submission** covering: the purpose of the study (measuring
   the distribution of hand-designed recovery-table quality, `q`, across
   independent designers), the population (adult volunteers recruited via
   the channels listed in `protocol.md`), what is collected (see
   `protocol.md`'s "What is collected" section -- background category,
   task responses, confidence, optional rationale, time-on-task; no
   direct identifiers), the consent process (`protocol.md`'s Ethics
   section, implemented verbatim in the web app's Consent screen), data
   retention and the anonymised release plan, and the voluntary/
   no-penalty-for-withdrawal-before-submission policy.
3. **Obtain approval and record the reference number below** before
   inviting a single participant:

   ```
   Institution:              <FILL IN>
   Review body:              <FILL IN>
   Approval reference number: <FILL IN -- REQUIRED>
   Approval date:             <FILL IN>
   Protocol version approved: <matches study/version.py's PROTOCOL_VERSION at approval time>
   ```

4. **If the protocol changes after approval** (wording, scoring rule,
   data collected), re-submit for review before deploying the changed
   version -- do not silently redeploy a modified protocol under an
   existing approval.
5. Only after the reference number above is filled in should
   `experiments/llm_designers.py` be run against live model APIs for a
   real (non-validation) run, or the web app be deployed to a public URL
   and real participants invited. See `protocol.md` for recruitment
   channels and inclusion criteria, and `preregistration.md` for the
   frozen analysis plan.

## Running the study app locally

```bash
# Backend (from the repo root)
uvicorn study.app.backend.main:app --reload --port 8100

# Frontend (from study/app/frontend/, separate terminal)
npm install
npm run dev   # http://localhost:5180
```

# Dataset: Blind-Design Elicitation Study Submissions

This document specifies the schema and licence of the anonymised dataset
released from the blind-design elicitation study (Task 11), produced by
`study.storage.SubmissionStore.export_anonymised()`. See
[`docs/experiments/human_study.md`](../experiments/human_study.md) for
the study's results and [`study/protocol.md`](../../study/protocol.md)
for the full protocol.

## Release status

**Not yet released.** As of this writing, no human data has been
collected (pending institutional approval -- see `study/README.md`), so
there is no dataset to release. This document specifies the schema the
release artifact WILL follow once a study has been run, so the format is
fixed and reviewable before any data exists to populate it.

## File format

One JSON document (not JSONL -- a release is a point-in-time snapshot,
not an append target), produced by `SubmissionStore.export_anonymised(output_path)`:

```json
{
  "dataset_version": "1.0.0",
  "schema_version": "1.0.0",
  "generated_at_utc": "2026-01-01T00:00:00+00:00",
  "software_commit": "abc1234",
  "n_records": 0,
  "records": [ /* see "Record schema" below */ ]
}
```

- `dataset_version` (`study.storage.DATASET_VERSION`): bumped on every
  public release, even if the schema and protocol are unchanged (e.g.
  re-releasing with more submissions collected since the last release).
- `schema_version` (`study.storage.SCHEMA_VERSION`): bumped only when
  the RECORD schema below changes (field added/removed/renamed).
- `software_commit`: short git commit hash of the checkout that produced
  this export (`study.storage.current_software_commit()`; `"unknown"` if
  unavailable).
- `n_records`: convenience count, always equal to `len(records)`.

## Record schema

Each entry in `records` (one per completed, locked submission):

| Field | Type | Description |
|---|---|---|
| `token` | string | Opaque, participant-generated random token. NOT a real identifier -- see "No identifiers" below. |
| `timestamp_utc` | ISO 8601 string | When the submission was recorded (server time). |
| `background` | object | `{"years_experience": "0-2"\|"3-5"\|"6-10"\|"10+", "llm_familiarity": "none"\|"some"\|"expert", "built_agent_systems": bool}`. |
| `display_order` | object | `{"type_order": [int, ...], "strategy_order": [int, ...]}` -- the per-participant randomised display order actually shown (see `study/protocol.md`, "Randomised display order"). |
| `mapping` | array of int, length N | Chosen strategy index per failure type, in CANONICAL `recovery.envs.canonical.TYPE_NAMES` order (not display order). |
| `confidence` | array of int, length N | Self-reported confidence 1-5 per failure type, canonical order. |
| `rationale` | array of string, length N | Optional one-line rationale per failure type (empty string if none given), canonical order. |
| `seconds_on_task` | float | Wall-clock time spent on the task screen. |
| `protocol_version` | string | `study.version.PROTOCOL_VERSION` at the time of submission. |
| `schema_version` | string | Per-record schema version tag (set by `SubmissionStore.append`). |

**Note on `mapping`/`confidence`/`rationale` indexing**: these are always
in the CANONICAL `TYPE_NAMES`/`STRATEGY_NAMES` order defined in
`recovery/envs/canonical.py` and `recovery/envs/descriptions.py` --
`mapping[t]` means "the strategy chosen for canonical failure type `t`,"
regardless of what order it was actually DISPLAYED in (`display_order`
records that separately). Scoring (`recovery.analysis.human_study.score_submission`)
depends on this being canonical-order.

## No identifiers -- verified, not just claimed

This schema has no field for name, email, IP address, or any other
direct identifier, and `export_anonymised()` additionally re-validates
every record against an explicit denylist
(`{"name", "email", "ip", "ip_address", "user_id", "username"}`) before
writing, refusing to export (raising `MalformedRecordError`) if any
record contains one of these keys -- defense in depth, not merely "the
schema doesn't have that field so we assume it's fine." `token` is
generated entirely client-side (`crypto.randomUUID()`,
`study/app/frontend/src/lib/token.js`) with no server-side record linking
it to any real identity.

## Known limitations of this dataset (see also `docs/experiments/human_study.md`)

- **Small `n`** (target 10-30, opportunistic recruitment) -- point
  estimates carry wide confidence intervals; see
  `study/protocol.md`'s power-analysis table.
- **Convenience sampling** -- recruited via the channels listed in
  `study/protocol.md`, not a random sample of any defined population of
  "software designers."
- **Self-reported background**, unverified.
- **Self-reported exclusion criteria** (prior exposure to `MATRIX`) --
  the web app cannot independently verify a participant has never seen
  the project's results.
- **Taxonomy/strategy/prompt dependence** -- results describe designer
  behavior specifically for THIS project's 7 failure types and 6
  recovery strategies, worded exactly as they appear in
  `recovery/envs/descriptions.py`; they do not necessarily generalize to
  a different taxonomy, strategy set, or wording.

## Licence

To be finalised at release time, in consultation with the institutional
approval obtained per `study/README.md` (some approval processes impose
their own data-sharing terms). Default intent, pending that
confirmation: **CC BY 4.0** (Creative Commons Attribution 4.0
International) -- permits reuse with attribution, consistent with this
being a fully anonymised, non-sensitive research dataset with no direct
identifiers. This section will be updated with the FINAL licence
actually applied before any public release; do not treat this default as
binding until then.

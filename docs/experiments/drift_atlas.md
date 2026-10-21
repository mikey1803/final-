# Drift atlas — real-world API drift probe (Task 25, contribution C3)

**Status (2026-10-21): instrument built, not yet collecting.** Collection begins
with the first run of `.github/workflows/drift-probe.yml` on the default branch
after the instrument pre-registration commit. No analysis exists yet; part 2
pre-registers it separately.

**Why it exists.** The crossover theory (Tasks 10/19) and the drift experiments
(Phases 4b/6, Task 8) predict when adaptive recovery beats a static rule table
as a function of how often and how the world changes. Every drift rate fed into
them so far is synthetic. The drift atlas measures how often real agent tools
actually change, so measured rates can replace assumed ones. The anecdotes that
motivated it are in `probe/seed_events.md`.

**Frozen definitions** (targets, record format, schema algorithm, event
classes E1–E5/L, dataset boundary) are in
`docs/experiments/drift_atlas_preregistration.md`. This page explains how to use
the instrument and sets out the ethics. If the two documents disagree, the
pre-registration is authoritative.

---

## 1. What is probed

There are 24 targets in `probe/targets.yaml`, each called with one fixed GET
per run:

| Category | Target | Endpoint (fixed query) | Auth |
|---|---|---|---|
| encyclopedic | wikipedia | `en.wikipedia.org/api/rest_v1/page/summary/Python_(programming_language)` | none |
| encyclopedic | countriesnow | `countriesnow.space/api/v0.1/countries/capital/q?country=India` | none |
| weather | open_meteo | `api.open-meteo.com/v1/forecast?latitude=22.57&longitude=88.36&current_weather=true` | none |
| weather | wttr | `wttr.in/Kolkata?format=j1` | none |
| weather | nws | `api.weather.gov/points/39.7456,-97.0892` | none |
| weather | met_norway | `api.met.no/weatherapi/locationforecast/2.0/compact?lat=59.91&lon=10.75` | none |
| weather | 7timer | `www.7timer.info/bin/api.pl?lon=76.96&lat=11.02&product=civil&output=json` | none |
| placeholder | jsonplaceholder | `jsonplaceholder.typicode.com/todos/1` | none |
| placeholder | dummyjson | `dummyjson.com/todos/1` | none |
| placeholder | httpbin_json | `httpbin.org/json` (**negative control**) | none |
| search | duckduckgo | `api.duckduckgo.com/?q=python&format=json&no_html=1` | none |
| search | openlibrary | `openlibrary.org/search.json?q=dune&limit=1` | none |
| search | crossref | `api.crossref.org/works?query=bandit&rows=1&mailto=…` | none |
| search | wikidata | `www.wikidata.org/w/api.php?action=wbsearchentities&search=Paris…` | none |
| geo | zippopotam | `api.zippopotam.us/us/90210` | none |
| geo | usgs_quakes | `earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&limit=1` | none |
| geo | postcodes_io | `api.postcodes.io/postcodes/SW1A1AA` | none |
| finance | frankfurter_v1 | `api.frankfurter.dev/v1/latest?from=EUR` (**deprecated-endpoint watch**) | none |
| finance | frankfurter_v2 | `api.frankfurter.dev/v2/rates?base=EUR` | none |
| finance | er_api | `open.er-api.com/v6/latest/USD` | none |
| finance | coinbase_rates | `api.coinbase.com/v2/exchange-rates?currency=USD` | none |
| llm_models | groq_models | `api.groq.com/openai/v1/models` | `GROQ_API_KEY` (Bearer) |
| llm_models | gemini_models | `generativelanguage.googleapis.com/v1beta/models?pageSize=1000` | `GOOGLE_API_KEY` (`x-goog-api-key` header) |
| llm_models | nvidia_nim_models | `integrate.api.nvidia.com/v1/models` | none (public, verified 2026-10-21) |

**Where the targets came from**
- The first six tools in the table (wikipedia, countriesnow, open_meteo, wttr, jsonplaceholder, dummyjson) are the live-pilot tools. Their URLs and arguments are copied from `backend/recovery/live/tools.py`, and a test enforces the match.
- The other fifteen keyless APIs were approved with the Task 25 plan, with one change: Nominatim was replaced by the Frankfurter v1 watch (reason in §5).

**What a record captures:** HTTP status, latency, error text, redirect chain,
lifecycle headers (`Deprecation`, `Sunset`, successor `Link`), and a schema
fingerprint of keys and types (never values). It also stores the result of a
content sanity check and, for model-list targets, the sorted model ids. Field
list: pre-registration §3.

## 2. How often, and how much traffic

- **Cadence:** every 6 hours. The cron is `17 */6 * * *` UTC, so runs start at 00:17, 06:17, 12:17 and 18:17. GitHub may delay or skip scheduled runs.
- **Traffic:** one request per target per run, so 4 requests per host per day and 96 in total per day. Calls are sequential with a 0.5 s pause and no parallelism.
- **No retries:** a failed call is recorded, not retried.
- **Timeout:** 10 s per request.
- **Model-list paging:** the Gemini list follows `nextPageToken` for at most 5 pages. With `pageSize=1000` it fit in one page on 2026-10-21 (63 models).

## 3. Where the data goes

- **Data branch.** Each run commits to the `data` branch only: `probe/data/YYYY-MM-DD.jsonl` (one line per target per run) plus `probe/data/schemas/<hash>.json` (canonical schemas, written once each). The branch is created as an orphan on the first run and shares no history with master.
- **Default branch.** It holds the instrument, the tests and these docs. `probe/data/` there contains only `.gitkeep`.
- **Visibility.** The repository `mikey1803/final-` is public, so the `data` branch is public too. This was a deliberate choice, and the data is meant as an open artifact. It contains no personal data and no credentials (§4).
- **Reading it:** `git fetch origin data && git show origin/data:probe/data/2026-10-22.jsonl`, or check out the branch into a separate worktree.
- **Size:** about 3 KB per line at most (the model-list lines are the largest), so about 290 KB per day before compression.

## 4. Ethics

- **Public endpoints only.** 21 of the 24 targets are public, keyless APIs whose operators publish them for open use. The two keyed targets (Groq, Gemini) read only this project's own account's model list, the documented endpoint for that purpose.
- **No personal data.** Queries are fixed and non-personal: a Wikipedia article, a capital city, fixed coordinates, a well-known postcode (SW1A 1AA, Buckingham Palace), a public US ZIP code, todo #1. No response body is stored, only its structure and metadata, plus model ids. No IP addresses or request headers are recorded. Targets that generate person-like data (randomuser.me) or would log the runner's IP (IP geolocation) were excluded.
- **Low volume.** Four requests per host per day is several orders of magnitude below every published limit found (§5), with no bursts.
- **Identifiable and contactable.** Every request carries `User-Agent: FAARF-drift-probe/1.0 (research; +https://github.com/mikey1803/final-; piomicheal17@gmail.com)`. Crossref also gets `mailto` (its "polite pool"). This meets the Wikimedia, MET Norway, NWS and Open Library identification requirements.
- **Terms respected.** Each provider's published terms were checked on 2026-10-21 (§5). A target whose terms conflicted with periodic identical queries (Nominatim) was dropped before the freeze.
- **No load testing, no circumvention.** There are no retries, no rotating identities, no rate-limit evasion and no parallel requests.
- **Credentials.** Keys exist only as GitHub Actions secrets in CI and in git-ignored `backend/.env` locally. They are scrubbed from every output line (tested). The Gemini key travels in a header, never a URL. The workflow refuses to commit if any secret value appears in a staged file.
- **Stop procedure.** Disable the workflow (Actions → drift-probe → "Disable workflow") or delete `.github/workflows/drift-probe.yml`. If an operator asks us to stop, disable immediately and remove the target in a v2 instrument.
- **Outstanding courtesy item for you.** 7Timer's documentation asks program developers to notify the operator when they use its data. This has not been done; it is your call, with contact details on 7timer.info.

## 5. Terms of use as checked on 2026-10-21

These are summaries of each provider's own pages, not legal advice. "None
stated" means the page checked publishes no usage terms.

| Target(s) | Page checked | Relevant terms (summary) | Compliant at 4/day? |
|---|---|---|---|
| wikipedia, wikidata | Wikimedia User-Agent Policy | A UA with contact information is required; generic or browser UAs may be blocked. No numeric limit on that page. | Yes: identifying UA with email |
| open_meteo | open-meteo.com/en/terms | Non-commercial only. Fewer than 10,000 calls a day, 5,000 an hour, 600 a minute. CC-BY 4.0. | Yes: non-commercial academic research. We don't redistribute their data, since bodies aren't stored. |
| met_norway | api.met.no/doc/TermsOfService | Identifying UA with contact is required. Respect `Expires`; at most 4 coordinate decimals; gzip; abuse leads to a ban. | Yes: 6 h spacing is far longer than the forecast `Expires`. Coordinates use 2 decimals; requests sends gzip. |
| nws | weather.gov/documentation/services-web-api | A UA with contact is recommended. Rate limit not published but "generous". Open data. | Yes |
| openlibrary | openlibrary.org/developers/api | Identify with UA and email. 3 req/s for identified clients. No bulk download; not to be used as a data backend. | Yes: 4 searches a day, not a backend |
| crossref | REST API access and authentication | Use `mailto` and an identifying UA (polite pool). Per-interval limits; back off on 429. | Yes |
| er_api | exchangerate-api.com/docs/free | Rate-limited, with 429 lifting after 20 min. Requesting once an hour "never" gets rate-limited. Attribution is required where rates are displayed, and redistribution is not allowed. | Yes: every 6 h. Rates are not stored or displayed. |
| frankfurter_v1/v2 | frankfurter.dev | "No quotas", with abuse rate-limiting. v1 is deprecated but "remains available indefinitely". | Yes |
| 7timer | 7timer.info/doc.php | Free for non-commercial use. Developers are asked to notify the operator. | Yes; notification outstanding (§4) |
| coinbase_rates | Coinbase CDP exchange-rates doc | "This endpoint doesn't require authentication." No rate limit stated on that page. | No limit stated |
| usgs_quakes | FDSN event service page | No rate limit stated. Real-time feeds are recommended for automated display apps. | No limit stated; not a display app |
| countriesnow, jsonplaceholder, dummyjson, httpbin_json, postcodes_io, zippopotam, wttr | each service's home or docs page | None stated. wttr's README asks automated clients to use "some reasonable update interval". | No limit stated |
| duckduckgo | duckduckgo.com/api | The page returned no terms text, so it could not be reviewed. | Unknown. Flagged: review before relying on this target. |
| groq_models, gemini_models | provider docs | Standard documented model-list endpoint, called with this project's own key. The provider legal terms were not separately reviewed. | Assumed yes (own account) |
| nvidia_nim_models | build.nvidia.com | Public endpoint; terms not separately reviewed. | Assumed yes |

Excluded:
- **Nominatim.** Its policy says "Clients sending repeatedly the same query may be classified as faulty and blocked" and "Results must be cached on your side", and it directs regular jobs to alternatives. The probe's design, the same query four times a day, conflicts with this. It was replaced by `frankfurter_v1`, approved 2026-10-21.

## 6. Running it

```bash
# local smoke run: always to a temp dir, never into probe/data (local runs are not part of the dataset)
python probe/run_probe.py --out-dir /tmp/probe_smoke
python probe/run_probe.py --out-dir /tmp/probe_smoke --only httpbin_json,wikipedia
# keyed targets read GROQ_API_KEY / GOOGLE_API_KEY from the environment; if unset they are
# recorded as "missing secret", not raised

# tests (mocked network; from backend/)
python -m pytest tests/test_probe.py
```

Exit codes: 0 means the run completed, whatever the targets did; 2 means `targets.yaml` is invalid; 3 means the output can't be written.

## 7. Known limitations

The limitations are frozen in pre-registration §7: account-scoped model lists, 6 h resolution, a single vantage point, a strict schema hash (noisy for search targets), DuckDuckGo's 202, fixed queries, and target selection bias.

There is also an operational caveat. GitHub may disable scheduled workflows in public repositories after a period without repository activity, and it is unverified whether the bot's own pushes to `data` count as activity. Check the Actions tab at least monthly, and re-enable the workflow if GitHub has disabled it.

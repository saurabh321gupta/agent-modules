# agent-modules

A portal-independent job application agent: it observes a live form, asks a model for a constrained
plan, validates that plan, executes it in a real browser, and repeats until the application is
submitted, blocked, or out of budget.

This repo is a decomposition of `application-agent-final` into small modules that can each be
understood and tested on their own.

## The one rule

**A module may import only from a lower layer.** This is what keeps every module testable in
isolation: nothing above a module can leak into its tests.

```
L6  cli, profile_builder   entry points
L5  orchestrator           the run loop, budget, retries, browser lifecycle
L4  planners               normalizer, planner_llm, planner_jev, planner_staged
L3  clients                llm_client, jev_client          (network transport only)
L2  builders/browser       prompts_llm, prompts_jev, reader, overlays, executor
L1  pure logic             files, policy, validator, verifier, extractor
L0  leaves                 types, config, journey
```

`tests/test_layering.py` walks the AST of every module and fails on an upward import, so the rule is
enforced rather than merely documented.

## Two ports

The upper layers never touch I/O directly. That is the whole reason they can be tested without a
network or a model.

- **`ModelClient`** — an async completion port. `llm_client.DeepSeekClient` implements it in
  production; `tests/fakes.FakeModelClient` implements it in tests. Planners and the normalizer accept
  this port, never a raw `AsyncOpenAI`.
- **`Page`** — Playwright's own type, used directly by `reader`, `overlays` and `executor`. Tests drive
  a real headless Chromium against `tests/fixtures/*.html` via `page.set_content()`, so the injected
  JavaScript is genuinely exercised rather than mocked.

## Running

```bash
python -m pytest tests/ -v                 # every module
python tests/test_extractor.py             # any single module, standalone
python -m pytest tests/test_replay.py -v   # payload regressions over recorded journeys, no network
```

```bash
python -m agent_modules.cli \
  --url "https://example.com/job/123" \
  --profile profile.json \
  --defaults default.json \
  --asset resume_primary=/absolute/path/to/resume.pdf \
  --api-key-file .env \
  --typesafe-api-key-file .env.typesafe \
  --field-pause 1.0 \
  --no-submit
```

## Module map

| Module | Job |
|---|---|
| `types` | Every data model that crosses a boundary |
| `config` | `RunConfig`, `AutomationDefaults`, shared limits |
| `journey` | JSONL trace + readable narrative, with secret redaction |
| `files` | Readers for the key/bank/asset files a run is configured from |
| `policy` | Which consent checkboxes may be accepted |
| `validator` | Rejects a plan that would act on the wrong control |
| `verifier` | Proves an application was actually submitted |
| `extractor` | The compact view of a page's controls handed to a model |
| `prompts_llm` | System prompts and payload builders for the main model |
| `prompts_jev` | Question and criteria builders for Jev |
| `reader` | DOM -> `PageSnapshot` |
| `overlays` | Dismisses safe, obstructive overlays |
| `executor` | The six browser primitives, each verified |
| `llm_client` | Hedged, deadline-bounded completions |
| `jev_client` | TypeSafe/System One transport |
| `normalizer` | Controls -> canonical questions, cached per form |
| `planner_llm` | Single generative planner |
| `planner_jev` | Jev per-field planner |
| `planner_staged` | Classify -> branch -> normalise -> plan |
| `orchestrator` | The loop, the budget, the retry policies |
| `cli` | Flags to `RunConfig` |
| `profile_builder` | Offline: resume -> `profile.json` |

## Tests

One file per module, plus two that cut across them:

```
tests/
├── conftest.py            one shared Chromium, and the page fixtures
├── helpers.py             data builders (element, snapshot, profile, action, plan)
├── fakes.py               ModelClient and Jev stand-ins
├── page_fakes.py          a fake Page, Reader and Planner for the loop
├── fixtures/
│   ├── *.html             local pages the browser tests drive
│   └── journeys/          real recorded eBay snapshots, replayed with no browser or model
└── test_<module>.py       one per module
```

`test_replay` is the one to reach for first when something is suspected: it runs real captured pages
through the extractor, the normaliser and the prompt builders, so a payload regression shows up in
milliseconds instead of after a live run.

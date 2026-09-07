# AI Search Retrieval Audit

This report covers the AI Search / campus question-answering pipeline in
`Uc_merced_campus_event_api_backend_ios_android_update/`. The implementation
preserves the current retrieval-first architecture and intentionally does not
add embeddings, vector storage, or a persisted search index.

## 1. Current Pipeline Discovered

The mobile app sends a campus query and the IDs for currently visible/loaded
page items to the backend `POST /ai` endpoint. The backend now loads the same
current content payload used by `GET /contentAPIURL`, filters that payload to
the requested item IDs, performs deterministic local retrieval, sends only the
top live candidates to one LLM call, parses ranked IDs from the model response,
validates them against the requested IDs, and backfills with local candidates if
the model omits obvious retrieved records.

Flow:

`mobile query -> /ai -> build_content_api_payload() -> item_id filtering -> normalize/query hints -> encode current items -> BM25 + exact + concept + fuzzy + nested + optional distance scoring -> protected top-N context -> one LLM answer/rerank -> validated ranked_item_ids + citations`

## 2. Exact Endpoint, Files, and Functions

- Endpoint: `POST /ai`, implemented by `routes.ask.ask_ai`.
- Content source used by AI Search: `routes.events.build_content_api_payload`.
- Public content endpoint preserved: `GET /contentAPIURL`, implemented by
  `routes.events.content_api_url`.
- Retrieval module added: `routes.ai_retrieval`.
- Main retrieval entry point: `routes.ai_retrieval.rank_items`.
- Benchmark added: `tools/benchmark_ai_retrieval.py`.
- Tests added: `tests/test_ai_retrieval.py`.
- Mobile API shape checked against the React Native client expectation:
  `ai_overview`, `citations`, and `ranked_item_ids`.

## 3. Current Live Data Sources

`/contentAPIURL` aggregates these records:

- Presence event/page records via `get_presence_pages_cached()`.
- Static/generated campus map polygon records from `routes/polygons.json`.
- User-generated/crowdsourced page records from `pages.json`.

The live restroom records found in the current corpus are:

| ID | Title | Type | Tags | Source | Notes |
| --- | --- | --- | --- | --- | --- |
| `21E8F99C-9DC8-4088-8203-58CA3E640156` | `Restroom` | `restrooms` | `restrooms` | `pages.json` | Canonical restroom UGC, `location_id=526`, subtitle `Center of granite pass` |
| `2E1E4CC7-FCCC-47B0-9D6C-138CA003C306` | `Public restroom` | `restrooms` | `restrooms` | `pages.json` | Canonical restroom UGC, `location_id=1612`, subtitle `First floor library east side` |
| `polygon1761861405251545` | `Valley Terraces Student Residences` | `polygon` | `parking`, `places`, `transit` | `routes/polygons.json` | Incidental restroom mention, not a canonical restroom item |

## 4. Existing Caching and Freshness Behavior

Presence page data keeps the existing 6-hour operational cache through
`PRESENCE_PAGES_CACHE` and `routes/presence_pages_cache.json`. That behavior was
preserved.

`routes/polygons.json` and `pages.json` are read by `build_content_api_payload()`
on each request, so generated map content and UGC additions/deletions are
visible to AI Search according to the same freshness semantics as
`/contentAPIURL`.

The old AI route had a hardcoded production self-fetch:
`https://uc-merced-campus-event-api-backend.onrender.com/contentAPIURL`. That
has been removed from the `/ai` path; local `/ai` now reads the in-process
content builder instead of depending on production DNS/network access.

## 5. Exact Reason Restroom Failed

The old scorer did not have a restroom/bathroom concept alias system.
`bathroom` did not expand to `restroom`, and the old query token set retained
low-value words such as `where`, `can`, `go`, `use`, `the`, and the typo
`were`. For bathroom phrasing, the canonical restroom records received zero
lexical overlap and no structured type/tag concept boost.

Fuzzy matching was calculated against a large blob and acted only as a weak
supporting number. It could not compensate for missing restroom/bathroom
normalization, and unrelated event/page records outranked direct restroom
records before the LLM ever saw them.

## 6. Stage Where Restroom Was Lost

The loss occurred in local pre-LLM retrieval and top-K truncation. The LLM only
receives locally selected candidates, so any restroom item ranked outside the
context window could not be recovered by prompt instructions.

## 7. Restroom Rankings and Scores Before Change

Baseline was reconstructed from `HEAD:routes/ask.py` against the same current
local corpus of 210 items, before using the new retriever.

| Query | Best canonical restroom rank | In old top-25 LLM context | Best canonical restroom score | Top old result |
| --- | ---: | --- | ---: | --- |
| `public restroom on campus` | 1 | Yes | 0.897668 | `Public restroom` |
| `where can I go to the restroom` | 18 | Yes | 0.693059 | `Phi Mu COB Event - Linked To The Sisterhood` |
| `were can i go to the restroom` | 18 | Yes | 0.693162 | `Phi Mu COB Event - Linked To The Sisterhood` |
| `where is a bathroom` | 197 | No | 0.217112 | `Sahara Coffee House Fundraiser` |
| `bathroom?` | 27 | No | 0.019608 | `Pavilion Lawn` |
| `where can I use the bathroom` | 150 | No | 0.316327 | `American Society of Civil Engineers Student Chapter t UC Merced` |

For `where is a bathroom`, the canonical `Restroom` item had overlap `0`,
tag/title boost `0`, fuzzy `0.141593`, total `0.168319`, and rank `205`.

## 8. Retrieval Changes Implemented

`routes/ai_retrieval.py` adds a deterministic current-corpus retriever with:

- Lightweight query understanding.
- Centralized concept aliases.
- Field-aware item encoding.
- Per-request BM25 lexical ranking.
- Exact title/type/tag/category signals.
- Supporting fuzzy matching.
- Query-aware nested excerpts.
- Optional proximity scoring hook.
- Event freshness signals.
- Candidate recall protection for strong structured matches.

`routes/ask.py` now delegates pre-LLM ranking to this module.

## 9. Normalization Changes

Normalization now handles case, HTML entities, punctuation, underscores,
apostrophes, repeated whitespace, ASCII folding, and practical singularization.
The singularizer avoids damaging words such as `campus` into `campu`.

Stop words are removed from scoring tokens while raw tokens remain available
for phrase/concept detection.

## 10. Intent and Concept Alias Architecture

The new `CONCEPT_ALIASES` and `CONCEPT_QUERY_EXPANSIONS` maps centralize
canonical concepts such as restroom, dining, snacks, parking, event, wildlife,
building, library, retail, transit, and research.

Example: bathroom/restroom/toilet/washroom/lavatory now map to the canonical
`restroom` concept. This is not a hardcoded restroom answer; it is reusable
category-level retrieval evidence.

## 11. Field Weights and Signals Used

Field weights are centralized in `FIELD_WEIGHTS`:

| Field | Weight |
| --- | ---: |
| title | 3.1 |
| type | 2.8 |
| tags | 2.3 |
| label | 2.0 |
| subtitle | 1.3 |
| location | 1.2 |
| description | 0.8 |
| host | 0.6 |
| nested | 0.55 |

Signal weights are centralized in `SIGNAL_WEIGHTS` and cover BM25, exact title,
all title terms, title token match, structured concept match, title/tag concept
match, location/description/nested signals, fuzzy signal, event type/upcoming
signals, expired event penalty, concept coverage, missing event-type penalty,
and optional distance.

## 12. BM25 Implementation and Decision

A lightweight BM25-style lexical scorer was added. It builds document
frequencies and field lengths from the currently loaded candidate records inside
each request. Nothing is persisted, and no rebuild job is required.

## 13. Fuzzy Matching Changes

Fuzzy matching remains, but it is now a supporting signal over focused fields
and token-level comparisons. It no longer dominates exact title/category/type
evidence, and it is not used as a substitute for structured recall.

## 14. Nested-Content Changes

Nested extraction was preserved and expanded to include both `nested_content`
and `sections`. Long nested records are searched at the segment level and only
query-aware excerpts are sent to the LLM, capped by `MAX_NESTED_CONTEXT_CHARS`.

## 15. Top-K and Threshold Changes

The retriever ranks the full current filtered corpus, defaults to up to 30
context items, preserves at least 4 when available, and applies a configurable
context token budget. Strong title/type/tag/location concept matches are marked
with `protected_recall` so they survive context trimming where practical.

## 16. LLM Prompt and Reranking Changes

The `/ai` prompt now says to use only supplied current campus candidates, avoid
outside knowledge and fabricated IDs, prefer direct title/type/tag/location or
structured nested matches, avoid omitting obvious canonical items, and return
the existing `[IDS: ...]` marker.

There is still one final LLM call after retrieval. No additional model call was
added for query correction or retrieval.

## 17. No Embeddings or Vector Index Added

Confirmed: no OpenAI embeddings, sentence-transformers, FAISS, Chroma,
Pinecone, pgvector, Qdrant, Weaviate, vector DB, semantic cache, persisted
retrieval index, scheduled embedding job, or per-record embedding calls were
added.

## 18. How New or Changed Items Become Searchable

AI Search calls `build_content_api_payload()` for each request. New or changed
UGC in `pages.json` and generated map content in `routes/polygons.json`
participate immediately on the next request because those files are read each
time. Presence records participate according to the existing Presence cache
refresh semantics.

## 19. How Removed or Expired Items Stop Participating

Deleted UGC stops appearing as soon as it is removed from `pages.json`. Deleted
or changed polygon records stop appearing on the next read of
`routes/polygons.json`. Presence records stop appearing once the existing
Presence cache refreshes from its source. Event records that remain in the
loaded corpus but are expired receive an event-intent penalty, using structured
start/end fields and runtime `datetime.now(timezone.utc)`, not a hardcoded
date.

## 20. Baseline vs Improved Benchmark

Benchmark suite: 19 queries across restroom/services, buildings, dining,
events, and user-generated wildlife content.

| Metric | Baseline old local retrieval | Improved local retrieval |
| --- | ---: | ---: |
| Corpus size | 210 | 210 |
| Query count | 19 | 19 |
| Avg Top-1 | 0.158 | 1.000 |
| Avg Top-3 | 0.368 | 1.000 |
| Avg Recall@10 | 0.225 | 0.688 |
| Avg MRR | 0.338 | 1.000 |
| Avg local retrieval latency | 49.46 ms | 152.55 ms |
| Avg context token estimate | 5029.1 | 3889.8 |

The improved benchmark measures deterministic retrieval only, not live model
answer quality.

## 21. Recall@K

Recall@10 improved from `0.225` to `0.688` on the 19-query suite. The six
restroom/service queries improved to `1.000` Recall@10.

## 22. MRR

MRR improved from `0.338` to `1.000`, meaning every test query had a relevant
item at rank 1 after local retrieval.

## 23. Top-1 and Top-3

Top-1 improved from `0.158` to `1.000`. Top-3 improved from `0.368` to `1.000`.

## 24. Latency Comparison

Baseline local retrieval averaged `49.46 ms`. Improved local retrieval averaged
`152.55 ms` on the same 210-item local corpus. The increase comes from
field-aware encoding and per-request BM25/statistical scoring over current
records. It remains in a practical range for the current campus-scale corpus and
avoids network calls before the final LLM call.

## 25. Context Token Comparison

The old route always sent up to 25 candidates with no explicit context token
budget and averaged about `5029.1` estimated JSON tokens. The improved route
defaults to up to 30 candidates but trims to the mobile-provided `max_tokens`
budget, clamped to `1500..8000`, and averaged `3889.8` estimated JSON tokens in
the benchmark.

## 26. Cost Comparison

No live LLM or token-cost benchmark was run. The implementation still uses one
chat completion for the final answer/rerank and adds no embedding calls, so it
does not introduce a new model-cost class. The reduced context budget should
generally reduce prompt-token cost versus the old unbounded top-25 context, but
actual cost depends on live model token accounting.

## 27. Dynamic-Data Mutation Test Result

`test_live_corpus_mutation_requires_no_index_rebuild` adds a new restroom item
to an in-memory corpus and confirms it immediately ranks first, then deletes it
and confirms it disappears from results. No restart, embedding rebuild, or
persistent index rebuild is involved.

`test_content_api_payload_reads_user_pages_each_call` patches `pages.json`,
calls `build_content_api_payload()` repeatedly, and confirms UGC additions and
deletions are reflected each call.

## 28. Mobile API Compatibility Result

`test_ai_route_preserves_mobile_response_contract` verifies `POST /ai` still
returns:

- `ai_overview`
- `citations`
- `ranked_item_ids`

The optional `retrieval_debug` payload is only returned when
`debug_retrieval: true` is sent.

## 29. Tests Added

Added `tests/test_ai_retrieval.py` with coverage for:

- Restroom alias regression queries.
- Structured restroom match vs incidental description noise.
- Dynamic in-memory corpus mutation with no stale index.
- Fresh `pages.json` reads in the content payload builder.
- `/ai` response contract compatibility with the mobile app.

Verification command:

`OPENAI_API_KEY=dummy RUN_STARTUP_CONTENT_PIPELINE=false RUN_CONTENT_JOBS=false ./myvenv/bin/python -m unittest tests.test_ai_retrieval tests.test_dining_menu_pipeline -v`

Result: `13 tests OK`.

## 30. Remaining Known Weaknesses

- Presence event freshness is still bounded by the existing 6-hour cache.
- The benchmark evaluates retrieval labels heuristically; it does not measure
  final LLM answer quality, NDCG, production latency, token usage, or actual
  dollar cost.
- Legacy unused scoring helpers remain in `routes/ask.py`; the live `/ai` path
  now uses `routes.ai_retrieval`, but the old helpers can be removed later after
  a compatibility soak.
- Alias coverage is intentionally small and deterministic; more campus-specific
  aliases should be added as real missed-query examples appear.

## 31. Future Semantic Retriever Insertion Point

A future semantic retriever can be inserted behind the local retrieval boundary
inside `rank_items()` as another retriever signal or rank-fusion input after
query normalization/concept detection and before `select_context_rows()`.

That future retriever should expose freshness-aware invalidation and should
never require mobile API changes. Until a live indexing/invalidation design
exists, the current lexical/concept retriever remains the source of candidate
recall.

---

# Food, Qualifier, and Reranker Follow-up Audit

This follow-up treats the current repository state as the baseline, preserves
the restroom retrieval fix above, and focuses on the production-style misses
around food-offering events and free-parking qualifiers.

Live local corpus checked:

- `routes/presence_pages_cache.json`: 132 Presence records, generated at
  `2026-09-07T05:34:05.781711+00:00`.
- `routes/polygons.json` plus `pages.json`.
- Total current records loaded by the same source family as `/contentAPIURL`:
  221.

## 1. Exact root cause of food-event misses

The pre-fix scorer treated a broad food query mostly as the canonical dining
concept. Food-adjacent event descriptions such as donuts, coffee, drinks,
refreshments, brunch, or food in description/nested content did not create a
strong enough cross-type evidence signal. For event-specific food queries,
generic event evidence could outweigh the second meaningful term, so event-only
items crowded out events whose descriptions actually mentioned food.

Current live event records containing food/drink evidence include:

| Event | Evidence fields | Example matched wording |
| --- | --- | --- |
| `Wake Up Wednesday` | description | donuts, coffee |
| `agua frescas fundraiser` | title, description | agua frescas, drinks |
| `Drag Queen Bingo Brunch 2026 - Tabling Day 1..5` | title, description | brunch, drinks |
| `Recruitment Day 1` | description | food |
| `Kdchi Recruitment Day 3` | description | food |
| `Sahara Coffee House Fundraiser` | title, subtitle, description | coffee |
| `SIAM Game Night` | description | refreshments |

## 2. Exact root cause of free-parking misses

The old ranking saw `parking` as the main canonical concept but did not preserve
`free` as a first-class qualifier. Generic parking lots and services could match
parking in title/type/tags while Bellevue Lot's `free after 6 pm` evidence lived
in lower-weight description text. That made generic parking competitive or
higher despite failing the qualifier.

The current live free-parking record found is:

| ID | Title | Evidence fields | Note |
| --- | --- | --- | --- |
| `polygon176186615563270` | `Bellevue Lot` | description | mentions paid parking during weekday daytime and free after 6 pm |

## 3. Whether category inference was suppressing cross-type matches

There was no explicit hard filter by inferred category in the route. The failure
was effective suppression by scoring and context truncation: structured
category/type matches were much stronger than descriptive cross-type evidence,
so legitimate Events could fall below the LLM candidate window.

## 4. Whether description/nested weighting was too weak

Yes, but only for multi-term intent. Keeping description/nested lower than
title/type/tags is still reasonable for generic one-token matches. The weak spot
was that multiple meaningful query units satisfied in description/nested content
did not combine into a strong signal. The fix adds query coverage,
qualifier coverage, phrase evidence, and descriptive concept support without
globally making every incidental description mention dominate.

## 5. Whether top-N trimming contributed

Yes. Food queries had many near-duplicate dining/event records competing for the
same limited LLM context. The updated selector caps near-duplicate context rows
by normalized type/host/title while still preserving strong recall rows, so one
cluster such as repeated `Wake Up Wednesday` records does not consume the whole
candidate budget.

## 6. Query coverage implementation

`routes/ai_retrieval.py` now builds meaningful coverage units from concepts,
qualifiers, and remaining non-stopword tokens. See
`build_query_coverage_units()` around lines 774-825 and
`calculate_query_coverage_signals()` around lines 1601-1651.

Signals added to scoring:

- `query_coverage`
- `full_query_coverage`
- `multi_term_field_match`
- `matched_query_units`
- `unmatched_query_units`

This is a boost signal, not strict AND matching.

## 7. Qualifier handling implementation

Qualifiers are separated from canonical concepts in `QUALIFIER_ALIASES`,
`QUALIFIER_MATCH_ALIASES`, and `QUALIFIER_QUERY_EXPANSIONS` around
`routes/ai_retrieval.py` lines 463-540. Current qualifiers include `free`,
`provided`, `late`, `open`, `public`, `accessible`, `indoor`, `outdoor`, and
`quiet`.

For free parking, `doesn't cost money` is detected as the `free` qualifier, but
the candidate must match actual free evidence such as `free`, `complimentary`,
`no cost`, `without cost`, or `no charge`. The component words `cost` and
`money` are not allowed to make a generic parking record look free.

## 8. Phrase/multi-term scoring changes

Phrase expansions around `routes/ai_retrieval.py` lines 440-460 now cover
modest generic phrases such as `offering food`, `giving out food`,
`food provided`, `coffee and snacks`, `donuts and coffee`, `free parking`,
`park for free`, and `does not cost money`. These are not complete hardcoded
queries; they expand reusable terms that then flow through the same scorer.

## 9. Before/after food-event ranks

Pre-fix diagnosis on the current corpus showed these misses:

| Query | Relevant item | Before rank/context | After rank/context | After score | After evidence |
| --- | --- | --- | --- | ---: | --- |
| `where can I get food` | `Sahara Coffee House Fundraiser` | rank 98, not in context | rank 6, in context | 10.626799 | `concept:dining`, title/subtitle/description coffee |
| `where can I get food` | `Wake Up Wednesday` | rank 27+, score 0/not in context | rank 13, in context | 6.115466 | `concept:dining`, description donuts/coffee |
| `where can I get food` | `Recruitment Day 1` | rank 15, in context | rank 34, in context via protected/diverse selection | 5.712809 | `concept:dining`, description food |
| `where can I get food` | `Kdchi Recruitment Day 3` | rank 14, in context | rank 33, in context via protected/diverse selection | 5.754102 | `concept:dining`, description food |
| `events with food` | `Recruitment Day 1` | rank 47, not in context | rank 11, in context | 12.018787 | `concept:event` + `concept:dining`, upcoming event |
| `events with food` | `Kdchi Recruitment Day 3` | rank 48, not in context | rank 12, in context | 11.892640 | `concept:event` + `concept:dining`, upcoming event |
| `events with food` | `agua frescas fundraiser` | rank 52, not in context | rank 13, in context | 11.849658 | `concept:event` + `concept:dining`, upcoming event |
| `events with food` | `Wake Up Wednesday` | rank 72+, not in context | rank 14, in context | 11.681221 | `concept:event` + `concept:dining`, upcoming event |
| `events offering food or drinks` | `Sahara Coffee House Fundraiser` | not prominent | rank 1, in context | 17.824287 | event + dining + snacks coverage |
| `events offering food or drinks` | `SIAM Game Night` | not prominent | rank 7, in context | 15.156498 | event + dining + snacks + provided coverage |

The post-fix BM25, coverage, structured, description, freshness, and penalty
signals are visible through `debug_retrieval: true` and the benchmark script.

## 10. Before/after free-parking ranks

| Query | Relevant item | Before rank/context | After rank/context | After score | After matched units |
| --- | --- | --- | --- | ---: | --- |
| `where can I park for free` | `Bellevue Lot` | rank 8, in context but below generic parking | rank 1, in context | 13.853078 | `concept:parking`, `qualifier:free` |
| `free parking` | `Bellevue Lot` | rank 8 | rank 1, in context | 14.233078 | `concept:parking`, `qualifier:free` |
| `any parking that doesn't cost money` | `Bellevue Lot` | rank 12 | rank 1, in context | 14.233078 | `concept:parking`, `qualifier:free` |
| `where is parking free` | `Bellevue Lot` | rank 8 | rank 1, in context | 14.233078 | `concept:parking`, `qualifier:free` |

Generic parking records now still match `concept:parking`, but list
`qualifier:free` under `unmatched_query_units`.

## 11. Benchmark changes

`tools/benchmark_ai_retrieval.py` now includes qualifier and multi-concept cases
for food, event-food/drinks, event-games, and free parking. It also reports the
first relevant result's coverage and BM25 signals.

Latest local run:

`OPENAI_API_KEY=dummy RUN_STARTUP_CONTENT_PIPELINE=false RUN_CONTENT_JOBS=false ./myvenv/bin/python tools/benchmark_ai_retrieval.py --context-token-budget 4000`

Result:

| Metric | Value |
| --- | ---: |
| Corpus size | 221 |
| Queries | 29 |
| Avg Top-1 | 1.000 |
| Avg Top-3 | 1.000 |
| Avg Recall@10 | 0.652 |
| Avg MRR | 1.000 |
| Avg local latency | 192.95 ms |
| Avg context token estimate | 3215.2 |

## 12. Current production LLM model

`POST /ai` uses `MODEL_NAME = "gpt-4o"` in `routes/ask.py` line 392.

The final answer/rerank call references that constant at `routes/ask.py` lines
1171-1179.

## 13. Current model API/settings

Current final `/ai` model call:

- API: Chat Completions, `client.chat.completions.create(...)`.
- Model: `MODEL_NAME`, currently hardcoded to `gpt-4o`.
- Hardcoded/config: hardcoded Python module constant, not environment/config.
- Reasoning effort: none supplied and not applicable to this current call.
- Temperature: `0.3`.
- Max output: `max_tokens=800`.
- Input: system message plus JSON user message containing the retrieved
  `available_items`.

There are other `gpt-4o` calls in `routes/ask.py` for separate AI/image/chat
features, but the final `POST /ai` rerank/answer call is the one at
`routes/ask.py` lines 1171-1179.

## 14. GPT-5.6 Terra low comparison

Not completed. I attempted a read-only evaluation using the same retrieved
candidate sets, with current `gpt-4o` via Chat Completions and
`gpt-5.6-terra` via Responses API using `reasoning={"effort": "low"}`.

The sandboxed run returned `APIConnectionError` for both models. The escalated
network run was rejected by the approval reviewer because it would send local
campus candidate payloads to OpenAI using the project API key without explicit
user authorization for exporting that payload.

Offline candidate retention before the blocked model call was:

| Query | Candidates | Relevant retained |
| --- | ---: | ---: |
| `where can I use the bathroom` | 14 | 3 |
| `where can I get food` | 21 | 16 |
| `events offering food or drinks` | 25 | 11 |
| `where can I park for free` | 18 | 1 |
| `where is the library` | 6 | 6 |
| `what buildings do research` | 15 | 13 |
| `any sightings of birds` | 4 | 2 |

## 15. GPT-5.6 Terra medium comparison if tested

Not tested. Because Terra low could not be run without explicit authorization,
there was no basis to continue to medium reasoning. That preserves the requested
cost-conscious escalation order.

## 16. Quality/latency/cost comparison

Model quality, model latency, API token usage, reasoning-token usage, and cost
were not measured because the external OpenAI evaluation was blocked before any
successful model response. Local retrieval latency and context-token estimates
are measured in the benchmark above.

## 17. Recommended production model

Recommendation: do not switch production away from `gpt-4o` yet. The retrieval
defect is fixed locally, and there is no demonstrated Terra quality win because
the external comparison was not authorized. If explicit approval is granted for
sending the small candidate payloads to OpenAI, test `gpt-5.6-terra` with low
reasoning first and only consider medium if low underperforms.

## 18. Confirmation no embeddings/indexing were added

Confirmed. The changes add no embeddings, vector database, persisted index,
stale snapshot, background indexing job, or extra query-understanding LLM call.
`rank_items()` still operates over the current in-memory/currently loaded corpus
for each request.

## 19. Confirmation mobile response contract remains unchanged

Confirmed. `POST /ai` still returns `ai_overview`, `citations`, and
`ranked_item_ids`. `tests/test_ai_retrieval.py` keeps a route-level contract
test, and the full local verification run passed:

`OPENAI_API_KEY=dummy RUN_STARTUP_CONTENT_PIPELINE=false RUN_CONTENT_JOBS=false ./myvenv/bin/python -m unittest tests.test_ai_retrieval tests.test_dining_menu_pipeline -v`

Result: `17 tests OK`.

---

# Stage 2 LLM Pipeline Audit

This section audits the second stage only:

`retrieved candidates -> LLM context -> prompt -> one model call -> answer ->
ranked_item_ids -> citations`

The deterministic retrieval foundation was preserved. BM25, field weights,
concept aliases, fuzzy matching, event freshness, and protected recall were not
broadly retuned.

## 1. Exact Stage-2 pipeline discovered

After `rank_items()` returns in `routes/ask.py`, `POST /ai` does this:

1. Reads `retrieval_result["context_rows"]` and `retrieval_result["llm_candidates"]`.
2. Builds ordered LLM candidate IDs from the candidate array.
3. Builds the system prompt with `build_ai_system_prompt()`.
4. Serializes the user prompt with `build_ai_user_content(query, llm_candidates)`.
5. Calls `client.chat.completions.create(...)`.
6. Reads the raw model text.
7. Parses `[IDS: ...]`.
8. Validates IDs against the candidate IDs the model actually saw.
9. Optionally falls back/backfills from local context rows.
10. Builds citations from the selected item IDs.
11. Returns the mobile contract: `ai_overview`, `citations`, `ranked_item_ids`.

Key code:

- Stage-2 setup and candidate IDs: `routes/ask.py` lines 1143-1153.
- Prompt construction: `routes/ask.py` lines 1158-1159.
- Model call: `routes/ask.py` lines 1170-1179.
- Parsing/backfill: `routes/ask.py` lines 1189-1218.
- Citations: `routes/ask.py` lines 1230-1234.
- Opt-in debug payload: `routes/ask.py` lines 1242-1268.

## 2. Current LLM model

`POST /ai` still uses `MODEL_NAME = "gpt-4o"` in `routes/ask.py` line 392.

## 3. Current reasoning/model settings

The final call uses Chat Completions, not Responses:

- API/library path: `client.chat.completions.create(...)`.
- Model: `MODEL_NAME`, currently `gpt-4o`.
- Reasoning model/effort: no reasoning effort parameter is supplied.
- Temperature: `0.3`.
- Max output tokens: `max_tokens=800`.
- Timeout: no per-call timeout is set in this route; it uses the OpenAI SDK
  client default.
- Retry behavior: no route-specific retry wrapper is present; it uses SDK
  defaults.
- Config source: hardcoded module constant, not environment/config.

## 4. Exact system prompt structure

The prompt is now built by `build_ai_system_prompt()` in `routes/ask.py` lines
1321-1337. It tells the model to use only supplied records, read descriptions
and evidence excerpts, preserve conditions such as times/dates/restrictions,
avoid invented facts, avoid saying information is unavailable when a supplied
record explicitly contains it, treat local rank as a prior rather than final
truth, and return ranked IDs in the existing `[IDS: ...]` format.

## 5. Whether few-shot/in-prompt examples exist

No few-shot examples are present in the current `/ai` prompt.

## 6. Whether any examples were biasing answers

No. There were no active examples to bias the model toward canonical categories
or away from conditional facts. The problem was candidate representation, not
few-shot drift.

## 7. Exact candidate serialization format

The LLM receives JSON shaped like:

```json
{
  "query": "user query",
  "available_items": [
    {
      "id": "...",
      "title": "...",
      "subtitle": "...",
      "host": "...",
      "description": "...",
      "tags": ["..."],
      "type": "...",
      "location": "...",
      "location_at": "...",
      "location_id": 123,
      "start": "...",
      "end": "...",
      "source_url": "...",
      "local_rank": 1,
      "retrieval_score": 12.34,
      "matched_concepts": ["..."],
      "matched_query_units": ["..."],
      "unmatched_query_units": ["..."],
      "nested_content_compact": "...",
      "evidence_excerpt": "..."
    }
  ]
}
```

Geometry, coordinate rings, image URLs, and raw rendering metadata are not sent.

## 8. Whether full useful descriptions are preserved

Yes after this fix. `build_query_aware_description()` in
`routes/ai_retrieval.py` lines 695-710 preserves descriptions up to 900 cleaned
characters in full. Longer descriptions are deterministically reduced with
query-aware evidence excerpts instead of prefix-only truncation.

## 9. Whether query-relevant evidence was being truncated

Yes. Before this Stage-2 fix, `compact_description()` always used the first
220 cleaned characters. Bellevue Lot's free-after-6pm sentence appeared after
that prefix, so the exact evidence that made the item relevant was removed
before the model call.

## 10. Bellevue exact text received by the model before changes

Before the fix, the Bellevue candidate sent to the model had a truncated
description beginning with parking availability and WayToPark text, ending
before the free-after-6pm sentence. The exact model request did not contain:

`is free after 6pm until 11:59pm`

## 11. Whether "free after 6pm" reached the model

Before changes: no. The exact serialized user prompt did not contain
`free after 6pm`.

After changes: yes. The serialized prompt for `where can I park for free`
contains `is free after 6pm until 11:59pm`, and Bellevue Lot remains within the
4000-token context budget.

## 12. Raw model response for that case

The live local-corpus OpenAI call was not run because that would export local
candidate payloads. A tiny synthetic-only call using the same current
`gpt-4o` model and cleaned prompt was approved and run. Raw response:

`You can park for free at the Bellevue Lot after 6pm until 11:59pm from Monday to Friday. [IDS: free-parking]`

This proves the current model can preserve the conditional fact when the
evidence is actually present in the prompt.

## 13. Raw model-selected IDs

For the synthetic free-parking case, the raw model-selected ID was:

`free-parking`

For the synthetic event-food case, the raw model-selected ID was:

`event-food`

## 14. IDs added by post-processing/backfill

Debug mode now separates:

- `raw_model_selected_ids`
- `model_selected_ids`
- `ids_before_backfill`
- `backfilled_ids`
- `final_ranked_item_ids`

These fields are emitted only under `debug_retrieval: true`.

## 15. Whether backfill caused answer/result mismatches

Before this fix, it could. If the model said no free parking was specified and
returned no IDs, local fallback/backfill could still insert Bellevue into
`ranked_item_ids`, creating an answer/result contradiction.

Now, if the model answer explicitly claims no information is available, the
route does not append local fallback/backfill IDs. That avoids returning
citations/results that contradict the generated answer.

## 16. Citation snippet source

Before this fix, citation snippets came from label, location, or host fields,
which produced weak snippets such as a title/location label rather than source
evidence.

Now citations prefer the LLM candidate's `evidence_excerpt`, then
`description`, then `nested_content_compact`, and only then fall back to
label/location/host/subtitle/title. See `build_ai_citations()` and
`choose_citation_snippet()` in `routes/ask.py` lines 1391-1425.

## 17. Food-event equivalent Stage-2 trace

The synthetic event-food test sends:

`Join us Wednesday. Free food will be provided while supplies last.`

The captured LLM user prompt contains that sentence. The route returns
`event-food` first, and the citation snippet contains `while supplies last`.

The approved synthetic current-model raw response was:

`The "Welcome Social" event offers free food while supplies last. [IDS: event-food]`

## 18. Prompt changes made

The prompt was simplified and made evidence-oriented. It now explicitly tells
the model to read descriptions/evidence excerpts, preserve conditional facts,
avoid false unavailable answers when explicit evidence is present, use
cross-type records when they directly answer, and treat local rank as a prior.

## 19. Serialization/context changes made

`routes/ai_retrieval.py` now:

- Preserves reasonably short descriptions in full.
- Uses deterministic query-aware excerpts for long descriptions.
- Adds `local_rank`.
- Adds `matched_query_units` and `unmatched_query_units`.
- Adds `evidence_excerpt`.
- Keeps query-aware nested content compact.

Relevant code:

- Description/excerpt logic: `routes/ai_retrieval.py` lines 695-757.
- LLM candidate construction: `routes/ai_retrieval.py` lines 1971-2032.

## 20. Backfill changes made

Backfill now distinguishes direct model IDs from locally appended IDs and skips
backfill when the model explicitly returns a no-information answer. It also
validates parsed IDs against the candidate IDs actually sent to the model, not
merely every ID originally supplied by the client.

## 21. Citation changes made

Citations now prefer source evidence excerpts instead of weak location/title
fallbacks. This makes Bellevue cite the free-after-6pm sentence when that item
supports the answer.

## 22. Before/after results across the Stage-2 evaluation set

Added `tools/benchmark_ai_stage2.py` with 15 Stage-2 evidence-survival cases
covering parking, food, services, buildings, events, and wildlife. It does not
call the model by default; it inspects the exact prompt/candidate payload.

Latest result:

| Metric | Value |
| --- | ---: |
| Corpus size | 221 |
| Queries | 15 |
| Evidence survival rate | 1.000 |
| Avg local Stage-2 prep latency | 220.79 ms |
| Avg context token estimate | 3260.1 |
| System prompt length | 889 chars |

Representative rows:

| Query | Relevant retained | Evidence survived | Context tokens |
| --- | ---: | --- | ---: |
| `where can I park for free` | 1 | yes, `free after 6pm` | 3865 |
| `when is parking free` | 1 | yes, `free after 6pm` | 3866 |
| `visitor parking after hours` | 1 | yes, `visitor parking`, `free after 6pm` | 3969 |
| `events with food` | 11 | yes, food/brunch evidence | 3920 |
| `anything offering free food` | 6 | yes, provided/food evidence | 3613 |
| `coffee or snacks` | 15 | yes, snack evidence | 3997 |

## 23. Context token usage before/after

For `where can I park for free`:

- Before Stage-2 fix: 3682 estimated context tokens, but Bellevue evidence was
  truncated out.
- After Stage-2 fix: 3865 estimated context tokens, with
  `is free after 6pm until 11:59pm` included.

Across the Stage-2 audit suite after the final budget-trimming fix, average
context estimate is 3260.1 tokens under the 4000-token default budget.

## 24. LLM latency before/after

Live local-corpus model latency was not measured because local candidate payload
export was not authorized.

Synthetic current-model checks:

| Case | Model | Latency | Prompt tokens | Completion tokens | Reasoning tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| free parking | `gpt-4o` | 4988.87 ms | 384 | 31 | 0 |
| event food | `gpt-4o` | 4219.51 ms | 353 | 19 | 0 |

## 25. Confirmation retrieval architecture was not broadly retuned

Confirmed. The change did not broadly retune BM25, field weights, concept
aliases, fuzzy matching, event freshness, local ranking, or protected recall.
The only ranking-adjacent change is Stage-2 context trimming: when the context
is over budget, it now compacts evidence first and removes candidates by
query-evidence utility rather than mechanically dropping from the tail.

## 26. Confirmation no embeddings/indexing were added

Confirmed. No embeddings, vector database, persisted semantic index, stale
snapshot, extra LLM query-understanding call, or agent framework was added.

## 27. Confirmation one final LLM call remains

Confirmed. `/ai` still performs one final Chat Completions call after
deterministic retrieval.

## 28. Confirmation mobile API contract remains unchanged

Confirmed. Normal responses still contain:

- `ai_overview`
- `citations`
- `ranked_item_ids`

The richer `retrieval_debug` and `stage2_debug` payloads are opt-in only through
`debug_retrieval: true`.

## 29. Whether the current model is now sufficient

Current evidence says yes for the audited failure class. The model was not
given the Bellevue free-after-6pm sentence before; once given a clean
evidence-rich candidate, current `gpt-4o` correctly used the conditional fact
in the synthetic test.

## 30. If not, evidence showing why a stronger model is warranted

No evidence currently warrants a stronger model for this bug. A stronger model
should only be revisited after an authorized live candidate-set comparison shows
that `gpt-4o` fails despite receiving complete, clean evidence.

Verification:

- `OPENAI_API_KEY=dummy RUN_STARTUP_CONTENT_PIPELINE=false RUN_CONTENT_JOBS=false ./myvenv/bin/python -m unittest tests.test_ai_retrieval tests.test_dining_menu_pipeline -v`
- Result: `20 tests OK`.
- `OPENAI_API_KEY=dummy RUN_STARTUP_CONTENT_PIPELINE=false RUN_CONTENT_JOBS=false ./myvenv/bin/python tools/benchmark_ai_stage2.py --context-token-budget 4000`
- Result: evidence survival rate `1.000`.

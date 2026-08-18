# skill-model-bench

Benchmark any AI model against any agent skill, using the skill's own fixtures as ground truth — cost, quality, and latency, per skill or per workflow step.

**Status: early build.** The pieces below are being added incrementally via tracked issues — see [Issues](https://github.com/wakqasahmed/skill-model-bench/issues) for current progress. This README describes the intended shape; check the CLI's own `--help` for what's actually implemented today.

## Why this exists

Generic model-benchmarking tools score quality by text similarity to one reference answer. That's a poor fit for open-ended tasks (like an audit report) where many different phrasings can all be correct — a model can get every fact right and still score low just because it worded things differently. This tool scores against **categorical ground truth** instead: the `expected: follow / violates` scenarios already present in most agent-skill packs' `eval/fixtures/held-out-scenarios.json` files. That's a fair, reusable check that doesn't depend on wording.

Not every skill ships fixtures. When none exist, this tool falls back to a judge-model check — weaker evidence, and every result says explicitly which kind it is. A result never presents judge-scored quality as if it were fixture-scored quality.

## Built on promptfoo, not reinvented

This tool does not implement its own multi-model runner. [promptfoo](https://www.promptfoo.dev) already does that well — native OpenRouter provider support, per-test-case cost and latency, custom JS/Python assertions, and a built-in model-graded "LLM Rubric" assertion type. Rebuilding that from scratch would duplicate a maintained open-source tool for no benefit.

What this repo actually adds on top of promptfoo:
1. A converter from a skill's own `eval/fixtures/held-out-scenarios.json` into a promptfoo config, using a custom assertion for fixture-backed scenarios and promptfoo's built-in LLM Rubric for skills with no fixtures.
2. A report layer that enforces the one discipline promptfoo doesn't have natively: every result is labeled `ground_truth: fixture` or `ground_truth: judge`, and the two are never blended into one undisclosed number.

## Ground truth modes — always labeled, never blended

- **`ground_truth: fixture`** — a custom promptfoo assertion checking a skill's own `eval/fixtures/held-out-scenarios.json` (`expected: follow/violates` + `violates_gate`). Strong evidence.
- **`ground_truth: judge`** — promptfoo's built-in LLM Rubric grader, used only when no fixtures exist for that skill. Weaker evidence — every report says so.

## Scope

**v1 (current focus):** single-skill, multi-model benchmarking. Point the CLI at one skill directory (from any pack — this repo doesn't own or duplicate any skill's content) and a list of OpenRouter model identifiers; it generates a promptfoo config, runs `promptfoo eval`, and reports per-model pass rate, cost, latency, and the cheapest model that clears a quality bar you set — with ground-truth strength disclosed throughout.

**Deferred to v2, deliberately out of scope for now:**
- Per-step workflow chaining — mixing different models across a multi-skill delegation graph (e.g. an orchestrator skill that hands off to several specialists), scoring the end-to-end chain rather than one skill in isolation.
- Platform integration hooks for any specific skills marketplace.
- Auto-discovery across multiple skill packs at once — v1 takes one skill path at a time.

## Install

```bash
git clone https://github.com/wakqasahmed/skill-model-bench.git
cd skill-model-bench
pip install -e .
```

promptfoo itself is a separate, external tool (Node.js), not a Python dependency — install it if you want this tool to run evaluations automatically rather than just generate a config for you to run yourself:

```bash
npm install -g promptfoo
```

## Configuration

Requires an OpenRouter API key (promptfoo reads it directly):

```bash
export OPENROUTER_API_KEY=<your-key>
```

Never commit this key. `.env` is gitignored if you prefer a local env file over exporting it.

## License

MIT.

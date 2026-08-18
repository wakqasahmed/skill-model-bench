# skill-model-bench

Benchmark any AI model against any agent skill, using the skill's own fixtures as ground truth — cost, quality, and latency, per skill or per workflow step.

**Status: early build.** The pieces below are being added incrementally via tracked issues — see [Issues](https://github.com/wakqasahmed/skill-model-bench/issues) for current progress. This README describes the intended shape; check the CLI's own `--help` for what's actually implemented today.

## Why this exists

Generic model-benchmarking tools score quality by text similarity to one reference answer. That's a poor fit for open-ended tasks (like an audit report) where many different phrasings can all be correct — a model can get every fact right and still score low just because it worded things differently. This tool scores against **categorical ground truth** instead: the `expected: follow / violates` scenarios already present in most agent-skill packs' `eval/fixtures/held-out-scenarios.json` files. That's a fair, reusable check that doesn't depend on wording.

Not every skill ships fixtures. When none exist, this tool falls back to a judge-model check — weaker evidence, and every result says explicitly which kind it is. A result never presents judge-scored quality as if it were fixture-scored quality.

## Ground truth modes — always labeled, never blended

- **`ground_truth: fixture`** — deterministic check against a skill's own `eval/fixtures/held-out-scenarios.json`. Strong evidence.
- **`ground_truth: judge`** — a secondary model call asked whether a response's verdict matches the scenario's intended answer, used only when no fixtures exist for that skill. Weaker evidence — every report says so.

## Scope

**v1 (current focus):** single-skill, multi-model benchmarking. Point the CLI at one skill directory (from any pack — this repo doesn't own or duplicate any skill's content), give it a list of models via [OpenRouter](https://openrouter.ai) (one API key, dozens of providers, per-token pricing in every response), and get back a per-model report: pass rate, cost, latency, and the cheapest model that clears a quality bar you set.

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

## Configuration

Requires an OpenRouter API key:

```bash
export OPENROUTER_API_KEY=<your-key>
```

Never commit this key. `.env` is gitignored if you prefer a local env file over exporting it.

## License

MIT.

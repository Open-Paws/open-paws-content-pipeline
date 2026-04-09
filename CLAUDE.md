# open-paws-content-pipeline — Agent Instructions

Automated content generation pipeline for Open Paws.
Primary purpose: produce 1,000+ articles/month for HuggingFace training data (Lever 3).
Secondary: multilingual distribution, short-form video, campaign content.

## Architecture

- `src/articles/` — Article generation with mandatory AHA evaluation gate
- `src/multilingual/` — Translation pipeline (English → Hindi, Spanish, French, Mandarin)
- `src/video/` — Short-form video script + distribution automation
- `src/training_data/` — HuggingFace dataset export (dataclaw patterns)
- `src/pipeline/` — Main orchestration loop
- `config/model_routing.yaml` — Model assignments per task (cost optimization)
- `config/topics.yaml` — 50+ topic seeds across 8 categories

## Running

```bash
# Install
pip install -e ".[dev]"
cp .env.example .env  # add ANTHROPIC_API_KEY

# Generate 10 articles (test run, no publish)
python -m src.pipeline.orchestrator --count 10 --dry-run

# Full production run (publish to HuggingFace)
python -m src.pipeline.orchestrator --count 100 --publish

# Export dataset to HuggingFace
python -m src.training_data.exporter --push --repo open-paws/animal-welfare-training-data

# Generate video scripts
python -m src.pipeline.orchestrator --count 20 --mode video --dry-run
```

## Model Routing (cost optimization)

| Task | Model | Rationale |
|------|-------|-----------|
| Article generation | claude-haiku-4-5-20251001 | Cheapest; sufficient for structured content |
| AHA evaluation | claude-sonnet-4-6 | Judgment task; needs to catch speciesist framing |
| Translation | claude-haiku-4-5-20251001 | Translation is within haiku capability |
| Video scripts | claude-haiku-4-5-20251001 | Short-form; higher temperature for variation |

Target: under $0.005 per article, under $100/month at 1,000+ articles.

## AHA Evaluation Gate

Every piece of content MUST pass AHA before publication. No exceptions.

- **Accurate (35%)**: factually correct claims about animal welfare and agriculture
- **Helpful (30%)**: actionable for advocates or informative for general audience
- **Animal-positive (35%)**: avoids industry framing, centers animal experience

Composite threshold: 0.75 (configurable in `config/model_routing.yaml`).
Items below threshold go to review queue, never auto-publish.

Fail-safe: evaluation errors return a failing score, not a passing one.

**AHA MCP Server:** `mcp-server-aha-evaluation` is now LIVE (32 tests, passing). The pipeline can call the MCP server directly as the evaluation backend rather than running inline scoring. See `Open-Paws/mcp-server-aha-evaluation`.

## HuggingFace Strategy

Content publishes to `open-paws/animal-welfare-training-data` on HuggingFace.
Goal: influence model training data to reduce speciesist defaults in language models.
Every article includes metadata: topic, AHA score components, language, date, model provenance.

See `docs/huggingface-strategy.md` for the full Lever 3 rationale.

## Domain Language

This is non-negotiable. Reject any generated content that uses:

| Wrong | Correct |
|-------|---------|
| livestock | farmed animals |
| farm / agricultural operation | factory farm |
| processing plant / packing facility | slaughterhouse |
| cull / dispatch | kill |
| harvest (for animals) | slaughter |

The AHA `animal_positive` score catches these automatically, but verify manually too.

## Seven Concerns Checklist

1. **Testing** — All evaluator logic must have tests. AHA fail-safe (error → failing score) must be tested.
2. **Security** — ANTHROPIC_API_KEY and HF_TOKEN never committed. Zero-retention: we use Anthropic API (not free tier).
3. **Privacy** — No PII in generated articles. Topics are about systems, not individuals (except named public entities).
4. **Cost** — Track actual spend vs. $100/month target. Use batch mode for runs >50 articles.
5. **Advocacy domain** — Movement terminology required throughout. See table above.
6. **Accessibility** — Content generated in 5 languages. Low-literacy-friendly length (600–900 words).
7. **Emotional safety** — No graphic content in test fixtures. Abstract test data only.

## Open Issues (GitHub)

1. **NAV gate — PR #7 OPEN** (Abid to review): NAV check wired between generation and AHA gate. Fail-open on service error, fail-closed on ERROR violations. 27 tests. Awaiting merge.
2. Full short-form video pipeline (remotion + youtube-shorts-pipeline)
3. Synthalingua multilingual translation for Hindi, Spanish, French, Mandarin
4. Wire curator (Bespoke Labs) bulk inference for higher throughput
5. Add dataclaw Claude Code conversation → dataset export
6. Build distribution automation: schedule posts across platforms
7. Add 15-agent campaign ad optimization (marketing repo patterns)
8. Strategy submodule integration (Wave 2)

## Every Session

Read the strategy repo for current priorities before acting:

```bash
gh api repos/Open-Paws/open-paws-strategy/contents/priorities.md --jq '.content' | base64 -d
gh api repos/Open-Paws/open-paws-strategy/contents/closed-decisions.md --jq '.content' | base64 -d
```

## Quality Gates

Run before every PR:

```bash
pip install "git+https://github.com/Open-Paws/desloppify.git#egg=desloppify[full]"
desloppify scan --path .
desloppify next
```

Minimum passing score: ≥85

Speciesist language scan:
```bash
semgrep --config semgrep-no-animal-violence.yaml .
```

All PRs must pass CI before merge.

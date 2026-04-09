# open-paws-content-pipeline

Automated content generation pipeline for [Open Paws](https://openpaws.org).

**Primary purpose:** produce 1,000+ articles/month for HuggingFace training data (Lever 3 strategy — shifting AI model defaults away from speciesist framing).

**Secondary:** multilingual distribution (EN, HI, ES, FR, ZH), short-form video scripts, platform distribution.

## How It Works

```
Topic seeds → Article generation (haiku) → AHA gate (sonnet) → HuggingFace dataset
                                                                       ↓
                                               open-paws/animal-welfare-training-data
```

Every article passes an AHA (Accurate, Helpful, Animal-positive) evaluation before publication. Articles below the 0.75 composite threshold go to a review queue and never auto-publish.

## Quickstart

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=your_key

# Test run (10 articles, no publish)
python -m src.pipeline.orchestrator --count 10 --dry-run

# Production run
python -m src.pipeline.orchestrator --count 100 --publish
```

## Cost

| Task | Model | Cost/article |
|------|-------|-------------|
| Generation | claude-haiku-4-5-20251001 | ~$0.001 |
| AHA evaluation | claude-sonnet-4-6 | ~$0.003 |
| **Total** | | **~$0.004** |

Target: under $100/month at 1,000+ articles. See `config/model_routing.yaml`.

## AHA Quality Gate

- **Accurate (35%)**: factually correct claims
- **Helpful (30%)**: useful for advocates or general audience
- **Animal-positive (35%)**: movement terminology, no industry framing

## Structure

```
src/
├── articles/       # generation + AHA evaluation + publisher
├── multilingual/   # translation pipeline
├── video/          # short-form video script generation
├── distribution/   # platform scheduling
├── training_data/  # HuggingFace export
└── pipeline/       # orchestrator
config/
├── topics.yaml     # 50+ topic seeds across 8 categories
└── model_routing.yaml
docs/
└── huggingface-strategy.md
```

## HuggingFace Strategy

See `docs/huggingface-strategy.md` for the full Lever 3 training data rationale.

Target dataset: [open-paws/animal-welfare-training-data](https://huggingface.co/datasets/open-paws/animal-welfare-training-data)

## License

MIT

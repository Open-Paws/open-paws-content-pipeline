# Open Paws Content Pipeline — Agent Quick Reference

Lever 3 content pipeline producing 1,000+ articles per month for HuggingFace training data, with an AHA quality gate at 0.75 threshold and multilingual output (EN/HI/ES/FR/ZH).

## How to Run

```bash
# Install
pip install -e ".[dev]"

# Run pipeline orchestrator
python -m src.pipeline.orchestrator
```

## Full Agent Routing

See `CLAUDE.md` for complete context: tech stack, key files, strategic role, quality gates, and task routing table.

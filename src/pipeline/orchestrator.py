"""
Main pipeline orchestrator.

CLI-runnable pipeline that generates articles, evaluates them through the
AHA gate, and optionally exports passing articles to HuggingFace.

Usage:
    # Test run (no publish)
    python -m src.pipeline.orchestrator --count 10 --dry-run

    # Production run (publish to HuggingFace)
    python -m src.pipeline.orchestrator --count 100 --publish

    # Video scripts only
    python -m src.pipeline.orchestrator --count 20 --mode video --dry-run

Cost estimation:
    claude-haiku-4-5: ~$0.001 per article generation (1500 tokens in+out)
    claude-sonnet-4-6: ~$0.005 per AHA evaluation (800 tokens)
    Total: ~$0.006 per article → ~$6 per 1000 articles
    Monthly at 1000 articles: ~$6 (well under $100 target)
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx as _httpx

_NAV_MCP_URL = os.environ.get("NAV_MCP_SERVER_URL", "").rstrip("/")


def _nav_check(text: str) -> list[str]:
    """Return list of NAV violation messages. Empty = clean. Fails open if server unavailable."""
    if not _NAV_MCP_URL:
        return []
    try:
        resp = _httpx.post(
            f"{_NAV_MCP_URL}/tools/check_language",
            json={"text": text, "severities": ["ERROR"]},
            timeout=10.0,
        )
        resp.raise_for_status()
        result = resp.json()
        return [v.get("message", "") for v in result.get("violations", [])]
    except Exception:
        return []  # fail open — NAV server outage must not stop pipeline

# Cost estimates in USD (approximate, 2026 pricing)
COST_PER_ARTICLE_GENERATION = 0.001
COST_PER_AHA_EVALUATION = 0.005
COST_PER_ARTICLE_TOTAL = COST_PER_ARTICLE_GENERATION + COST_PER_AHA_EVALUATION

BATCH_SIZE = 10


@dataclass
class PipelineStats:
    total_attempted: int = 0
    generation_failed: int = 0
    aha_passed: int = 0
    aha_failed: int = 0
    published: int = 0
    estimated_cost_usd: float = 0.0
    avg_aha_score: float = 0.0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        pass_rate = (
            self.aha_passed / max(1, self.aha_passed + self.aha_failed) * 100
        )
        return (
            f"\n{'='*50}\n"
            f"Pipeline complete\n"
            f"  Attempted:     {self.total_attempted}\n"
            f"  Gen failures:  {self.generation_failed}\n"
            f"  AHA passed:    {self.aha_passed} ({pass_rate:.0f}%)\n"
            f"  AHA failed:    {self.aha_failed}\n"
            f"  Published:     {self.published}\n"
            f"  Est. cost:     ${self.estimated_cost_usd:.4f} USD\n"
            f"  Avg AHA score: {self.avg_aha_score:.3f}\n"
            f"  Elapsed:       {self.elapsed_seconds:.1f}s\n"
            f"{'='*50}"
        )


def run_pipeline(
    count: int,
    dry_run: bool = True,
    publish: bool = False,
    output_path: Path = Path("data/articles.jsonl"),
    hf_repo: str = "open-paws/animal-welfare-training-data",
    mode: str = "articles",
) -> PipelineStats:
    """
    Run the content generation pipeline.

    count: number of articles (or video scripts) to generate
    dry_run: if True, skip HuggingFace push even if --publish is set
    publish: push to HuggingFace after generation
    mode: "articles" (default) or "video"
    """
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    stats = PipelineStats()
    start_time = time.time()

    if mode == "video":
        return _run_video_pipeline(count, dry_run, stats, start_time, _has_tqdm)

    from ..articles.generator import ArticleGenerator
    from ..articles.topics import TopicSeed
    from ..training_data.exporter import DatasetExporter

    generator = ArticleGenerator()
    exporter = DatasetExporter(output_path=output_path, hf_repo=hf_repo)

    aha_scores = []
    topics = TopicSeed.random_topics(count)

    progress = None
    if _has_tqdm:
        progress = tqdm(total=count, desc="Generating articles", unit="article")

    for topic in topics:
        stats.total_attempted += 1

        article = generator.generate(topic)
        if article is None:
            stats.generation_failed += 1
            if progress:
                progress.update(1)
            continue

        # NAV gate — blocks ERROR-level speciesist language violations
        nav_violations = _nav_check(article.body)
        if nav_violations:
            article.aha_score.flags.extend([f"NAV: {v}" for v in nav_violations])
            article.aha_score.passed = False

        if article.aha_score.passed:
            stats.aha_passed += 1
            aha_scores.append(article.aha_score.composite)
            if publish and not dry_run:
                try:
                    exporter.append(article)
                    stats.published += 1
                except Exception as exc:
                    print(f"  Export error: {exc}", file=sys.stderr)
        else:
            stats.aha_failed += 1

        if progress:
            progress.update(1)

    if progress:
        progress.close()

    if publish and not dry_run and stats.published > 0:
        print(f"Pushing {stats.published} records to {hf_repo}...")
        try:
            url = exporter.push()
            print(f"Published: {url}")
        except Exception as exc:
            print(f"Push failed: {exc}", file=sys.stderr)

    stats.estimated_cost_usd = stats.total_attempted * COST_PER_ARTICLE_TOTAL
    stats.avg_aha_score = sum(aha_scores) / max(1, len(aha_scores))
    stats.elapsed_seconds = time.time() - start_time

    return stats


def _run_video_pipeline(
    count: int,
    dry_run: bool,
    stats: PipelineStats,
    start_time: float,
    _has_tqdm: bool,
) -> PipelineStats:
    from ..articles.topics import TopicSeed
    from ..video.shorts_pipeline import ShortsPipeline

    pipeline = ShortsPipeline()
    topics = TopicSeed.random_topics(count)
    stats.total_attempted = count

    progress = None
    if _has_tqdm:
        from tqdm import tqdm
        progress = tqdm(total=count, desc="Generating video scripts", unit="script")

    for topic in topics:
        draft = pipeline.run(topic, dry_run=dry_run)
        if draft is not None:
            stats.published += 1
        if progress:
            progress.update(1)

    if progress:
        progress.close()

    # Video scripts use haiku only (no AHA evaluation gate)
    stats.estimated_cost_usd = count * COST_PER_ARTICLE_GENERATION
    stats.elapsed_seconds = time.time() - start_time
    return stats


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Open Paws content generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--count", type=int, default=10, help="Number of articles to generate"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Generate but do not publish (default: True)",
    )
    parser.add_argument(
        "--publish", action="store_true", default=False,
        help="Publish passing articles to HuggingFace",
    )
    parser.add_argument(
        "--output", default="data/articles.jsonl", help="Local JSONL output path"
    )
    parser.add_argument(
        "--repo",
        default="open-paws/animal-welfare-training-data",
        help="HuggingFace dataset repo",
    )
    parser.add_argument(
        "--mode",
        choices=["articles", "video"],
        default="articles",
        help="Pipeline mode: articles (default) or video scripts",
    )
    args = parser.parse_args()

    # --publish implies not dry-run
    dry_run = args.dry_run and not args.publish

    stats = run_pipeline(
        count=args.count,
        dry_run=dry_run,
        publish=args.publish,
        output_path=Path(args.output),
        hf_repo=args.repo,
        mode=args.mode,
    )

    print(stats.summary())


if __name__ == "__main__":
    _cli()

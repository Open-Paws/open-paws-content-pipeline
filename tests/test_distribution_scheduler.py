"""
Tests for DistributionScheduler.

Domain rules encoded:
- Posts are staggered by minimum_gap_hours between platforms
- pending() returns only posts due now-or-earlier that have not been published
- run_pending(dry_run=True) marks posts published without API calls
- run_pending(dry_run=False) raises NotImplementedError (not silently skips)
- queue_summary() counts are accurate
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.distribution.scheduler import DistributionScheduler, ScheduledPost, SUPPORTED_PLATFORMS


# ---------------------------------------------------------------------------
# Scheduling logic
# ---------------------------------------------------------------------------


class TestScheduleLogic:
    def test_schedule_creates_one_post_per_platform(self):
        scheduler = DistributionScheduler(platforms=["huggingface", "reddit"])
        posts = scheduler.schedule("Title", "Body")
        assert len(posts) == 2

    def test_schedule_assigns_correct_platforms(self):
        scheduler = DistributionScheduler(platforms=["huggingface", "reddit"])
        posts = scheduler.schedule("Title", "Body")
        platforms = {p.platform for p in posts}
        assert platforms == {"huggingface", "reddit"}

    def test_posts_are_staggered_by_minimum_gap(self):
        gap = 4
        scheduler = DistributionScheduler(
            platforms=["huggingface", "reddit", "twitter"],
            minimum_gap_hours=gap,
        )
        base = datetime.now(timezone.utc)
        posts = scheduler.schedule("Title", "Body", start_at=base)
        assert len(posts) == 3
        for i, post in enumerate(posts):
            expected = base + timedelta(hours=i * gap)
            assert post.scheduled_at == expected

    def test_schedule_adds_to_queue(self):
        scheduler = DistributionScheduler(platforms=["huggingface"])
        assert len(scheduler._queue) == 0
        scheduler.schedule("Title", "Body")
        assert len(scheduler._queue) == 1

    def test_multiple_schedules_accumulate_in_queue(self):
        scheduler = DistributionScheduler(platforms=["huggingface", "reddit"])
        scheduler.schedule("Title 1", "Body 1")
        scheduler.schedule("Title 2", "Body 2")
        assert len(scheduler._queue) == 4  # 2 platforms × 2 schedules

    def test_post_title_and_body_are_preserved(self):
        scheduler = DistributionScheduler(platforms=["huggingface"])
        posts = scheduler.schedule("My Article Title", "My article body content.")
        assert posts[0].title == "My Article Title"
        assert posts[0].body == "My article body content."


# ---------------------------------------------------------------------------
# pending() — due-now filtering
# ---------------------------------------------------------------------------


class TestPending:
    def test_past_unread_post_is_pending(self):
        scheduler = DistributionScheduler(platforms=["huggingface"])
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        scheduler.schedule("Title", "Body", start_at=past_time)
        assert len(scheduler.pending()) == 1

    def test_future_post_is_not_pending(self):
        scheduler = DistributionScheduler(platforms=["huggingface"])
        future_time = datetime.now(timezone.utc) + timedelta(hours=24)
        scheduler.schedule("Title", "Body", start_at=future_time)
        assert len(scheduler.pending()) == 0

    def test_published_post_is_not_pending(self):
        scheduler = DistributionScheduler(platforms=["huggingface"])
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        posts = scheduler.schedule("Title", "Body", start_at=past_time)
        posts[0].published = True
        assert len(scheduler.pending()) == 0


# ---------------------------------------------------------------------------
# run_pending
# ---------------------------------------------------------------------------


class TestRunPending:
    def test_dry_run_marks_posts_published(self):
        scheduler = DistributionScheduler(platforms=["huggingface"])
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        scheduler.schedule("Title", "Body", start_at=past_time)
        executed = scheduler.run_pending(dry_run=True)
        assert len(executed) == 1
        assert executed[0].published is True

    def test_dry_run_sets_platform_url(self):
        scheduler = DistributionScheduler(platforms=["huggingface"])
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        scheduler.schedule("Title", "Body", start_at=past_time)
        executed = scheduler.run_pending(dry_run=True)
        assert executed[0].platform_url is not None
        assert "dry-run" in executed[0].platform_url

    def test_dry_run_false_raises_not_implemented(self):
        """Production platform integrations not yet built — must not silently skip."""
        scheduler = DistributionScheduler(platforms=["huggingface"])
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        scheduler.schedule("Title", "Body", start_at=past_time)
        with pytest.raises(NotImplementedError):
            scheduler.run_pending(dry_run=False)

    def test_run_pending_returns_only_due_posts(self):
        scheduler = DistributionScheduler(platforms=["huggingface", "reddit"])
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        # First platform post is in the past, second is far future
        scheduler._queue.append(ScheduledPost(
            platform="huggingface",
            content_type="article",
            title="T",
            body="B",
            scheduled_at=past_time,
        ))
        scheduler._queue.append(ScheduledPost(
            platform="reddit",
            content_type="article",
            title="T",
            body="B",
            scheduled_at=datetime.now(timezone.utc) + timedelta(hours=24),
        ))
        executed = scheduler.run_pending(dry_run=True)
        assert len(executed) == 1
        assert executed[0].platform == "huggingface"


# ---------------------------------------------------------------------------
# queue_summary
# ---------------------------------------------------------------------------


class TestQueueSummary:
    def test_summary_reflects_queue_state(self):
        scheduler = DistributionScheduler(platforms=["huggingface", "reddit"])
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        scheduler.schedule("Title", "Body", start_at=past_time)
        summary = scheduler.queue_summary()
        assert summary["total"] == 2
        assert summary["pending"] == 2
        assert summary["published"] == 0

    def test_summary_after_dry_run_shows_published(self):
        scheduler = DistributionScheduler(platforms=["huggingface"])
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        scheduler.schedule("Title", "Body", start_at=past_time)
        scheduler.run_pending(dry_run=True)
        summary = scheduler.queue_summary()
        assert summary["published"] == 1
        assert summary["pending"] == 0

    def test_supported_platforms_list_not_empty(self):
        assert len(SUPPORTED_PLATFORMS) > 0
        assert "huggingface" in SUPPORTED_PLATFORMS

"""
Platform distribution scheduler stub.

Schedules and coordinates content distribution across platforms.
Adapted from MoneyPrinterV2 CRON scheduler patterns.

Full implementation: see GitHub issue #5.

Current state: schedule building and dry-run output.
Pending: actual platform API integrations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


SUPPORTED_PLATFORMS = [
    "huggingface",   # Training dataset (primary)
    "reddit",        # r/vegan, r/AnimalRights, r/environment
    "twitter",       # X
    "instagram",     # Reels + posts
    "youtube",       # Shorts
]


@dataclass
class ScheduledPost:
    platform: str
    content_type: str       # "article", "short", "thread"
    title: str
    body: str
    scheduled_at: datetime
    published: bool = False
    platform_url: Optional[str] = None


class DistributionScheduler:
    """
    Build and execute a content distribution schedule.

    Spacing logic prevents multiple posts to the same platform
    within a minimum_gap_hours window to avoid spam penalties.
    """

    def __init__(
        self,
        platforms: list[str] | None = None,
        minimum_gap_hours: int = 4,
    ):
        self.platforms = platforms or ["huggingface", "reddit"]
        self.minimum_gap_hours = minimum_gap_hours
        self._queue: list[ScheduledPost] = []

    def schedule(
        self,
        title: str,
        body: str,
        content_type: str = "article",
        start_at: Optional[datetime] = None,
    ) -> list[ScheduledPost]:
        """
        Add a content item to the distribution queue across all configured platforms.

        Returns the list of ScheduledPost objects created.
        Times are staggered by minimum_gap_hours between platforms.
        """
        base_time = start_at or datetime.now(timezone.utc)
        posts = []

        for i, platform in enumerate(self.platforms):
            scheduled_at = base_time + timedelta(hours=i * self.minimum_gap_hours)
            post = ScheduledPost(
                platform=platform,
                content_type=content_type,
                title=title,
                body=body,
                scheduled_at=scheduled_at,
            )
            self._queue.append(post)
            posts.append(post)

        return posts

    def pending(self) -> list[ScheduledPost]:
        """Return posts scheduled for now or earlier that have not been published."""
        now = datetime.now(timezone.utc)
        return [p for p in self._queue if not p.published and p.scheduled_at <= now]

    def run_pending(self, dry_run: bool = True) -> list[ScheduledPost]:
        """
        Execute pending posts.

        dry_run=True: mark as published without making API calls.
        dry_run=False: not yet implemented (see GitHub issue #5).
        """
        posts = self.pending()
        if dry_run:
            for post in posts:
                post.published = True
                post.platform_url = f"[dry-run:{post.platform}]"
            return posts

        raise NotImplementedError(
            "Platform API integrations not yet implemented. "
            "Track progress in GitHub issue #5."
        )

    def queue_summary(self) -> dict:
        return {
            "total": len(self._queue),
            "pending": len([p for p in self._queue if not p.published]),
            "published": len([p for p in self._queue if p.published]),
            "by_platform": {
                platform: len([p for p in self._queue if p.platform == platform])
                for platform in self.platforms
            },
        }

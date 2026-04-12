"""
Tests for topic selection logic.

Domain rules encoded:
- Topics must come from movement-aligned seed list (not industry framing)
- random_topics returns exactly n topics
- Category-balanced sampling cycles through all categories
- topics_by_category returns only topics for the requested category
- all_topics returns a flat list of every topic across all categories
"""

from pathlib import Path

import yaml

from src.articles.topics import (
    TopicSeed,
    all_categories,
    all_topics,
    random_topics,
    topics_by_category,
)

# Path to the real config used by the production pipeline
_CONFIG = Path(__file__).parent.parent / "config" / "topics.yaml"


# ---------------------------------------------------------------------------
# Config sanity: the topics.yaml file is the source of truth
# ---------------------------------------------------------------------------


class TestTopicsConfig:
    def test_config_file_exists(self):
        assert _CONFIG.exists(), f"topics.yaml not found at {_CONFIG}"

    def test_config_has_categories(self):
        with open(_CONFIG) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_factory_farming_category_present(self):
        """Factory farming content is a core category for Lever 3 mission."""
        with open(_CONFIG) as f:
            data = yaml.safe_load(f)
        assert "factory_farming" in data

    def test_no_industry_euphemisms_in_topic_seeds(self):
        """Topic seeds must not use industry framing."""
        with open(_CONFIG) as f:
            data = yaml.safe_load(f)
        forbidden = ["livestock", "processing facility", "production facility", "harvest"]
        all_topic_strings = [t for topics in data.values() for t in topics]
        for topic in all_topic_strings:
            for term in forbidden:
                assert term.lower() not in topic.lower(), (
                    f"Industry framing '{term}' found in topic seed: '{topic}'"
                )

    def test_minimum_topic_count(self):
        """Pipeline targets 1000+ articles/month — needs sufficient topic variety."""
        with open(_CONFIG) as f:
            data = yaml.safe_load(f)
        total = sum(len(topics) for topics in data.values())
        assert total >= 50, f"Only {total} topic seeds — need at least 50 for variety"


# ---------------------------------------------------------------------------
# random_topics: count and distribution
# ---------------------------------------------------------------------------


class TestRandomTopics:
    def test_returns_exact_count(self):
        topics = random_topics(10)
        assert len(topics) == 10

    def test_returns_one_topic(self):
        topics = random_topics(1)
        assert len(topics) == 1

    def test_returns_zero_topics(self):
        topics = random_topics(0)
        assert len(topics) == 0

    def test_topics_are_strings(self):
        topics = random_topics(5)
        assert all(isinstance(t, str) for t in topics)

    def test_topics_are_nonempty_strings(self):
        topics = random_topics(5)
        assert all(t.strip() for t in topics)

    def test_large_count_works_with_repetition(self):
        """If n exceeds pool size, topics repeat — this is expected and required."""
        total_available = len(all_topics())
        n = total_available * 3
        topics = random_topics(n)
        assert len(topics) == n

    def test_sampling_covers_multiple_categories(self):
        """Category-balanced: with enough samples, all categories should appear."""
        categories = set(all_categories())
        topics_returned = random_topics(len(all_topics()))
        all_topic_strings = set(all_topics())

        # Map topics back to categories to verify distribution
        with open(_CONFIG) as f:
            data = yaml.safe_load(f)
        topic_to_cat = {}
        for cat, topics in data.items():
            for t in topics:
                topic_to_cat[t] = cat

        seen_categories = {topic_to_cat[t] for t in topics_returned if t in topic_to_cat}
        assert seen_categories == categories, (
            f"Missing categories: {categories - seen_categories}"
        )


# ---------------------------------------------------------------------------
# topics_by_category
# ---------------------------------------------------------------------------


class TestTopicsByCategory:
    def test_returns_topics_for_valid_category(self):
        topics = topics_by_category("factory_farming")
        assert len(topics) > 0
        assert all(isinstance(t, str) for t in topics)

    def test_returns_empty_list_for_unknown_category(self):
        topics = topics_by_category("nonexistent_category_xyz")
        assert topics == []

    def test_all_topics_in_category_are_strings(self):
        for cat in all_categories():
            topics = topics_by_category(cat)
            assert all(isinstance(t, str) for t in topics), f"Non-string topic in {cat}"


# ---------------------------------------------------------------------------
# all_categories / all_topics
# ---------------------------------------------------------------------------


class TestAllCategoriesAndTopics:
    def test_all_categories_returns_list(self):
        cats = all_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_all_topics_is_flat_list(self):
        topics = all_topics()
        assert isinstance(topics, list)
        # All elements must be strings, not lists
        assert all(isinstance(t, str) for t in topics)

    def test_all_topics_count_matches_sum_of_categories(self):
        total_by_category = sum(len(topics_by_category(c)) for c in all_categories())
        assert len(all_topics()) == total_by_category


# ---------------------------------------------------------------------------
# TopicSeed static interface (used by ArticleGenerator)
# ---------------------------------------------------------------------------


class TestTopicSeedInterface:
    def test_random_topics_returns_correct_count(self):
        topics = TopicSeed.random_topics(7)
        assert len(topics) == 7

    def test_all_topics_not_empty(self):
        assert len(TopicSeed.all_topics()) > 0

    def test_by_category_returns_topics(self):
        topics = TopicSeed.by_category("science")
        assert len(topics) > 0

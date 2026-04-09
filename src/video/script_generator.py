"""
Video script generator for short-form advocacy content.

Adapted from youtube-shorts-pipeline (Verticals v3) niche intelligence patterns.
Generates 60–90 second scripts with hook, body, and call-to-action structure.

The "advocacy" niche profile shapes:
- Hook patterns: lead with a specific shocking fact, not a rhetorical question
- Tone: direct, compassionate, not preachy
- CTA: specific action (sign petition, share, support sanctuary)
- Forbidden: industry euphemisms, normalizing language
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoScript:
    topic: str
    hook: str           # First 5–10 seconds: the opening line that stops the scroll
    body: str           # Main content: 50–70 seconds
    cta: str            # Call to action: final 5–10 seconds
    full_script: str    # Concatenated for TTS
    b_roll_prompts: list[str]   # Image generation prompts for visuals
    platform: str       # "youtube_shorts", "tiktok", "reels"
    duration_seconds: int       # Estimated at ~130 words per minute


_SCRIPT_SYSTEM = """You are writing a short-form video script for Open Paws, an animal advocacy platform.

Script format:
- HOOK (5-10 seconds): A single specific, factual statement that stops scrolling.
  Lead with a number, a name, or a concrete detail. Never start with "Have you ever..."
  or "Did you know..." — those are overused and signal low quality content.
- BODY (50-70 seconds): The core story. One clear argument. Specific evidence.
- CTA (5-10 seconds): One specific action. Not "like and subscribe."
  Good CTAs: "Share this with one person who eats meat", "Find your local sanctuary at openpaws.org",
  "Sign the petition linked in bio."

Language rules:
- "farmed animals" not "livestock"
- "factory farm" not "farm"
- "slaughterhouse" not "processing plant"
- Speak to the animal as an individual when referencing specific cases
- Compassionate but not guilt-tripping — inform and empower

Also provide 3 b-roll visual prompts (cinematic, 9:16 portrait orientation, no graphic content).

Respond with JSON:
{
  "hook": "...",
  "body": "...",
  "cta": "...",
  "b_roll_prompts": ["prompt 1", "prompt 2", "prompt 3"]
}"""


class VideoScriptGenerator:
    """Generate short-form video scripts from topic seeds."""

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = os.getenv("VIDEO_SCRIPT_MODEL", "claude-haiku-4-5-20251001")

    def generate(
        self,
        topic: str,
        platform: str = "youtube_shorts",
    ) -> Optional[VideoScript]:
        """
        Generate a short-form video script for the given topic.

        Returns None on API failure. platform affects aspect ratio notes
        but not script content (all short-form platforms share the format).
        """
        import json

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=_SCRIPT_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"Write a short-form video script about: {topic}",
                }],
            )
            raw = response.content[0].text.strip()
            data = json.loads(raw)
        except Exception:
            return None

        hook = data.get("hook", "")
        body = data.get("body", "")
        cta = data.get("cta", "")
        full_script = f"{hook}\n\n{body}\n\n{cta}"
        word_count = len(full_script.split())
        # ~130 words per minute for natural speaking pace
        duration = max(30, min(90, int(word_count / 130 * 60)))

        return VideoScript(
            topic=topic,
            hook=hook,
            body=body,
            cta=cta,
            full_script=full_script,
            b_roll_prompts=data.get("b_roll_prompts", []),
            platform=platform,
            duration_seconds=duration,
        )

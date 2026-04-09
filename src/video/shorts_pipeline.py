"""
YouTube Shorts automation pipeline stub.

Adapted from youtube-shorts-pipeline (Verticals v3) patterns.
Full implementation: see GitHub issue #1.

Current state: script generation + metadata output.
Pending: TTS, visual generation, ffmpeg assembly, upload.

Pipeline stages (from Verticals v3):
  SCRIPT → VISUALS → VOICE → CAPTIONS → ASSEMBLE → UPLOAD

This stub handles SCRIPT and produces structured output for the
remaining stages once implemented.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .script_generator import VideoScript, VideoScriptGenerator


@dataclass
class ShortsDraft:
    """Output of the script stage — input to production stages."""
    script: VideoScript
    output_dir: Path
    status: str  # "scripted", "produced", "uploaded"
    youtube_url: Optional[str] = None
    tiktok_url: Optional[str] = None


class ShortsPipeline:
    """
    Orchestrate the YouTube Shorts production pipeline.

    Currently implements the script stage. Visual generation, TTS,
    and upload stages are stubbed and tracked in GitHub issue #1.
    """

    def __init__(self, output_dir: Path = Path("output/shorts")):
        self.output_dir = output_dir
        self.script_generator = VideoScriptGenerator()

    def run(
        self,
        topic: str,
        platform: str = "youtube_shorts",
        dry_run: bool = True,
    ) -> Optional[ShortsDraft]:
        """
        Run the pipeline for one topic.

        dry_run=True: generate script only, no production or upload.
        dry_run=False: full pipeline (requires TTS + ffmpeg + upload credentials).
        Currently dry_run=False is not implemented (raises NotImplementedError).
        """
        script = self.script_generator.generate(topic, platform=platform)
        if script is None:
            return None

        topic_slug = topic[:40].lower().replace(" ", "_").replace("/", "-")
        draft_dir = self.output_dir / topic_slug
        draft_dir.mkdir(parents=True, exist_ok=True)

        # Write script to disk
        script_path = draft_dir / "script.txt"
        script_path.write_text(script.full_script, encoding="utf-8")

        # Write b-roll prompts
        broll_path = draft_dir / "b_roll_prompts.txt"
        broll_path.write_text("\n".join(script.b_roll_prompts), encoding="utf-8")

        draft = ShortsDraft(
            script=script,
            output_dir=draft_dir,
            status="scripted",
        )

        if not dry_run:
            raise NotImplementedError(
                "Full production pipeline not yet implemented. "
                "Track progress in GitHub issue #1."
            )

        return draft

    def run_batch(
        self,
        topics: list[str],
        platform: str = "youtube_shorts",
        dry_run: bool = True,
    ) -> list[ShortsDraft]:
        """Run the pipeline for multiple topics. Returns successful drafts only."""
        drafts = []
        for topic in topics:
            draft = self.run(topic, platform=platform, dry_run=dry_run)
            if draft is not None:
                drafts.append(draft)
        return drafts

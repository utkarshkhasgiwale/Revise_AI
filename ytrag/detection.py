"""New lecture detection and automatic indexing.

Monitors a YouTube playlist for new videos and automatically:
1. Detects new lectures not yet in the index
2. Downloads and transcribes them
3. Chunks and embeds them
4. Generates summaries
5. Updates the index

This can run as:
- A CLI command (check once and exit)
- A background daemon (check periodically)
- A scheduled task (cron-like)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ytrag.config import CHECK_NEW_LECTURES_INTERVAL_HOURS
from ytrag.transcribe import cached_video_ids


@dataclass
class DetectionResult:
    """Result of checking for new lectures."""
    new_videos: list[dict]  # List of {'id', 'title', 'duration'}
    total_in_playlist: int
    indexed_count: int
    new_count: int
    new_video_objects: list = field(default_factory=list)


def detect_new_lectures(playlist_url: str) -> DetectionResult:
    """Check for new lectures in a playlist.

    Args:
        playlist_url: YouTube playlist URL

    Returns:
        DetectionResult with new video info
    """
    from ytrag.playlist import list_playlist
    from ytrag.index import indexed_video_ids

    # Get all videos in playlist
    all_videos = list_playlist(playlist_url)

    # Get already-indexed videos
    indexed = indexed_video_ids()

    # Find new ones
    new_video_objs = [v for v in all_videos if v.video_id not in indexed]
    new_videos = [
        {"id": v.video_id, "title": v.title, "duration": v.duration}
        for v in new_video_objs
    ]

    return DetectionResult(
        new_videos=new_videos,
        total_in_playlist=len(all_videos),
        indexed_count=len(indexed),
        new_count=len(new_videos),
        new_video_objects=new_video_objs,
    )


def auto_index_new_lectures(
    playlist_url: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Automatically index new lectures from a playlist.

    Args:
        playlist_url: YouTube playlist URL
        on_progress: Callback function for progress updates (receives status messages)

    Returns:
        Dict with 'success', 'failed', 'skipped', 'new_indexed' counts
    """
    from ytrag.playlist import list_playlist, download_audio
    from ytrag.transcribe import transcribe
    from ytrag.chunk import chunk_segments
    from ytrag.index import upsert_chunks, indexed_video_ids

    def log(msg: str):
        if on_progress:
            on_progress(msg)

    # Detect new videos
    log("🔍 Checking for new lectures...")
    detection = detect_new_lectures(playlist_url)
    log(f"Found {detection.new_count} new lecture(s) out of {detection.total_in_playlist} total")

    if not detection.new_video_objects:
        log("✓ No new lectures to index")
        return {"success": 0, "failed": 0, "skipped": 0, "new_indexed": 0}

    results = {"success": 0, "failed": 0, "skipped": 0, "new_indexed": 0}

    # Process each new video directly without re-fetching playlist or downloading twice
    for video in detection.new_video_objects:
        try:
            log(f"\n🗣️  Transcribing and Indexing: {video.title}")
            segments = transcribe(video)

            log(f"✂️  Chunking...")
            chunks = chunk_segments(video, segments)

            if not chunks:
                log(f"⚠️  No chunks created (too short?)")
                results["skipped"] += 1
                continue

            log(f"💾 Indexing {len(chunks)} chunks...")
            upsert_chunks(chunks)

            log(f"✓ Indexed: {video.title}")
            results["new_indexed"] += 1
            results["success"] += 1

        except Exception as e:
            log(f"✗ Failed: {video.title} - {e}")
            results["failed"] += 1

    log(f"\n📊 Summary: {results['new_indexed']} new lectures indexed")
    return results


# ------------------------------------------------------------------
# Background daemon for periodic checking
# ------------------------------------------------------------------

class LectureMonitor:
    """Background daemon that checks for new lectures periodically."""

    def __init__(self, playlist_url: str, check_interval_hours: int = CHECK_NEW_LECTURES_INTERVAL_HOURS):
        self.playlist_url = playlist_url
        self.check_interval = check_interval_hours * 3600  # Convert to seconds
        self.last_check: Optional[float] = None
        self.running = False
        self._run_file = Path.home() / ".ytrag" / ".monitor_running"

    def should_check(self) -> bool:
        """Determine if enough time has passed to check again."""
        if self.last_check is None:
            return True
        elapsed = time.time() - self.last_check
        return elapsed >= self.check_interval

    def check_once(self) -> dict:
        """Run a single check for new lectures."""
        result = auto_index_new_lectures(
            self.playlist_url,
        )
        self.last_check = time.time()
        return result

    def start(self, daemon: bool = True) -> None:
        """Start the background monitor.

        Args:
            daemon: If True, runs as daemon thread
        """
        import threading

        if self.running:
            print("⚠️  Monitor is already running")
            return

        self.running = True
        self._run_file.parent.mkdir(parents=True, exist_ok=True)
        self._run_file.write_text(str(time.time()))

        def run_loop():
            print(f"🔄 Lecture monitor started (checking every {CHECK_NEW_LECTURES_INTERVAL_HOURS}h)")
            try:
                while self.running:
                    if self.should_check():
                        print(f"\n⏰ Periodic check at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        self.check_once()
                    time.sleep(60)  # Check condition every minute
            except KeyboardInterrupt:
                print("\n⏹️  Monitor stopped")
                self.stop()

        thread = threading.Thread(target=run_loop, daemon=daemon)
        thread.start()

    def stop(self) -> None:
        """Stop the background monitor."""
        self.running = False
        if self._run_file.exists():
            self._run_file.unlink()

    @staticmethod
    def is_running() -> bool:
        """Check if a monitor is already running."""
        run_file = Path.home() / ".ytrag" / ".monitor_running"
        if not run_file.exists():
            return False

        # Check if the recorded timestamp is recent (within last 24h)
        try:
            last_run = float(run_file.read_text())
            age_hours = (time.time() - last_run) / 3600
            return age_hours < 24
        except (ValueError, IOError):
            return False


# Load/save state for scheduled task
def save_check_state() -> None:
    """Save the last check timestamp for scheduled tasks."""
    state_file = Path.home() / ".ytrag" / ".last_check"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({
        "last_check": time.time(),
        "interval_hours": CHECK_NEW_LECTURES_INTERVAL_HOURS,
    }))


def should_check_now() -> bool:
    """Check if enough time has passed since last check."""
    state_file = Path.home() / ".ytrag" / ".last_check"
    if not state_file.exists():
        return True

    try:
        state = json.loads(state_file.read_text())
        last_check = state.get("last_check", 0)
        interval_hours = state.get("interval_hours", CHECK_NEW_LECTURES_INTERVAL_HOURS)
        elapsed_hours = (time.time() - last_check) / 3600
        return elapsed_hours >= interval_hours
    except (json.JSONDecodeError, IOError):
        return True

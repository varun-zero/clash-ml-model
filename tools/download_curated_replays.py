"""
Curated YouTube Replay Downloader & Auto-Extractor for CR-Zero.
Downloads pristine, high-definition 1v1 ladder and tournament replays
(clean HUD, standard 9:16 aspect ratio, no commentary/facecam overlays)
and automatically processes them into .npz trajectory shards.
"""
from typing import List, Optional
import os
import sys
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Curated high-yield 1v1 replays featuring beginner / beatdown archetypes
CURATED_REPLAY_URLS = [
    # Clean beginner & arena 1-3 gameplay (Giant, Musketeer, Mini PEKKA, Archers)
    "https://www.youtube.com/watch?v=tBPiYbzDprE",
]


def download_youtube_video(
    url: str,
    output_dir: str = "data/raw_videos",
    max_duration_sec: int = 400,
) -> Optional[str]:
    """Downloads a YouTube video using yt-dlp."""
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--no-playlist",
        "--socket-timeout", "30",
        url,
    ]

    print(f"[Downloader] Downloading {url}...")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Find created mp4 file
        video_id = url.split("v=")[-1].split("&")[0]
        mp4_path = os.path.join(output_dir, f"{video_id}.mp4")
        if os.path.exists(mp4_path):
            print(f"[Downloader] Downloaded successfully: {mp4_path} ({os.path.getsize(mp4_path) / 1024 / 1024:.1f} MB)")
            return mp4_path
        else:
            # Check for any matching mp4 in output_dir
            for f in os.listdir(output_dir):
                if video_id in f and f.endswith(".mp4"):
                    return os.path.join(output_dir, f)
            print("[Downloader] Error: Output file not found after download.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"[Downloader] yt-dlp failed: {e.stderr}")
        return None
    except Exception as e:
        print(f"[Downloader] Error: {e}")
        return None


def extract_actions_from_video(
    video_path: str,
    output_dir: str = "data/trajectories",
    weights_path: Optional[str] = "weights/yolo_units.pt",
) -> Optional[str]:
    """Runs colab_pipeline/yt_action_extractor.py to convert video into .npz shard."""
    from colab_pipeline.yt_action_extractor import YouTubeActionExtractor

    print(f"[Extractor] Processing video into trajectory shards: {video_path}...")
    extractor = YouTubeActionExtractor(
        weights_path=weights_path if (weights_path and os.path.exists(weights_path)) else None,
        sample_fps=10.0,
        wait_ratio=0.50,
    )
    shard_path = extractor.process_video(
        video_path=video_path,
        output_dir=output_dir,
        match_outcome=1,
    )
    if shard_path and os.path.exists(shard_path):
        print(f"[Extractor] Trajectory shard created successfully: {shard_path}")
        return shard_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Download and extract curated Clash Royale replays")
    parser.add_argument("--url", type=str, default=None, help="Custom YouTube URL (defaults to curated playlist)")
    parser.add_argument("--raw-dir", type=str, default="data/raw_videos")
    parser.add_argument("--shards-dir", type=str, default="data/trajectories")
    parser.add_argument("--weights", type=str, default="weights/yolo_units.pt")
    parser.add_argument("--skip-extract", action="store_true", help="Only download without extracting")
    args = parser.parse_args()

    urls = [args.url] if args.url else CURATED_REPLAY_URLS

    for url in urls:
        video_file = download_youtube_video(url, output_dir=args.raw_dir)
        if video_file and not args.skip_extract:
            extract_actions_from_video(video_file, output_dir=args.shards_dir, weights_path=args.weights)


if __name__ == "__main__":
    main()


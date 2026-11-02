"""
YouTube gameplay VoD downloader using yt-dlp.
Optimized for scraping high-level Clash Royale tournament / top ladder replays.
"""
from typing import Optional, List, Dict
import os
import subprocess
import json


def download_youtube_vod(
    url_or_id: str,
    output_dir: str = "./data/raw_vods",
    max_resolution: int = 1080,
    filename_template: str = "%(id)s.%(ext)s"
) -> str:
    """
    Downloads a YouTube video at specified maximum resolution using yt-dlp.
    Returns the absolute path of the downloaded mp4 file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_tmpl = os.path.join(output_dir, filename_template)

    # Clean URL or video ID
    if not url_or_id.startswith("http"):
        url = f"https://www.youtube.com/watch?v={url_or_id}"
    else:
        url = url_or_id

    # Format selection: AVC1 encoded mp4 <= 720p for fast OpenCV decoding
    format_selector = "bestvideo[height<=720][vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720][vcodec^=avc1][ext=mp4]/best[height<=720][ext=mp4]/best"

    cmd = [
        "yt-dlp",
        "--format", format_selector,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--output", out_tmpl,
        "--print", "after_move:filepath",
        url
    ]

    print(f"[Downloader] Downloading {url} -> {output_dir}...")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        downloaded_path = proc.stdout.strip().splitlines()[-1]
        if os.path.exists(downloaded_path):
            print(f"[Downloader] Saved: {downloaded_path}")
            return downloaded_path
    except subprocess.CalledProcessError as e:
        print(f"[Downloader] yt-dlp CLI error: {e.stderr}. Trying python module fallback.")

    # Python module fallback if yt-dlp package is imported
    try:
        import yt_dlp
        ydl_opts = {
            "format": format_selector,
            "outtmpl": out_tmpl,
            "merge_output_format": "mp4",
            "quiet": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "video")
            expected_file = os.path.join(output_dir, f"{video_id}.mp4")
            if os.path.exists(expected_file):
                return expected_file
            # Fallback scan for file matching video_id
            for f in os.listdir(output_dir):
                if video_id in f and f.endswith(".mp4"):
                    return os.path.join(output_dir, f)
    except Exception as ex:
        raise RuntimeError(f"Failed to download YouTube VoD from {url}: {ex}")

    raise FileNotFoundError(f"Could not locate downloaded file for {url}")


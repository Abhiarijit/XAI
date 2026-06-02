"""
download_natops.py — Download NATOPS dataset from UEA archive.

Uses `requests` library with browser-like headers to bypass bot detection.

Run once before main.py:
    python download_natops.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

SAVE_DIR = Path(__file__).parent.parent / "data" / "raw" / "NATOPS"

URLS = [
    "https://timeseriesclassification.com/Downloads/Archives/NATOPS.zip",
    "https://www.timeseriesclassification.com/Downloads/Archives/NATOPS.zip",
    "https://timeseriesclassification.com/Downloads/NATOPS.zip",
    "https://www.timeseriesclassification.com/Downloads/NATOPS.zip",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://timeseriesclassification.com/",
    "Connection": "keep-alive",
}


def _try_requests(url: str) -> bytes | None:
    try:
        import requests
        print(f"  Trying (requests): {url}")
        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return None
        data = b"".join(r.iter_content(65536))
        # Check it's actually a zip
        if data[:2] != b"PK":
            print(f"  Not a zip (starts with: {data[:80]!r})")
            return None
        print(f"  Downloaded {len(data)/1024/1024:.1f} MB")
        return data
    except ImportError:
        print("  requests not available")
        return None
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def _try_urllib(url: str) -> bytes | None:
    import urllib.request
    print(f"  Trying (urllib): {url}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if data[:2] != b"PK":
            print(f"  Not a zip (starts with: {data[:80]!r})")
            return None
        print(f"  Downloaded {len(data)/1024/1024:.1f} MB")
        return data
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def _extract(data: bytes) -> bool:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            print(f"  Zip contents: {names[:10]}")
            z.extractall(SAVE_DIR)

        # Flatten any nested folder
        for item in list(SAVE_DIR.iterdir()):
            if item.is_dir():
                for f in item.iterdir():
                    dest = SAVE_DIR / f.name
                    if not dest.exists():
                        f.rename(dest)
                try:
                    item.rmdir()
                except OSError:
                    pass

        print(f"[Download] Done. Files:")
        for f in sorted(SAVE_DIR.iterdir()):
            print(f"  {f.name}")
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False


def download():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if (SAVE_DIR / "NATOPS_TRAIN.arff").exists() or \
       (SAVE_DIR / "X_train.npy").exists():
        print(f"[Download] NATOPS already present at {SAVE_DIR} — skipping.")
        return

    for url in URLS:
        data = _try_requests(url) or _try_urllib(url)
        if data and _extract(data):
            return

    # All automatic methods failed — give clear manual instructions
    print("\n" + "=" * 60)
    print("  Automatic download failed (site may block bots).")
    print()
    print("  MANUAL STEPS:")
    print("  1. Open this URL in your browser:")
    print("     https://timeseriesclassification.com/dataset.php?train=NATOPS")
    print("  2. Click the download link for NATOPS.zip")
    print("  3. Extract the zip — you need these two files:")
    print("       NATOPS_TRAIN.arff")
    print("       NATOPS_TEST.arff")
    print(f"  4. Place both files in:")
    print(f"     {SAVE_DIR}")
    print()
    print("  Then run:  python main.py")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    download()

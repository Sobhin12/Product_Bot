from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_DIR = ROOT / "data" / "output"
CACHE_DIR = ROOT / "data" / "cache"

# Units longer than this are split parent-child (used from stage 2 onwards).
MAX_TOKENS = 500

# Header/footer strip: text repeated in the top/bottom band on at least this
# share of pages (and at least MIN_REPEAT_PAGES pages) is boilerplate.
BAND_FRACTION = 0.15
REPEAT_SHARE = 0.3
MIN_REPEAT_PAGES = 3

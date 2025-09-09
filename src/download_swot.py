from __future__ import annotations

from typing import Sequence, List
from pathlib import Path
from datetime import datetime
import earthaccess as ea

UNSMOOTHED = "SWOT Level 2 KaRIn Low Rate Sea Surface Height Data Product - Unsmoothed, Version C"
EXPERT     = "SWOT Level 2 KaRIn Low Rate Sea Surface Height Data Product - Expert, Version C"

# ---------------- LOGIN ----------------

def login(strategy: str | None = None):
    """
    Log in to NASA Earthdata via earthaccess.
    - strategy="netrc" to use ~/.netrc
    - None -> interactive/browser if needed
    """
    if strategy:
        return ea.login(strategy=strategy)
    return ea.login()

def check_login() -> bool:
    """
    Verify that Earthdata login works (using netrc/env/session).
    Returns True if credentials are valid, False otherwise.
    """
    try:
        auth = ea.login(strategy="netrc")
        if not auth.authenticated:
            print("[warn] Not authenticated — check ~/.netrc or env vars")
            return False
        print("[info] Earthdata login OK (netrc)")
        return True
    except Exception as e:
        print(f"[error] Earthdata login failed: {e}")
        return False

# ---------------- HELPERS ----------------

def _iso(ts: str) -> str:
    """Accept 'YYYY-MM-DD' or full ISO; return ISO8601 with 'Z' when date-only."""
    if "T" in ts:
        # allow ...Z or with offset
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return ts
    datetime.fromisoformat(ts)  # validate date
    return f"{ts}T00:00:00Z"

def _validate_bbox(bbox: Sequence[float]) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must be [west, south, east, north], got {bbox}")
    w, s, e, n = map(float, bbox)
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f"bbox out of range: {bbox}")
    return [w, s, e, n]

def search_download(
    collection_title: str,
    bbox: Sequence[float],
    t0: str,
    t1: str,
    outdir: str | Path,
) -> List[Path]:
    """
    Search PO.DAAC (Earthdata) for SWOT granules by collection title, bbox, and time window,
    and download them into outdir. Returns list of local Paths.
    """
    bbox = _validate_bbox(bbox)
    t0, t1 = _iso(t0), _iso(t1)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Find collection
    cols = ea.search.DataCollections().keyword(collection_title).get()
    if not cols:
        raise ValueError(f"Collection not found: {collection_title}")
    cid = cols[0]["meta"]["concept-id"]

    # Search granules
    grans = (
        ea.search.DataGranules()
        .concept_id(cid)
        .bounding_box(bbox[0], bbox[1], bbox[2], bbox[3])
        .temporal(t0, t1)
        .get()
    )
    if not grans:
        print(f"[warn] No granules for '{collection_title}' in {t0}..{t1} bbox={bbox}")
        return []

    # Download
    local_paths = ea.download(grans, str(outdir))
    paths = [Path(p) for p in local_paths]
    total_mb = sum(p.stat().st_size for p in paths) / (1024**2) if paths else 0.0
    print(f"[info] Downloaded {len(paths)} files → {outdir} ({total_mb:.1f} MB)")
    return paths

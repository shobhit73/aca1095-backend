# aca_builder.py
from __future__ import annotations
import pandas as pd
from typing import List, Tuple, Dict
from datetime import date
from calendar import monthrange

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------- helpers ----------
def month_edges(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, int, str]]:
    out = []
    for m in range(1, 13):
        start = pd.Timestamp(year=year, month=m, day=1)
        end   = pd.Timestamp(year=year, month=m, day=monthrange(year, m)[1])
        out.append((start, end, m, MONTHS[m-1]))
    return out

def _covered_full_month(ranges: List[Tuple[pd.Timestamp, pd.Timestamp]],
                        mstart: pd.Timestamp, mend: pd.Timestamp) -> bool:
    """True if union of ranges covers every day in [mstart, mend]."""
    if not ranges:
        return False
    # clip to month, merge
    segs = []
    for s, e in ranges:
        if pd.isna(s) or pd.isna(e): 
            continue
        if e < mstart or s > mend:
            continue
        segs.append((max(s, mstart), min(e, mend)))
    if not segs:
        return False
    segs.sort()
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s <= (cur_e + pd.Timedelta(days=1)):
            cur_e = max(cur_e, e)
        else:
            # gap
            return False
    return cur_s <= mstart and cur_e >= mend

def _overlaps_any(ranges: List[Tuple[pd.Timestamp, pd.Timestamp]],
                  mstart: pd.Timestamp, mend: pd.Timestamp) -> bool:
    for s, e in ranges:
        if pd.isna(s) or pd.isna(e): 
            continue
        if not (e < mstart or s > mend):
            return True
    return False

def _has_future_elig(elig_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]],
                     m_end: pd.Timestamp) -> bool:
    for s, _ in elig_ranges:
        if pd.isna(s): 
            continue
        if s > m_end:
            return True
    return False

def _tier_has_spouse(tier: str) -> bool:
    t = (tier or "").upper()
    return "EMPFAM" in t or "EMPSPOUSE" in t

def _tier_has_child(tier: str) -> bool:
    t = (tier or "").upper()
    return "EMPFAM" i

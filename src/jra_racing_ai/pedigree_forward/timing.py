"""Same conservative minute/publication/observation contract as existing T10."""
from datetime import timedelta
import math
from ..win_sim import core


def quote_valid(quote,roster,cutoff):
    try:
        if not quote or set(quote['odds'])!=set(roster):return False
        cutoff=core.instant(cutoff) if isinstance(cutoff,str) else cutoff
        published=core.instant(quote['published_at']);seen=core.instant(quote['first_seen_at'])
        if not (published<=seen<=cutoff and published+timedelta(minutes=1)<=cutoff
                and 0<=(cutoff-published).total_seconds()<=300):return False
        return all(math.isfinite(float(v)) and float(v)>=1 for v in quote['odds'].values())
    except (ValueError,TypeError,KeyError,core.SimulationError):return False

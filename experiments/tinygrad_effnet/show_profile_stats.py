import pstats
from pstats import SortKey
p = pstats.Stats('prof_stats')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
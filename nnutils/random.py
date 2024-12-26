import os
import random
import sys
from typing import Tuple, Callable, Optional

# check that torch was not yet imported
if 'torch' in sys.modules or 'pytorch' in sys.modules:
    raise RuntimeError('To enable deterministic behavior the module has to be imported before torch module')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import numpy as np

def seed_all(seed, cuda=True) -> torch.Generator:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    return g

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def deterministic(enable=True, seed=0, cuda=True) -> Tuple[Optional[torch.Generator], Optional[Callable]]:
    if not enable:
        return None, None
    from torch._inductor import config
    config.fallback_random = True
    rng = seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if cuda:
        torch.backends.cudnn.benchmark = False
    return rng, seed_worker

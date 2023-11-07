import functools
import sys


def patch_tqdm():
    tqdm = sys.modules["tqdm"].tqdm
    sys.modules["tqdm"].tqdm = functools.partial(tqdm, file=sys.stdout)
    return

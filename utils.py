# -*- coding: utf-8 -*-
from contextlib import contextmanager
from time import time


@contextmanager
def timing(description: str) -> None:
    """
    Helper function to measure elapsed time for a code block.
    The usage:

    from utils import timing

    with timing('Doing the task'):
        [the computations]

    >> Doing the task... 0.42sec
    :param description: the description of the task
    :return: None
    """
    print(description+"... ", end="", flush=True)
    start = time()
    yield
    print(f"{(time() - start):.2f}sec")
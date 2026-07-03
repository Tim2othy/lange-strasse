"""Tiny logging shim so game logic can be silenced during simulation/training.

Game/dice/player code calls ``log(...)`` instead of ``print(...)``. Whether it
prints is decided once, by ``VERBOSE`` in config.py: set it False to run silent
(e.g. a big batch of simulated games), True to watch play-by-play.
"""

from config import VERBOSE


def log(*args, **kwargs) -> None:
    """print() that stays silent unless config.VERBOSE is set."""
    if VERBOSE:
        print(*args, **kwargs)

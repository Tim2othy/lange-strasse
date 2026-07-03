"""Tiny logging shim so game logic can be silenced during simulation/training.

Game/dice/player code calls ``log(...)`` instead of ``print(...)``. Interactive
play leaves VERBOSE=True (the default); the headless env sets it False so
millions of simulated games produce no output.
"""

from config import VERBOSE


def set_verbose(flag: bool) -> None:
    """Enable or disable game-logic output globally."""
    global VERBOSE
    VERBOSE = flag


def log(*args, **kwargs) -> None:
    """print() that obeys the global VERBOSE flag."""
    if VERBOSE:
        print(*args, **kwargs)

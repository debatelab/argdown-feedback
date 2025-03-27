
from abc import ABC

from pyargdown import Argdown

class Verifier(ABC):
    pass

class BaseArgdownVerifier(Verifier):
    """
    Base class for all Argdown verifiers.
    """

    def __init__(self, argdown: Argdown):
        if argdown is None:
            raise ValueError("Cannot initialize verifier with argdown object None")
        self.argdown = argdown


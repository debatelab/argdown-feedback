
from abc import ABC

from pyargdown import Argdown

class Verifier(ABC):
    pass



# !!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#     BIG REFACTORING TODO     # 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!! #

"""
* Recast verifiers as handlers that can be chained by judges.
"""


class BaseArgdownVerifier(Verifier):
    """
    Base class for all Argdown verifiers.
    """

    def __init__(self, argdown: Argdown):
        if argdown is None:
            raise ValueError("Cannot initialize verifier with argdown object None")
        self.argdown = argdown


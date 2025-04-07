
from abc import ABC

from pyargdown import Argdown

class Verifier(ABC):
    pass



# !!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#     BIG REFACTORING TODO     # 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!! #

"""
Recast verifiers as handlers that can be chained by judges.

Best practices:

- Define clear interfaces: Create a well-defined handler interface that all concrete handlers implement. This ensures consistency across the chain.
- Establish a standard request format: Define a consistent structure for requests that contains all necessary information handlers might need.
- Implement default behavior: Consider implementing a default "pass-through" behavior in a base handler class to reduce boilerplate code in concrete handlers.
- Decouple sender from receivers: The sender should only know about the first handler in the chain, not about specific handlers or the chain's structure.
- Use a composite pattern alongside chain of responsibility: Group verifiers into logical categories that can be applied selectively.
- Consider chain configuration: Provide a flexible way to configure and modify the chain at runtime rather than hard-coding it.
- Handle chain termination: Define clear conditions for when a request should stop propagating through the chain.
- Implement fallback handlers: Consider adding a default handler at the end of the chain to handle requests that weren't processed by any other handler.
- Be mindful of state: Be careful with handlers that maintain state, as this can lead to unexpected behavior in the chain.
- Consider asynchronous processing: For long-running operations, consider making handlers process requests asynchronously.
- Enable detailed logging: Implement logging within handlers to track how requests flow through the chain for debugging purposes.
- Handle exceptions gracefully: Implement proper error handling to prevent exceptions in one handler from breaking the entire chain.

"""


class BaseArgdownVerifier(Verifier):
    """
    Base class for all Argdown verifiers.
    """

    def __init__(self, argdown: Argdown):
        if argdown is None:
            raise ValueError("Cannot initialize verifier with argdown object None")
        self.argdown = argdown


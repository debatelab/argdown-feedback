
from abc import ABC, abstractmethod
import logging
from typing import Generic, Optional, TypeVar

from pyargdown import Argdown


from .verification_request import VerificationRequest




class BaseHandler(ABC):
    """Base handler interface for the Chain of Responsibility pattern."""
    
    def __init__(self, name: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self._next_handler: Optional['BaseHandler'] = None
        self.name = name or self.__class__.__name__
        self.logger = logger or logging.getLogger(self.__class__.__module__)
        
    def set_next(self, handler: 'BaseHandler') -> 'BaseHandler':
        """Set the next handler in the chain."""
        self._next_handler = handler
        return handler  # Return handler to allow chaining
    
    def process(self, request: VerificationRequest) -> VerificationRequest:
        """
        Process request and pass to next handler if it should continue.
        Returns the request with processing results added.
        """
        if not request.continue_processing:
            return request
            
        try:
            # Log handler execution
            self.logger.debug(f"Executing processing handler: {self.name}")
            request.executed_handlers.append(self.name)
            
            # Execute processing
            request = self.handle(request)
            
            # If there's a next handler and we should continue, pass the request along
            if self._next_handler and request.continue_processing:
                return self._next_handler.process(request)
                
        except Exception as e:
            # Log any exceptions
            self.logger.error(f"Error in processing handler {self.name}: {str(e)}", exc_info=True)
            request.add_result(
                self.name, 
                [], 
                False, 
                f"Processing error: {str(e)}"
            )
            
        return request

    
    @abstractmethod
    def handle(self, request: VerificationRequest) -> VerificationRequest:
        """
        Concrete processing logic to be implemented by subclasses.
        Should update the request with processing results and return it.
        """
        pass



# Define a type variable for handler types
H = TypeVar('H', bound=BaseHandler)

class CompositeHandler(BaseHandler, Generic[H]):
    """
    A composite handler that groups multiple handlers together.
    All handlers in the group are executed in sequence.
    """
    
    def __init__(self, 
                 name: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 handlers: list[H] | None = None):
        super().__init__(name, logger)
        self.handlers = handlers or []
        
    def add_handler(self, handler: H) -> None:
        """Add a handler to this composite."""
        self.handlers.append(handler)
    

    def handle(self, request: VerificationRequest) -> VerificationRequest:
        """Process request through all contained handlers."""
        current_request = request
        
        for handler in self.handlers:
            current_request = handler.process(current_request)
            
            # If processing should stop, break the chain
            if not current_request.continue_processing:
                break
        
        return current_request



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



from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from typing import Generic, Optional, TypeVar

from pyargdown import Argdown


from .verification_request import VerificationRequest, VDFilter, PrimaryVerificationData, VerificationDType, VerificationResult




class BaseHandler(ABC):
    """Base handler interface for the Chain of Responsibility pattern."""
            
    @staticmethod
    def create_metadata_filter(key: str, values: list) -> VDFilter:
        """Creates a filter that checks if the metadata of the verification data contains the given key and values."""
        def filter(vdata: PrimaryVerificationData) -> bool:
            if vdata.metadata is None:
                return False
            return vdata.metadata.get(key) in values
        return filter

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

        # request = deepcopy(request)  # Create a deep copy of the request to avoid side effects

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


from typing import Callable, Optional
import logging


from argdown_hirpo.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoCoherenceHandler,
    BaseArgannoInfrecoCoherenceHandler
)


class ArgannoLogrecoCoherenceHandler(ArgannoInfrecoCoherenceHandler):
    """Default arganno<>logreco coherence handler, same as for arganno<>infreco."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        filters: Optional[tuple[Callable,Callable]] = None,
        from_key: str = "from",
        handlers: list[BaseArgannoInfrecoCoherenceHandler] | None = None,
    ):
        super().__init__(name, logger, filters, from_key, handlers)
        

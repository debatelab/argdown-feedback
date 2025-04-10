from typing import Callable, Optional
import logging

from argdown_feedback.verifiers.coherence.arganno_argmap_handler import (
    ArgannoArgmapCoherenceHandler,
)
from argdown_feedback.verifiers.coherence.argmap_logreco_handler import (
    ArgmapLogrecoCoherenceHandler,
)
from argdown_feedback.verifiers.base import CompositeHandler


class ArgannoArgmapLogrecoCoherenceHandler(CompositeHandler[CompositeHandler]):
    """A composite handler that groups together multiple composite coherence handlers."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        filters: dict[str, tuple[Callable,Callable]] | None = None,
        from_key: str = "from",
        handlers: list[CompositeHandler] | None = None,
    ):
        super().__init__(name, logger, handlers)

        filters = filters or {}

        # Initialize with default handlers if none provided
        if not handlers:
            self.handlers = [
                ArgannoArgmapCoherenceHandler(
                    name="ArgannoArgmapCoherenceHandler", 
                    filters=filters.get("ArgannoArgmapCoherenceHandler")
                ),
                ArgmapLogrecoCoherenceHandler(
                    name="ArgmapLogrecoCoherenceHandler", 
                    filters=filters.get("ArgmapLogrecoCoherenceHandler"),
                    from_key=from_key
                ),
            ]
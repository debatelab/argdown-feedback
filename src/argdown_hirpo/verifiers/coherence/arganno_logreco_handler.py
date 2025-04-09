from typing import Optional
import logging


from argdown_hirpo.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoCoherenceHandler,
    BaseArgannoInfrecoCoherenceHandler
)
from argdown_hirpo.verifiers.verification_request import VDFilter


class ArgannoLogrecoCoherenceHandler(ArgannoInfrecoCoherenceHandler):
    """Default arganno<>logreco coherence handler, same as for arganno<>infreco."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        filters: Optional[tuple[VDFilter,VDFilter]] = None,
        from_key: str = "from",
        handlers: list[BaseArgannoInfrecoCoherenceHandler] | None = None,
    ):
        """Handler for evaluating coherence of Arganno and Logreco data.
        
        filters: Optional[tuple[VDFilter,VDFilter]] = None
            Filters for the verification data. The first filter is applied to extract argdown reco,
            and the second to extract the xml annotation.
            If None, default filters are used.
        """
        super().__init__(name, logger, filters, from_key, handlers)
        

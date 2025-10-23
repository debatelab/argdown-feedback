from typing import List

from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.core.content_check_handler import (
    HasAnnotationsHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    FencedCodeBlockExtractor,
    XMLParser,
)

from ..base import VerifierBuilder


class ArgannoBuilder(VerifierBuilder):
    """Builder for argumentative annotation verifier."""

    name = "arganno"
    description = "Validates argumentative annotations in XML format"
    input_types = ["xml"]
    allowed_filter_roles = ["arganno"]
    config_options = []  # ArgannoCompositeHandler doesn't accept any config options

    def build_handlers_pipeline(
        self, filters_spec: dict, **kwargs
    ) -> List[BaseHandler]:
        """Build arganno verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)

        return [
            FencedCodeBlockExtractor(name="FencedCodeBlockExtractor"),
            XMLParser(name="XMLAnnotationParser"),
            HasAnnotationsHandler(filter=vd_filters.get("arganno")),
            ArgannoCompositeHandler(filter=vd_filters.get("arganno")),
        ]

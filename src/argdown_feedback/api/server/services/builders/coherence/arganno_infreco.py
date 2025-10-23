from typing import List

from argdown_feedback.verifiers.base import BaseHandler
from argdown_feedback.verifiers.core.arganno_handler import ArgannoCompositeHandler
from argdown_feedback.verifiers.core.infreco_handler import (
    EndsWithConclusionHandler,
    HasArgumentsHandler,
    HasInferenceDataHandler,
    HasLabelHandler,
    HasPCSHandler,
    InfRecoCompositeHandler,
    NoDuplicatePCSLabelsHandler,
    PropRefsExistHandler,
    StartsWithPremiseHandler,
    UsesAllPropsHandler,
)
from argdown_feedback.verifiers.core.content_check_handler import (
    HasAnnotationsHandler,
    HasArgdownHandler,
)
from argdown_feedback.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.processing_handler import (
    DefaultProcessingHandler,
)

from ..base import VerifierBuilder
from .....shared.models import VerifierConfigOption


class ArgannoInfrecoBuilder(VerifierBuilder):
    """Builder for argumentative annotation and informal reconstruction coherence verifier."""

    name = "arganno_infreco"
    description = "Checks coherence between argumentative annotations and informal argument reconstructions"
    input_types = ["xml", "argdown"]
    allowed_filter_roles = ["arganno", "infreco"]
    config_options = [
        VerifierConfigOption(
            name="from_key",
            type="string",
            default="from",
            description="Key used for inference information in arguments",
            required=False
        )
    ]
    is_coherence_verifier = True

    def build_handlers_pipeline(
        self, filters_spec: dict, **kwargs
    ) -> List[BaseHandler]:
        """Build arganno_infreco coherence verification pipeline."""
        vd_filters = self._create_vd_filters(filters_spec)

        if "arganno" in vd_filters and "infreco" in vd_filters:
            arganno_filter = vd_filters["arganno"]
            infreco_filter = vd_filters["infreco"]
            # Filters for the verification data.
            # The first filter is applied to extract argdown reco,
            # and the second to extract the xml annotation.
            # If None, default filters are used.
            filters = (infreco_filter, arganno_filter)
        else:
            arganno_filter = None
            infreco_filter = None
            filters = None

        # Create InfRecoCompositeHandler
        infreco_handler = InfRecoCompositeHandler(
            handlers=[
                # Argument existence handlers
                HasArgumentsHandler(name="InfReco.HasArgumentsHandler"),
                HasPCSHandler(name="InfReco.HasPCSHandler"),
                # Argument form handlers
                StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler"),
                EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler"),
                NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler"),
                # Label and gist handlers
                HasLabelHandler(name="InfReco.HasLabelHandler"),
                # Inference data handlers
                HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler", **kwargs),
                PropRefsExistHandler(name="InfReco.PropRefsExistHandler", **kwargs),
                UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler", **kwargs),
            ],
            filter=infreco_filter,
            **kwargs
        )

        return [
            DefaultProcessingHandler(),
            HasAnnotationsHandler(filter=arganno_filter),
            HasArgdownHandler(filter=infreco_filter),
            ArgannoCompositeHandler(filter=arganno_filter),
            infreco_handler,
            ArgannoInfrecoCoherenceHandler(filters=filters, **kwargs),
        ]

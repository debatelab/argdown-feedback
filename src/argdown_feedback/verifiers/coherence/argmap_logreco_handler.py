from typing import Optional
import logging

from pyargdown import (
    Argdown,
    ArgdownMultiDiGraph,
    DialecticalType,
    Valence,
)

from argdown_feedback.logic import dialectics
from argdown_feedback.verifiers.coherence.argmap_infreco_handler import (
    BaseArgmapInfrecoCoherenceHandler,
)
from argdown_feedback.verifiers.verification_request import (
    VDFilter,
    VerificationRequest,
    PrimaryVerificationData,
    VerificationResult,
)
from argdown_feedback.verifiers.base import CompositeHandler



class BaseArgmapLogrecoCoherenceHandler(BaseArgmapInfrecoCoherenceHandler):
    """Base handler interface for evaluating coherence of Argmap and Logreco data."""

    def __init__(self, name: Optional[str] = None, logger: Optional[logging.Logger] = None,
                 filters: Optional[tuple[VDFilter,VDFilter]] = None, from_key: str = "from"):
        """Base handler interface for evaluating coherence of Argmap and Logreco data.
        
        filters: Optional[tuple[VDFilter,VDFilter]] = None
            Filters for the verification data. The first filter is applied to extract map,
            and the second to extract the reconstruction.
            If None, default filters are used.
        """
        super().__init__(name, logger, filters, from_key)

    def is_applicable(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> bool:
        return super().is_applicable(vdata1, vdata2, ctx)
    
    @staticmethod
    def get_labels(argdown_map: Argdown, argdown_reco: Argdown) -> tuple[list[str | None], list[str | None], list[str | None], list[str | None]]:
        """Get the labels of the arguments and propositions in the argdown map and reconstruction."""
        map_alabels = list(set(a.label for a in argdown_map.arguments))
        reco_alabels = list(set(a.label for a in argdown_reco.arguments))
        map_prop_labels = list(set(p.label for p in argdown_map.propositions))
        reco_prop_labels = list(set(p.label for p in argdown_reco.propositions))
        return map_alabels, reco_alabels, map_prop_labels, reco_prop_labels


class ArgmapLogrecoElemCohereHandler(BaseArgmapLogrecoCoherenceHandler):
    """Handler that checks coherence of elements between argmap and argument reconstruction."""
 
    def evaluate(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""
        assert isinstance(vdata1.data, ArgdownMultiDiGraph), "Internal error: vdata1.data is not ArgdownMultiDiGraph"
        assert isinstance(vdata2.data, ArgdownMultiDiGraph), "Internal error: vdata2.data is not ArgdownMultiDiGraph"
        argdown_map: Argdown = vdata1.data
        argdown_reco: Argdown = vdata2.data
        map_alabels, reco_alabels, map_prop_labels, reco_prop_labels = self.get_labels(argdown_map, argdown_reco)

        msgs = []
        for label in map_alabels:
            if label not in reco_alabels:
                msgs.append(f"Argument <{label}> in map is not reconstructed (argument label mismatch).")
        for label in reco_alabels:
            if label not in map_alabels:
                msgs.append(f"Reconstructed argument <{label}> is not in the map (argument label mismatch).")            
        for label in map_prop_labels:
            if label not in reco_prop_labels:
                msgs.append(f"Claim [{label}] in argument map has no corresponding proposition in reconstructions (proposition label mismatch).")

        is_valid = False if msgs else True
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata1.id, vdata2.id],
            is_valid=is_valid,
            message=" - ".join(msgs) if msgs else None,
        )


class ArgmapLogrecoRelationCohereHandler(BaseArgmapLogrecoCoherenceHandler):
    """Handler that checks coherence of relations between annotation and argument reconstruction."""

    def evaluate(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""
        assert isinstance(vdata1.data, ArgdownMultiDiGraph), "Internal error: vdata1.data is not ArgdownMultiDiGraph"
        assert isinstance(vdata2.data, ArgdownMultiDiGraph), "Internal error: vdata2.data is not ArgdownMultiDiGraph"
        argdown_map: Argdown = vdata1.data
        argdown_reco: Argdown = vdata2.data
        map_alabels, reco_alabels, map_prop_labels, reco_prop_labels = self.get_labels(argdown_map, argdown_reco)

        msgs = []

        for drel in argdown_map.dialectical_relations:
            if drel.source not in reco_alabels+reco_prop_labels or drel.target not in reco_alabels+reco_prop_labels:
                continue
            if DialecticalType.SKETCHED in drel.dialectics:
                rel_matches = argdown_reco.get_dialectical_relation(drel.source, drel.target)
                rel_matches = [] if rel_matches is None else rel_matches

                if any(
                    rm.valence == drel.valence
                    and DialecticalType.GROUNDED in rm.dialectics
                    for rm in rel_matches
                ):
                    continue

                if not any(rm.valence == drel.valence for rm in rel_matches):
                    msgs.append(
                        f"Dialectical {drel.valence.name} relation from node '{drel.source}' to node '{drel.target}' "
                        f"in argument map is not matched by any relation in the argument reconstruction."
                    )
                    continue
                msgs.append(
                    f"Dialectical {drel.valence.name} relation from node '{drel.source}' to node '{drel.target}' "
                    f"in argument map is not grounded in logical argument reconstructions."
                )


        for drel in argdown_reco.dialectical_relations:
            if drel.source not in map_alabels+map_prop_labels or drel.target not in map_alabels+map_prop_labels:
                continue
            if DialecticalType.GROUNDED in drel.dialectics:
                if drel.valence == Valence.SUPPORT:
                    if not dialectics.indirectly_supports(drel.source, drel.target, argdown_map):
                        msgs.append(
                            f"According to the argument reconstructions, item '{drel.source}' supports item '{drel.target}', "
                            f"but this dialectical relation is not captured in the argument map."
                        )
                elif drel.valence == Valence.ATTACK:
                    if not dialectics.indirectly_attacks(drel.source, drel.target, argdown_map):
                        msgs.append(
                            f"According to the argument reconstructions, item '{drel.source}' attacks item '{drel.target}', "
                            f"but this dialectical relation is not captured in the argument map."
                        )

        is_valid = False if msgs else True
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata1.id, vdata2.id],
            is_valid=is_valid,
            message=" - ".join(msgs) if msgs else None,
        )


class ArgmapLogrecoCoherenceHandler(CompositeHandler[BaseArgmapLogrecoCoherenceHandler]):
    """A composite handler that groups all argmap<>logreco coherence handlers together."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        filters: Optional[tuple[VDFilter,VDFilter]] = None,
        from_key: str = "from",
        handlers: list[BaseArgmapLogrecoCoherenceHandler] | None = None,
    ):
        super().__init__(name, logger, handlers)
        
        # Initialize with default handlers if none provided
        if not handlers:
            self.handlers = [
                ArgmapLogrecoElemCohereHandler(
                    name="ArgmapLogrecoElemCohereHandler", 
                    filters=filters,
                    from_key=from_key
                ),
                ArgmapLogrecoRelationCohereHandler(
                    name="ArgmapLogrecoRelationCohereHandler", 
                    filters=filters,
                    from_key=from_key
                ),
            ]
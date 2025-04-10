from typing import Optional
import logging

from pyargdown import (
    Argdown,
    ArgdownMultiDiGraph,
    Argument,
    Conclusion,
    DialecticalType,
    Proposition,
    Valence,
)

from argdown_feedback.logic import dialectics
from argdown_feedback.verifiers.verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
    VerificationResult,
    VDFilter,
)
from argdown_feedback.verifiers.base import CompositeHandler
from argdown_feedback.verifiers.coherence.coherence_handler import CoherenceHandler



class BaseArgmapInfrecoCoherenceHandler(CoherenceHandler):
    """Base handler interface for evaluating coherence of Argmap and Infreco data."""

    def __init__(self, name: Optional[str] = None, logger: Optional[logging.Logger] = None,
                 filters: Optional[tuple[VDFilter,VDFilter]] = None, from_key: str = "from"):
        """Base handler interface for evaluating coherence of Argmap and Infreco data.
        
        filters: Optional[tuple[VDFilter,VDFilter]] = None
            Filters for the verification data. The first filter is applied to extract map,
            and the second to extract the reconstruction.
            If None, default filters are used.
        """
        self._next_handler: Optional['CoherenceHandler'] = None
        self.name = name or self.__class__.__name__
        self.logger = logger or logging.getLogger(self.__class__.__module__)
        self.filters = filters
        self.from_key = from_key

    def is_applicable(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> bool:
        """Check if the handler is applicable to the given data pair.
        vdata1: last argdown argmap data
        vdata2: last argdown infreco data
        """
        if self.filters:
            filter_fn1, filter_fn2 = self.filters
        else:
            # Default filters for argmap and infreco data
            def filter_fn1(vd: PrimaryVerificationData) -> bool:
                metadata: dict = vd.metadata if vd.metadata is not None else {}
                return vd.dtype == VerificationDType.argdown and metadata.get("filename", "").startswith("map")
            def filter_fn2(vd: PrimaryVerificationData) -> bool:
                metadata: dict = vd.metadata if vd.metadata is not None else {}
                return vd.dtype == VerificationDType.argdown and metadata.get("filename", "").startswith("reconstructions")
        vds_map = [vd for vd in ctx.verification_data if filter_fn1(vd)]
        if not vds_map:
            return False
        vds_reco = [vd for vd in ctx.verification_data if filter_fn2(vd)]
        if not vds_reco:
            return False
        return (
            vdata1.id == vds_map[-1].id
            and vdata2.id == vds_reco[-1].id
        )
    

class ArgmapInfrecoElemCohereHandler(BaseArgmapInfrecoCoherenceHandler):
    """Handler that checks coherence of elements between annotation and argument reconstruction."""
 
    def evaluate(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""
        assert isinstance(vdata1.data, ArgdownMultiDiGraph), "Internal error: vdata1.data is not ArgdownMultiDiGraph"
        assert isinstance(vdata2.data, ArgdownMultiDiGraph), "Internal error: vdata2.data is not ArgdownMultiDiGraph"
        argdown_map: Argdown = vdata1.data
        argdown_reco: Argdown = vdata2.data

        msgs = []
        map_labels = list(set(a.label for a in argdown_map.arguments))
        reco_labels = list(set(a.label for a in argdown_reco.arguments))
        for label in map_labels:
            if label not in reco_labels:
                msgs.append(f"Argument <{label}> in map is not reconstructed (argument label mismatch).")
        for label in reco_labels:
            if label not in map_labels:
                msgs.append(f"Reconstructed argument <{label}> is not in the map (argument label mismatch).")            
        map_prop_labels = list(set(p.label for p in argdown_map.propositions))
        reco_prop_labels = list(set(p.label for p in argdown_reco.propositions))
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


class ArgmapInfrecoRelationCohereHandler(BaseArgmapInfrecoCoherenceHandler):
    """Handler that checks coherence of relations between argmap and argument reconstruction."""

    def evaluate(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""
        assert isinstance(vdata1.data, ArgdownMultiDiGraph), "Internal error: vdata1.data is not ArgdownMultiDiGraph"
        assert isinstance(vdata2.data, ArgdownMultiDiGraph), "Internal error: vdata2.data is not ArgdownMultiDiGraph"
        argdown_map: Argdown = vdata1.data
        argdown_reco: Argdown = vdata2.data

        msgs = []
        for drel in argdown_map.dialectical_relations:
            if DialecticalType.SKETCHED not in drel.dialectics:
                continue
            # get matched source nodes in reco
            source_m: Argument | Proposition | None
            target_m: Argument | Proposition | None
            if any(a.label==drel.source for a in argdown_map.arguments):
                source_m =  next( 
                    (a for a in argdown_reco.arguments if a.label==drel.source),
                    None
                )  
            else:
                source_m = next(
                    (p for p in argdown_reco.propositions if p.label==drel.source),
                    None
                )
            if any(a.label==drel.target for a in argdown_map.arguments):
                target_m = next(
                    (a for a in argdown_reco.arguments if a.label==drel.target),
                    None
                )
            else:
                target_m = next(
                    (p for p in argdown_reco.propositions if p.label==drel.target),
                    None
                )
            #print("drel:", drel)
            #print(f"source_m: {source_m}, target_m: {target_m}")
            if source_m is None or target_m is None:
                continue
            # check if the relation is grounded in reco
            if isinstance(source_m, Argument) and isinstance(target_m, Argument):
                if not source_m.pcs or not target_m.pcs:
                    continue
                if drel.valence == Valence.SUPPORT:
                    if any(
                        dialectics.are_identical(
                            argdown_reco.get_proposition(pr.proposition_label),
                            argdown_reco.get_proposition(source_m.pcs[-1].proposition_label)
                        )
                        for pr in target_m.pcs
                        if not isinstance(pr, Conclusion)
                    ):
                        continue
                    msgs.append(
                        f"Sketched support relation from <{drel.source}> to <{drel.target}> in argument map "
                        f"is not grounded in the argument reconstruction, conclusion of <{drel.source}> does "
                        f"not figure as premise in <{drel.target}>."
                    )
                elif drel.valence == Valence.ATTACK:
                    if any(
                        dialectics.are_contradictory(
                            argdown_reco.get_proposition(pr.proposition_label),
                            argdown_reco.get_proposition(source_m.pcs[-1].proposition_label),
                            argdown_reco
                        )
                        for pr in target_m.pcs
                        if not isinstance(pr, Conclusion)
                    ):
                        continue
                    msgs.append(
                        f"Sketched attack relation from <{drel.source}> to <{drel.target}> in argument map "
                        f"is not grounded in the argument reconstruction, conclusion of <{drel.source}> does "
                        f"not contradict any premise in <{drel.target}>."
                    )
            if isinstance(source_m, Proposition) and isinstance(target_m, Argument):
                if not target_m.pcs:
                    continue
                if drel.valence == Valence.SUPPORT:
                    if any(
                        dialectics.are_identical(
                            argdown_reco.get_proposition(pr.proposition_label),
                            source_m,
                        )
                        for pr in target_m.pcs
                        if not isinstance(pr, Conclusion)
                    ):
                        continue
                    msgs.append(
                        f"Sketched support relation from [{drel.source}] to <{drel.target}> in argument map "
                        f"is not grounded in the argument reconstruction, proposition [{drel.source}] does "
                        f"not figure as premise in <{drel.target}>."
                    )
                elif drel.valence == Valence.ATTACK:
                    if any(
                        dialectics.are_contradictory(
                            argdown_reco.get_proposition(pr.proposition_label),
                            source_m,
                            argdown_reco
                        )
                        for pr in target_m.pcs
                        if not isinstance(pr, Conclusion)
                    ):
                        continue
                    msgs.append(
                        f"Sketched attack relation from [{drel.source}] to <{drel.target}> in argument map "
                        f"is not grounded in the argument reconstruction, proposition [{drel.source}] does "
                        f"not contradict any premise in <{drel.target}>."
                    )
            if isinstance(source_m, Argument) and isinstance(target_m, Proposition):
                if not source_m.pcs:
                    continue
                if drel.valence == Valence.SUPPORT:
                    if dialectics.are_identical(
                        argdown_reco.get_proposition(source_m.pcs[-1].proposition_label),
                        target_m,
                    ):
                        continue
                    msgs.append(
                        f"Sketched support relation from <{drel.source}> to [{drel.target}] in argument map "
                        f"is not grounded in the argument reconstruction, proposition [{drel.target}] "
                        f"does not figure as conclusion in <{drel.source}>."
                    )
                if drel.valence == Valence.ATTACK:
                    if dialectics.are_contradictory(
                        argdown_reco.get_proposition(source_m.pcs[-1].proposition_label),
                        target_m,
                        argdown_reco
                    ):
                        continue
                    msgs.append(
                        f"Sketched attack relation from <{drel.source}> to [{drel.target}] in argument map "
                        f"is not grounded in the argument reconstruction, proposition [{drel.target}] "
                        f"does not contradict the conclusion of <{drel.source}>."
                    )

        is_valid = False if msgs else True
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata1.id, vdata2.id],
            is_valid=is_valid,
            message=" - ".join(msgs) if msgs else None,
        )


class ArgmapInfrecoCoherenceHandler(CompositeHandler[BaseArgmapInfrecoCoherenceHandler]):
    """A composite handler that groups all argmap<>infreco coherence handlers together."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        filters: Optional[tuple[VDFilter,VDFilter]] = None,
        from_key: str = "from",
        handlers: list[BaseArgmapInfrecoCoherenceHandler] | None = None,
    ):
        super().__init__(name, logger, handlers)
        
        # Initialize with default handlers if none provided
        if not handlers:
            self.handlers = [
                ArgmapInfrecoElemCohereHandler(
                    name="ArgmapInfrecoElemCohereHandler", 
                    filters=filters,
                    from_key=from_key
                ),
                ArgmapInfrecoRelationCohereHandler(
                    name="ArgmapInfrecoRelationCohereHandler", 
                    filters=filters,
                    from_key=from_key
                ),
            ]
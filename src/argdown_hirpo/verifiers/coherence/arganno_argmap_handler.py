from typing import Optional
import logging

from bs4 import BeautifulSoup
from pyargdown import (
    ArgdownMultiDiGraph,
    Valence,
)

from argdown_hirpo.verifiers.verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
    VerificationResult,
    VDFilter,
)
from argdown_hirpo.verifiers.base import CompositeHandler
from argdown_hirpo.verifiers.coherence.coherence_handler import CoherenceHandler


class BaseArgannoArgmapCoherenceHandler(CoherenceHandler):
    """Base handler interface for evaluating coherence of Arganno and Argmap data."""

    def __init__(self, name: Optional[str] = None, logger: Optional[logging.Logger] = None,
                 filters: Optional[tuple[VDFilter,VDFilter]] = None):
        """Base handler interface for evaluating coherence of Argmap and Arganno data.
        
        filters: Optional[tuple[VDFilter,VDFilter]] = None
            Filters for the verification data. The first filter is applied to extract argdown map,
            and the second to extract the xml annotation.
            If None, default filters are used.
        """
        self._next_handler: Optional['CoherenceHandler'] = None
        self.name = name or self.__class__.__name__
        self.logger = logger or logging.getLogger(self.__class__.__module__)
        self.filters = filters


    def is_applicable(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> bool:
        """Check if the handler is applicable to the given data pair.
        vdata1: last argdown data
        vdata2: last xml data
        """
        if self.filters:
            filter_fn1, filter_fn2 = self.filters
        else: 
            filter_fn1 = lambda vd: vd.dtype == VerificationDType.argdown  # noqa: E731
            filter_fn2 = lambda vd: vd.dtype == VerificationDType.xml  # noqa: E731

        vds_ad = [vd for vd in ctx.verification_data if filter_fn1(vd)]
        if not vds_ad:
            return False
        vds_xml = [vd for vd in ctx.verification_data if filter_fn2(vd)]
        if not vds_xml:
            return False
        return (
            vdata1.id == vds_ad[-1].id
            and vdata2.id == vds_xml[-1].id
        )
    
    @staticmethod
    def get_labels(argdown_map: ArgdownMultiDiGraph, soup_anno: BeautifulSoup) -> tuple[list[str], list, dict[str, str]]:
        all_argmap_labels = [node.label for node in argdown_map.propositions + argdown_map.arguments if node.label]
        all_annotation_ids = [
            a.get("id") for a in soup_anno.find_all("proposition") if a.get("id")  # type: ignore
        ]
        argument_label_map: dict[str,str] = {}
        for a in soup_anno.find_all("proposition"):
            a_label = a.get("argument_label")  # type: ignore
            a_id = a.get("id")  # type: ignore
            if a_label in all_argmap_labels:
                argument_label_map[str(a_id)] = str(a_label)

        return all_argmap_labels, all_annotation_ids, argument_label_map



class ArgannoArgmapElemCohereHandler(BaseArgannoArgmapCoherenceHandler):
 
    def evaluate(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""
        assert isinstance(vdata1.data, ArgdownMultiDiGraph), "Internal error: vdata1.data is not ArgdownMultiDiGraph"
        assert isinstance(vdata2.data, BeautifulSoup), "Internal error: vdata2.data is not BeautifulSoup"
        argdown_map: ArgdownMultiDiGraph = vdata1.data
        soup_anno: BeautifulSoup = vdata2.data
        all_argmap_labels, all_annotation_ids, argument_label_map = self.get_labels(argdown_map, soup_anno)

        msgs = []

        annos_illegal_label = [
            a for a in soup_anno.find_all("proposition")
            if a.get("argument_label") not in all_argmap_labels  # type: ignore
        ]
        if annos_illegal_label:
            for a in annos_illegal_label:
                msgs.append(
                    f"Illegal 'argument_label' reference of proposition element with id={a.get('id')}: "  # type: ignore
                    f"No node with label '{a.get('argument_label')}' in the Argdown argument map."  # type: ignore
                )

        for node in argdown_map.propositions + argdown_map.arguments:
            id_refs = node.data.get("annotation_ids", [])
            if not id_refs:
                msgs.append(
                    f"Missing 'annotation_ids' attribute of node with label '{node.label}'."
                )
                continue
            for id_ref in id_refs:
                if id_ref not in all_annotation_ids:
                    msgs.append(
                        f"Illegal 'annotation_ids' reference of node with label '{node.label}': "
                        f"No proposition element with id='{id_ref}' in the annotation."
                    )
                elif argument_label_map.get(id_ref) != node.label:
                    msgs.append(
                        f"Label reference mismatch: argument map node with label '{node.label}' "
                        f"has annotation_ids={str(id_refs)}, but the corresponding proposition element "
                        f"with id={id_ref} in the annotation has a different argument_label"
                        f"{': '+argument_label_map[id_ref] if id_ref in argument_label_map else ''}."
                    )

        is_valid = False if msgs else True
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata1.id, vdata2.id],
            is_valid=is_valid,
            message=" ".join(msgs) if msgs else None,
        )



class ArgannoArgmapDRelCohereHandler(BaseArgannoArgmapCoherenceHandler):
    """Handler for checking the coherence of dialectical relations between Argdown and Annotation data."""

 
    def evaluate(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""
        assert isinstance(vdata1.data, ArgdownMultiDiGraph), "Internal error: vdata1.data is not ArgdownMultiDiGraph"
        assert isinstance(vdata2.data, BeautifulSoup), "Internal error: vdata2.data is not BeautifulSoup"
        argdown_map: ArgdownMultiDiGraph = vdata1.data
        soup_anno: BeautifulSoup = vdata2.data
        _, all_annotation_ids, argument_label_map = self.get_labels(argdown_map, soup_anno)

        msgs = []
        annotated_relations: list[dict] = []
        for a in soup_anno.find_all("proposition"):
            from_id = a.get("id")  # type: ignore
            for support in a.get("supports", []):  # type: ignore
                if support in all_annotation_ids:
                    annotated_relations.append(
                        {
                            "from_id": from_id,
                            "to_id": support,
                            "valence": Valence.SUPPORT,
                        }
                    )
            for attacks in a.get("attacks", []):  # type: ignore
                if attacks in all_annotation_ids:
                    annotated_relations.append(
                        {
                            "from_id": from_id,
                            "to_id": attacks,
                            "valence": Valence.ATTACK,
                        }
                    )


        for ar in annotated_relations:
            if not any(
                dr.source == argument_label_map.get(ar["from_id"])
                and dr.target == argument_label_map.get(ar["to_id"])
                and dr.valence == ar["valence"]
                for dr in argdown_map.dialectical_relations
            ):
                msgs.append(
                    f"Annotated {str(ar['valence'])} relation {ar['from_id']} -> {ar['to_id']} is not "
                    f"matched by any relation in the argument map."
                )

        for dr in argdown_map.dialectical_relations:
            if not any(
                dr.source == argument_label_map.get(ar["from_id"])
                and dr.target == argument_label_map.get(ar["to_id"])
                and dr.valence == ar["valence"]
                for ar in annotated_relations
            ):
                msgs.append(
                    f"Dialectical {dr.valence.name} relation {dr.source} -> {dr.target} is not matched by any "
                    f"relation in the text annotation."
                )

        is_valid = False if msgs else True
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata1.id, vdata2.id],
            is_valid=is_valid,
            message=" ".join(msgs) if msgs else None,
        )



class ArgannoArgmapCoherenceHandler(CompositeHandler[BaseArgannoArgmapCoherenceHandler]):
    """A composite handler that groups all arganno<>argmap verification handlers together."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        filters: Optional[tuple[VDFilter,VDFilter]] = None,
        handlers: list[BaseArgannoArgmapCoherenceHandler] | None = None,
    ):
        super().__init__(name, logger, handlers)
        
        # Initialize with default handlers if none provided
        if not handlers:
            self.handlers = [
                ArgannoArgmapElemCohereHandler(name="ArgannoArgmapElemCohereHandler", filters=filters),
                ArgannoArgmapDRelCohereHandler(name="ArgannoArgmapDRelCohereHandler", filters=filters),
            ]
            
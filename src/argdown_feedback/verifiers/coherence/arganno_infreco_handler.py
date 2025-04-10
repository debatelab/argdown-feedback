from typing import Optional, Dict
import logging

from bs4 import BeautifulSoup
from pyargdown import (
    Argdown,
    ArgdownMultiDiGraph,
    Argument,
    Conclusion,
    Valence,
)

from argdown_feedback.verifiers.verification_request import (
    VDFilter,
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
    VerificationResult,
)
from argdown_feedback.verifiers.base import CompositeHandler
from argdown_feedback.verifiers.coherence.coherence_handler import CoherenceHandler


def _get_props_used_in_inference(
    argument: Argument, pr_label: str, from_key: str = "from"
) -> list[str]:
    """Get all proposition labels used directly or indirectly in the inference
    to a conclusion with label `pr_label`."""

    if argument is None or not argument.pcs:
        return []

    used_labels = set()

    def add_parent_labels(label: str):
        c = next(
            (c for c in argument.pcs if isinstance(c, Conclusion) and c.label == label),
            None,
        )
        if c is None:
            return []
        parent_labels = c.inference_data.get(from_key, [])
        used_labels.update(parent_labels)
        for ref in parent_labels:
            add_parent_labels(ref)

    add_parent_labels(pr_label)

    return list(used_labels)


class BaseArgannoInfrecoCoherenceHandler(CoherenceHandler):
    """Base handler interface for evaluating coherence of Arganno and InfReco data."""

    def __init__(self, name: Optional[str] = None, logger: Optional[logging.Logger] = None,
                 filters: Optional[tuple[VDFilter,VDFilter]] = None, from_key: str = "from"):
        """Base handler interface for evaluating coherence of Arganno and Infreco data.
        
        filters: Optional[tuple[VDFilter,VDFilter]] = None
            Filters for the verification data. The first filter is applied to extract argdown reco,
            and the second to extract the xml annotation.
            If None, default filters are used.
        """
        self._next_handler: Optional['CoherenceHandler'] = None
        self.name = name or self.__class__.__name__
        self.logger = logger or logging.getLogger(self.__class__.__module__)
        self.filters = filters
        self.from_key = from_key

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
    def get_labels(argdown_reco: Argdown, soup_anno: BeautifulSoup) -> tuple[list[str], list, Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Get labels from argdown and annotation data."""
        all_argument_labels = [arg.label for arg in argdown_reco.arguments if arg.label]
        all_annotation_ids = [
            a.get("id")  # type: ignore
            for a in soup_anno.find_all("proposition")
            if a.get("id")  # type: ignore
        ]

        # maps `id` of annotated proposition to its `argument_label`
        argument_label_map: Dict[str, str] = {}
        # maps `id` of annotated proposition to its `ref_reco_label`
        refreco_map: Dict[str, str] = {}
        # maps `id` of annotated proposition to `proposition_label` correponding to `ref_reco_label` in pcs of `argument_label` 
        proposition_label_map: Dict[str, str] = {}  

        for a in soup_anno.find_all("proposition"):
            a_label = a.get("argument_label")  # type: ignore
            a_id = a.get("id")  # type: ignore
            a_ref_reco = a.get("ref_reco_label")  # type: ignore
            
            if a_label in all_argument_labels and a_id is not None:
                argument_label_map[str(a_id)] = str(a_label)
                if a_ref_reco is not None:
                    refreco_map[str(a_id)] = str(a_ref_reco)
                    
                    # Find proposition label
                    argument = next((arg for arg in argdown_reco.arguments if arg.label == a_label), None)
                    if argument and argument.pcs:
                        pr = next((pr for pr in argument.pcs if pr.label == a_ref_reco), None)
                        if pr:
                            proposition_label_map[str(a_id)] = str(pr.proposition_label)

        return all_argument_labels, all_annotation_ids, argument_label_map, refreco_map, proposition_label_map


class ArgannoInfrecoElemCohereHandler(BaseArgannoInfrecoCoherenceHandler):
    """Handler that checks coherence of elements between annotation and argument reconstruction."""
 
    def evaluate(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""
        assert isinstance(vdata1.data, ArgdownMultiDiGraph), "Internal error: vdata1.data is not ArgdownMultiDiGraph"
        assert isinstance(vdata2.data, BeautifulSoup), "Internal error: vdata2.data is not BeautifulSoup"
        argdown_reco: Argdown = vdata1.data
        soup_anno: BeautifulSoup = vdata2.data
        
        all_argument_labels, all_annotation_ids, argument_label_map, refreco_map, proposition_label_map = self.get_labels(
            argdown_reco, soup_anno
        )

        msgs = []
        
        # Check annotation elements against argdown
        for a in soup_anno.find_all("proposition"):
            a_label = a.get("argument_label")  # type: ignore
            a_id = a.get("id")  # type: ignore
            a_ref_reco = a.get("ref_reco_label")  # type: ignore
            
            if a_label not in all_argument_labels:
                msgs.append(
                    f"Illegal 'argument_label' reference of proposition element with id={a_id}: "
                    f"No argument with label '{a_label}' in the Argdown snippet."
                )
                continue
                
            if a_id is not None and a_label is not None and a_ref_reco is not None:
                argument = next((arg for arg in argdown_reco.arguments if arg.label == a_label), None)
                if argument and argument.pcs:
                    if not any(a_ref_reco == pr.label for pr in argument.pcs):
                        msgs.append(
                            f"Illegal 'ref_reco_label' reference of proposition element with id={a_id}: "
                            f"No premise or conclusion with label '{a_ref_reco}' in argument '{a_label}'."
                        )
                    else:
                        pr = next(
                            pr for pr in argument.pcs if a_ref_reco == pr.label
                        )
                        proposition = next(
                            prop
                            for prop in argdown_reco.propositions
                            if prop.label == pr.proposition_label
                        )
                        id_refs = proposition.data.get("annotation_ids", [])
                        if str(a_id) not in id_refs:
                            msgs.append(
                                f"Label reference mismatch: proposition element with id={a_id} in the annotation "
                                f"references (via ref_reco) the proposition '{pr.label}' of argument '{argument.label}', "
                                f"but the annotation_ids={str(id_refs)} of that proposition do not include the id={a_id}."
                            )

        # Check argdown elements against annotation
        for argument in argdown_reco.arguments:
            if argument.label not in argument_label_map.values():
                msgs.append(
                    f"Free floating argument: Argument '{argument.label}' does not have any "
                    "corresponding elements in the annotation."
                )
            
            for pr in argument.pcs:
                proposition = next(
                    prop
                    for prop in argdown_reco.propositions
                    if prop.label == pr.proposition_label
                )
                id_refs = proposition.data.get("annotation_ids")
                if id_refs is None:
                    msgs.append(
                        f"Missing 'annotation_ids' attribute in proposition '{pr.label}' "
                        f"of argument '{argument.label}'."
                    )
                    continue
                for id_ref in id_refs:
                    if id_ref not in all_annotation_ids:
                        msgs.append(
                            f"Illegal 'annotation_ids' reference in proposition '{pr.label}' of argument '{argument.label}': "
                            f"No proposition element with id='{id_ref}' in the annotation."
                        )
                        continue

        # Check for overlapping proposition references
        for i in range(len(argdown_reco.propositions)):
            for j in range(i + 1, len(argdown_reco.propositions)):
                prop1 = argdown_reco.propositions[i]
                prop2 = argdown_reco.propositions[j]
                dps = [
                    f"'{x}'"
                    for x in prop1.data.get("annotation_ids", [])
                    if x in prop2.data.get("annotation_ids", [])
                ]
                if dps:
                    msgs.append(
                        f"Label reference mismatch: annotation text segment(s) {', '.join(dps)} "
                        f"are referenced by distinct propositions in the Argdown argument "
                        f"reconstruction ('{prop1.label}', '{prop2.label}')."
                    )

        is_valid = False if msgs else True
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata1.id, vdata2.id],
            is_valid=is_valid,
            message=" - ".join(msgs) if msgs else None,
        )


class ArgannoInfrecoRelationCohereHandler(BaseArgannoInfrecoCoherenceHandler):
    """Handler that checks coherence of relations between annotation and argument reconstruction."""

    def evaluate(self, vdata1: PrimaryVerificationData, vdata2: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""
        assert isinstance(vdata1.data, ArgdownMultiDiGraph), "Internal error: vdata1.data is not ArgdownMultiDiGraph"
        assert isinstance(vdata2.data, BeautifulSoup), "Internal error: vdata2.data is not BeautifulSoup"
        argdown_reco: Argdown = vdata1.data
        soup_anno: BeautifulSoup = vdata2.data
        
        all_argument_labels, all_annotation_ids, argument_label_map, refreco_map, proposition_label_map = self.get_labels(
            argdown_reco, soup_anno
        )

        msgs = []
        
        # Extract annotated relations
        annotated_support_relations: list[dict] = []
        annotated_attack_relations: list[dict] = []
        
        for a in soup_anno.find_all("proposition"):
            from_id = a.get("id")  # type: ignore
            for support in a.get("supports", []):  # type: ignore
                if support in all_annotation_ids:
                    annotated_support_relations.append(
                        {
                            "from_id": str(from_id),
                            "to_id": str(support),
                        }
                    )
            for attack in a.get("attacks", []):  # type: ignore
                if attack in all_annotation_ids:
                    annotated_attack_relations.append(
                        {
                            "from_id": str(from_id),
                            "to_id": str(attack),
                        }
                    )

        # Helper function for dialectical relations
        def _drel_fn(x, y):
            drels = argdown_reco.get_dialectical_relation(x, y)
            return drels if drels is not None else []

        # Check support relations
        for ar in annotated_support_relations:
            arglabel_from = argument_label_map.get(ar["from_id"])
            proplabel_from = proposition_label_map.get(ar["from_id"])
            arglabel_to = argument_label_map.get(ar["to_id"])
            proplabel_to = proposition_label_map.get(ar["to_id"])
            
            if arglabel_from is None or arglabel_to is None:
                msgs.append(
                    f"Annotated support relation {ar['from_id']} -> {ar['to_id']} is not "
                    f"matched by any relation in the reconstruction (illegal argument_labels)."
                )
                continue
                
            if arglabel_from != arglabel_to:
                drels = _drel_fn(
                    arglabel_from, arglabel_to,
                ) + _drel_fn(
                    arglabel_from, proplabel_to,
                ) + _drel_fn(
                    proplabel_from, arglabel_to,
                ) + _drel_fn(
                    proplabel_from, proplabel_to,
                )
                if drels is None or not any(
                    dr.valence == Valence.SUPPORT
                    for dr in drels
                ):
                    msgs.append(
                        f"Proposition elements {ar['from_id']} and {ar['to_id']} are annotated to support each other, but "
                        f"none of the corresponding Argdown elements <{arglabel_from}>/[{proplabel_from}] supports "
                        f"<{arglabel_to}> or [{proplabel_to}]."
                    )
                continue
                
            argument = next(
                (arg for arg in argdown_reco.arguments if arg.label == arglabel_from),
                None
            )
            ref_reco_from = refreco_map.get(ar["from_id"])
            ref_reco_to = refreco_map.get(ar["to_id"])
            
            if argument is None or ref_reco_from is None or ref_reco_to is None:
                continue
                
            if ref_reco_from not in _get_props_used_in_inference(argument, ref_reco_to, self.from_key):
                msgs.append(
                    f"Annotated support relation {ar['from_id']} -> {ar['to_id']} is not "
                    f"matched by the inferential relations in the argument '{argument.label}'."
                )

        # Check attack relations
        for ar in annotated_attack_relations:
            arglabel_from = argument_label_map.get(ar["from_id"])
            proplabel_from = proposition_label_map.get(ar["from_id"])
            arglabel_to = argument_label_map.get(ar["to_id"])
            proplabel_to = proposition_label_map.get(ar["to_id"])
            
            if arglabel_from is None or arglabel_to is None:
                msgs.append(
                    f"Annotated attack relation from {ar['from_id']} to {ar['to_id']} is not "
                    f"matched by any relation in the reconstruction (illegal argument_labels)."
                )
                continue
                
            if arglabel_from == arglabel_to:
                msgs.append(
                    f"Text segments assigned to the same argument cannot attack each other "
                    f"({ar['from_id']} attacks {ar['to_id']} while both are assigned to {arglabel_from})."
                )
                continue
                
            drels = _drel_fn(
                arglabel_from, arglabel_to,
            ) + _drel_fn(
                arglabel_from, proplabel_to,
            ) + _drel_fn(
                proplabel_from, arglabel_to,
            ) + _drel_fn(
                proplabel_from, proplabel_to,
            )
            if drels is None or not any(
                dr.valence == Valence.ATTACK
                for dr in drels
            ):
                msgs.append(
                    f"Proposition elements {ar['from_id']} and {ar['to_id']} are annotated to attack each other, but "
                    f"none of the corresponding Argdown elements <{arglabel_from}>/[{proplabel_from}] attacks "
                    f"<{arglabel_to}> or [{proplabel_to}]."
                )

        is_valid = False if msgs else True
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata1.id, vdata2.id],
            is_valid=is_valid,
            message=" - ".join(msgs) if msgs else None,
        )


class ArgannoInfrecoCoherenceHandler(CompositeHandler[BaseArgannoInfrecoCoherenceHandler]):
    """A composite handler that groups all arganno<>infreco coherence handlers together."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        filters: Optional[tuple[VDFilter,VDFilter]] = None,
        from_key: str = "from",
        handlers: list[BaseArgannoInfrecoCoherenceHandler] | None = None,
    ):
        super().__init__(name, logger, handlers)
        
        # Initialize with default handlers if none provided
        if not handlers:
            self.handlers = [
                ArgannoInfrecoElemCohereHandler(
                    name="ArgannoInfrecoElemCohereHandler", 
                    filters=filters,
                    from_key=from_key
                ),
                ArgannoInfrecoRelationCohereHandler(
                    name="ArgannoInfrecoRelationCohereHandler", 
                    filters=filters,
                    from_key=from_key
                ),
            ]
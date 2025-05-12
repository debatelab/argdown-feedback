from abc import abstractmethod
from typing import Optional
import logging

from pyargdown import (
    ArgdownMultiDiGraph,
    Conclusion,
    DialecticalType,
)
from pyargdown.parser.base import ArgdownParser

from argdown_feedback.verifiers.verification_request import (
    VDFilter,
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
    VerificationResult,
)
from argdown_feedback.verifiers.base import BaseHandler, CompositeHandler


class InfRecoHandler(BaseHandler):
    """Base handler interface for evaluating informal argument reconstructions."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        from_key: str = "from",
        filter: Optional[VDFilter] = None,
    ):
        super().__init__(name, logger)
        self.from_key = from_key
        self.filter = filter if filter else lambda vdata: True

    @abstractmethod
    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""

    def is_applicable(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> bool:
        """Check if the handler is applicable to the given data."""
        return vdata.dtype == VerificationDType.argdown and self.filter(vdata)

    def handle(self, request: VerificationRequest) -> VerificationRequest:
        for vdata in request.verification_data:
            if vdata.data is None:
                continue
            if self.is_applicable(vdata, request):
                vresult = self.evaluate(vdata, request)
                if vresult is not None:
                    request.add_result_record(vresult)
        return request


class HasArgumentsHandler(InfRecoHandler):
    """Handler that checks if there are any arguments in the argdown data."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
            
        if not argdown.arguments:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message="No arguments found in the argdown data.",
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )
    
class HasUniqueArgumentHandler(InfRecoHandler):
    """Handler that checks that there is a unique argument in the Argdown document."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")

        msg = None

        if len(argdown.arguments) > 1:
            msg = "More than one argument found in the argdown data."
        elif len(argdown.arguments) == 0:
            msg = "No arguments found in the argdown data."
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=msg is None,
            message=msg,
        )

class HasAtLeastNArgumentsHandler(InfRecoHandler):
    """Handler that checks that there are more than N arguments in the Argdown document."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        from_key: str = "from",
        filter: Optional[VDFilter] = None,
        N: int = 1,
    ):
        super().__init__(name, logger, from_key, filter)
        self.N = N

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")

        msg = None

        size = len(argdown.arguments)
        if size < self.N:
            msg = f"Not enough arguments (found {size}, expected ≥{self.N})."
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=msg is None,
            message=msg,
        )


class HasPCSHandler(InfRecoHandler):
    """Handler that checks if all arguments have premise conclusion structures."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
            
        if not argdown.arguments:
            return None
        
        invalid_args = []
        for idx, argument in enumerate(argdown.arguments):
            if not argument.pcs:
                invalid_args.append(f"<{argument.label}>" if argument.label else f"Argument #{idx+1}")
        
        msg = None
        if invalid_args:
            msg = f"The following arguments lack premise conclusion structure: {', '.join(invalid_args)}"
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=msg is None,
            message=msg,
        )


class StartsWithPremiseHandler(InfRecoHandler):
    """Handler that checks if all arguments start with a premise."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
        
        invalid_args = []
        for argument in argdown.arguments:
            if argument.pcs and isinstance(argument.pcs[0], Conclusion):
                invalid_args.append(f"<{argument.label}>" if argument.label else "<unlabeled argument>")
        
        msg = None
        if invalid_args:
            msg = f"The following arguments do not start with a premise: {', '.join(invalid_args)}"
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=msg is None,
            message=msg,
        )


class EndsWithConclusionHandler(InfRecoHandler):
    """Handler that checks if all arguments end with a conclusion."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
        
        invalid_args = []
        for argument in argdown.arguments:
            if argument.pcs and not isinstance(argument.pcs[-1], Conclusion):
                invalid_args.append(f"<{argument.label}>" if argument.label else "<unlabeled argument>")
        
        msg = None
        if invalid_args:
            msg = f"The following arguments do end with a conclusion: {', '.join(invalid_args)}"
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=msg is None,
            message=msg,
        )



class NotMultipleGistsHandler(InfRecoHandler):
    """Handler that checks if all arguments have at most one gist."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
        
        invalid_args = []
        for argument in argdown.arguments:
            if len(argument.gists) > 1:
                invalid_args.append(f"<{argument.label}>" if argument.label else "<unlabeled argument>")
        
        msg = None
        if invalid_args:
            msg = f"The following arguments have alternative gists (and are declared multiple times): {', '.join(invalid_args)}"
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=msg is None,
            message=msg,
        )


class NoDuplicatePCSLabelsHandler(InfRecoHandler):
    """Handler that checks if all arguments have no duplicate labels in their premise-conclusion structure."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
        
        invalid_args = []
        for argument in argdown.arguments:
            if not argument.pcs:
                continue
                
            pcs_labels = [p.label for p in argument.pcs]
            duplicates = list(
                set([label for label in pcs_labels if pcs_labels.count(label) > 1])
            )
            
            if duplicates:
                arg_label = f"<{argument.label}>" if argument.label else "<unlabeled argument>"
                invalid_args.append(f"{arg_label} (duplicates: {', '.join([f'({lbl})' for lbl in duplicates])})")
        
        msg = None
        if invalid_args:
            msg = f"The following arguments have duplicate premise/conclusion labels: {', '.join(invalid_args)}"
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=msg is None,
            message=msg,
        )



class HasLabelHandler(InfRecoHandler):
    """Handler that checks if all arguments have labels."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
        
        unlabeled_args = []
        for idx, argument in enumerate(argdown.arguments):
            if ArgdownParser.is_unlabeled(argument):
                unlabeled_args.append(f"Argument #{idx+1}")

        msg = (
            f"The following arguments lack labels: {', '.join(unlabeled_args)}"
            if unlabeled_args
            else None
        )
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=msg is None,
            message=msg,
        )


class HasGistHandler(InfRecoHandler):
    """Handler that checks if all arguments have gists."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
        
        invalid_args = []
        for argument in argdown.arguments:
            if not argument.gists:
                invalid_args.append(f"<{argument.label}>" if argument.label else "<unlabeled argument>")
        

        msg = (
            f"The following arguments lack gists: {', '.join(invalid_args)}"
            if invalid_args
            else None
        )
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=msg is None,
            message=msg,
        )



class HasInferenceDataHandler(InfRecoHandler):
    """Handler that checks if all arguments have inference data for all conclusions."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
        
        msgs = []
        for argument in argdown.arguments:
            arg_label = f"<{argument.label}>" if argument.label else "<unlabeled argument>"
            
            if not argument.pcs:
                continue
                
            for c in argument.pcs:
                if not isinstance(c, Conclusion):
                    continue
                    
                inf_data = c.inference_data
                if not inf_data:
                    msgs.append(f"In {arg_label}: Inference to conclusion {c.label} lacks yaml inference information.")
                else:
                    from_list = inf_data.get(self.from_key)
                    if from_list is None:
                        msgs.append(
                            f"In {arg_label}: Inference to conclusion {c.label} inference information lacks '{self.from_key}' key."
                        )
                    elif not isinstance(from_list, list):
                        msgs.append(
                            f"In {arg_label}: Inference to conclusion {c.label} inference information '{self.from_key}' value is not a list."
                        )
                    elif len(from_list) == 0:
                        msgs.append(
                            f"In {arg_label}: Inference to conclusion {c.label} inference information '{self.from_key}' value is empty."
                        )
        
        if msgs:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message=" ".join(msgs),
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class PropRefsExistHandler(InfRecoHandler):
    """Handler that checks if all proposition references in inference data exist."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
        
        msgs = []
        for argument in argdown.arguments:
            arg_label = f"<{argument.label}>" if argument.label else "<unlabeled argument>"
            
            if not argument.pcs:
                continue
                
            for enum, c in enumerate(argument.pcs):
                if isinstance(c, Conclusion):
                    inf_data = c.inference_data
                    from_list = inf_data.get(self.from_key, [])
                    if isinstance(from_list, list):
                        for ref in from_list:
                            if str(ref) not in [p.label for p in argument.pcs[:enum]]:
                                msgs.append(
                                    f"In {arg_label}: Item '{ref}' in inference information of conclusion {c.label} does "
                                    "not refer to a previously introduced premise or conclusion."
                                )
        
        if msgs:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message=" ".join(msgs),
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class UsesAllPropsHandler(InfRecoHandler):
    """Handler that checks if all propositions are used in inferences across all arguments."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
        
        msgs = []
        for argument in argdown.arguments:
            arg_label = f"<{argument.label}>" if argument.label else "<unlabeled argument>"
            
            if not argument.pcs:
                continue
                
            used_labels = set()
            for c in argument.pcs:
                if isinstance(c, Conclusion):
                    used_labels.update(c.inference_data.get(self.from_key, []))
            
            unused_props = [
                f"({p.label})" for p in argument.pcs[:-1] if p.label not in used_labels
            ]
            
            if unused_props:
                msgs.append(
                    f"In {arg_label}: Some propositions are not explicitly used in any inferences: {', '.join(unused_props)}."
                )
        
        if msgs:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message=" ".join(msgs),
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class NoExtraPropositionsHandler(InfRecoHandler):
    """Handler that checks if there are no extra propositions outside of arguments."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
            
        pcs_props = set(
            [p.proposition_label for argument in argdown.arguments for p in argument.pcs]
        )
        all_probs = set(
            [p.label for p in argdown.propositions if p.label is not None]
        )
        outside_props = list(all_probs - pcs_props)
        outside_props = [f"[{lb}]" for lb in outside_props]
        
        if outside_props:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message=f"Argdown snippet contains propositions not used in any argument: {', '.join(outside_props)}.",
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class OnlyGroundedDialecticalRelationsHandler(InfRecoHandler):
    """Handler that checks if only grounded dialectical relations are used."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
            
        if any(
            set(d.dialectics) != {DialecticalType.GROUNDED}
            for d in argdown.dialectical_relations
        ):
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message="Argdown snippet defines dialectical relations.",
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class NoPropInlineDataHandler(InfRecoHandler):
    """Handler that checks if no propositions have inline data."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
            
        if any(prop.data for prop in argdown.propositions):
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message="Some propositions contain yaml inline data.",
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class NoArgInlineDataHandler(InfRecoHandler):
    """Handler that checks if no arguments have inline data."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")
            
        if any(arg.data for arg in argdown.arguments):
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message="Some arguments contain yaml inline data.",
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class InfRecoCompositeHandler(CompositeHandler[InfRecoHandler]):
    """A composite handler that groups all informal reconstruction verification handlers together."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        handlers: list[InfRecoHandler] | None = None,
        from_key: str = "from",
        filter: Optional[VDFilter] = None,
    ):
        super().__init__(name, logger, handlers)
        
        # Initialize with default handlers if none provided
        if not handlers:
            self.handlers = [
                # Argument existence handlers
                HasArgumentsHandler(name="InfReco.HasArgumentsHandler", filter=filter),
                HasUniqueArgumentHandler(name="InfReco.HasUniqueArgumentHandler", filter=filter),
                HasPCSHandler(name="InfReco.HasPCSHandler", filter=filter),

                # Argument form handlers
                StartsWithPremiseHandler(name="InfReco.StartsWithPremiseHandler", filter=filter),
                EndsWithConclusionHandler(name="InfReco.EndsWithConclusionHandler", filter=filter),
                NotMultipleGistsHandler(name="InfReco.NotMultipleGistsHandler", filter=filter),
                NoDuplicatePCSLabelsHandler(name="InfReco.NoDuplicatePCSLabelsHandler", filter=filter),
                
                # Label and gist handlers
                HasLabelHandler(name="InfReco.HasLabelHandler", filter=filter),
                HasGistHandler(name="InfReco.HasGistHandler", filter=filter),
                
                # Inference data handlers
                HasInferenceDataHandler(name="InfReco.HasInferenceDataHandler", from_key=from_key, filter=filter),
                PropRefsExistHandler(name="InfReco.PropRefsExistHandler", from_key=from_key, filter=filter),
                UsesAllPropsHandler(name="InfReco.UsesAllPropsHandler", from_key=from_key, filter=filter),
                
                # Content restriction handlers
                NoExtraPropositionsHandler(name="InfReco.NoExtraPropositionsHandler", filter=filter),
                OnlyGroundedDialecticalRelationsHandler(name="InfReco.OnlyGroundedDialecticalRelationsHandler", filter=filter),
                NoPropInlineDataHandler(name="InfReco.NoPropInlineDataHandler", filter=filter),
                NoArgInlineDataHandler(name="InfReco.NoArgInlineDataHandler", filter=filter),
            ]
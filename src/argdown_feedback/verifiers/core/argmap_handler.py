from abc import abstractmethod
from textwrap import shorten
from typing import Optional
import logging

from pyargdown import (
    ArgdownMultiDiGraph,
    Proposition,
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


class ArgMapHandler(BaseHandler):
    """Base handler interface for evaluating argument maps."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        filter: Optional[VDFilter] = None,
    ):
        super().__init__(name, logger)
        self.filter = filter if filter else lambda vdata: True

    @abstractmethod
    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""

    def is_applicable(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> bool:
        """Check if the handler is applicable to the given data. Can be customized in subclasses."""
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


class CompleteClaimsHandler(ArgMapHandler):
    """Handler that checks if all claims have labels and are not empty."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")

        incomplete_claims: list[str] = []
        for claim in argdown.propositions:
            assert isinstance(claim, Proposition)
            if ArgdownParser.is_unlabeled(claim):
                if not claim.texts or not claim.texts[0]:
                    incomplete_claims.append("Empty claim")
                else:
                    incomplete_claims.append(shorten(claim.texts[0], width=40))
                    
        is_valid = len(incomplete_claims) == 0
        message = None
        if not is_valid:
            message = f"Missing labels for nodes: {', '.join(incomplete_claims)}"
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class NoDuplicateLabelsHandler(ArgMapHandler):
    """Handler that checks for duplicate labels in claims and arguments."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")

        duplicate_labels: list[str] = []
        for claim in argdown.propositions:
            if len(claim.texts) > 1 and claim.label:
                duplicate_labels.append(claim.label)
        for argument in argdown.arguments:
            if len(argument.gists) > 1 and argument.label:
                duplicate_labels.append(argument.label)
                
        is_valid = len(duplicate_labels) == 0
        message = None
        if not is_valid:
            message = f"Duplicate labels: {', '.join(duplicate_labels)}"
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class NoPCSHandler(ArgMapHandler):
    """Handler that checks if any argument has a premise-conclusion structure."""


    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise ValueError("Internal error: Argdown is not a MultiDiGraph")

        arguments_with_pcs = [
            argument for argument in argdown.arguments if argument.pcs
        ]
        
        is_valid = len(arguments_with_pcs) == 0
        message = None
        if not is_valid:
            arg_labels = [
                f"<{argument.label}>" if argument.label else "<unlabeled_argument>"
                for argument in arguments_with_pcs
            ]
            message = f"Found detailed reconstruction of individual argument(s) {', '.join(arg_labels)} as premise-conclusion-structures."
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class ArgMapCompositeHandler(CompositeHandler[ArgMapHandler]):
    """A composite handler that groups all argmap verification handlers together."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        handlers: list[ArgMapHandler] | None = None,
        filter: Optional[VDFilter] = None,
    ):
        super().__init__(name, logger, handlers)
        
        # Initialize with default handlers if none provided
        if not handlers:
            self.handlers = [
                CompleteClaimsHandler(name="ArgMap.CompleteClaimsHandler", filter=filter),
                NoDuplicateLabelsHandler(name="ArgMap.NoDuplicateLabelsHandler", filter=filter),
                NoPCSHandler(name="ArgMap.NoPCSHandler", filter=filter),
            ]
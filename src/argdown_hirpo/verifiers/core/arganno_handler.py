from abc import abstractmethod
from difflib import unified_diff
from textwrap import shorten
from typing import Optional, Sequence
import logging

from bs4 import BeautifulSoup


from argdown_hirpo.verifiers.verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
    VerificationResult,
)
from argdown_hirpo.verifiers.base import BaseHandler, CompositeHandler


class ArgannoHandler(BaseHandler):
    """Base handler interface for evaluating individual argumentative annotations."""

    @abstractmethod
    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""

    def is_applicable(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> bool:
        """Check if the handler is applicable to the given data. Can be customized in subclasses."""
        return vdata.dtype == VerificationDType.xml

    def handle(self, request: VerificationRequest):
        for vdata in request.verification_data:
            if vdata.data is None:
                continue
            if self.is_applicable(vdata, request):
                vresult = self.evaluate(vdata, request)
                if vresult is not None:
                    request.add_result_record(vresult)                    
        return request


class SourceTextIntegrityHandler(ArgannoHandler):
    """Handler that checks if the source text has been altered."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")
        msgs = []
        source = ctx.source
        if not source:
            return None
        if isinstance(source, str):
            source = source.strip()
        lines_o = " ".join(source.split()).splitlines(keepends=True)
        lines_a = " ".join(soup.get_text().split()).splitlines(keepends=True)
        lines_o = [line for line in lines_o if line.strip()]
        lines_a = [line for line in lines_a if line.strip()]

        diff = list(unified_diff(lines_o, lines_a, n=0))
        if diff:
            msgs.append(
                f"Source text '{shorten(source, 40)}' was altered. Diff:\n" + "".join(diff),
            )

        is_valid = False if msgs else True
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=" ".join(msgs) if msgs else None,
        )
    


class NestedPropositionHandler(ArgannoHandler):
    """Handler that checks for nested proposition annotations."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")

        nested_props = [
            f"'{shorten(str(proposition), 256)}'"
            for proposition in soup.find_all("proposition")
            if proposition.find_all("proposition")  # type: ignore
        ]
        
        is_valid = len(nested_props) == 0
        message = None
        if not is_valid:
            message = f"Nested annotations in proposition(s) {', '.join(nested_props)}"
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class PropositionIdPresenceHandler(ArgannoHandler):
    """Handler that checks that every proposition has an id."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")

        props_without_id = [
            f"'{shorten(str(proposition), 64)}'"
            for proposition in soup.find_all("proposition")
            if not proposition.get("id")  # type: ignore
        ]
        
        is_valid = len(props_without_id) == 0
        message = None
        if not is_valid:
            message = f"Missing id in proposition(s) {', '.join(props_without_id)}"
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class PropositionIdUniquenessHandler(ArgannoHandler):
    """Handler that checks that every proposition has a unique id."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")

        ids = [
            str(proposition.get("id"))  # type: ignore
            for proposition in soup.find_all("proposition")
        ]
        duplicates = {id for id in ids if ids.count(id) > 1}
        
        is_valid = len(duplicates) == 0
        message = None
        if not is_valid:
            message = f"Duplicate ids: {', '.join(duplicates)}"
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class SupportReferenceValidityHandler(ArgannoHandler):
    """Handler that checks that every "supports" reference is a valid id."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")

        ids = [
            proposition.get("id")  # type: ignore
            for proposition in soup.find_all("proposition")
        ]
        msgs = []
        for proposition in soup.find_all("proposition"):
            for support in proposition.get("supports", []):  # type: ignore
                if support not in ids:
                    msgs.append(
                        f"Supported proposition with id '{support}' in proposition '{shorten(str(proposition), 64)}' does not exist."
                    )
        
        is_valid = len(msgs) == 0
        message = " ".join(msgs) if msgs else None
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class AttackReferenceValidityHandler(ArgannoHandler):
    """Handler that checks that every "attacks" reference is a valid id."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")

        ids = [
            proposition.get("id")  # type: ignore
            for proposition in soup.find_all("proposition")
        ]
        msgs = []
        for proposition in soup.find_all("proposition"):
            for attack in proposition.get("attacks", []):  # type: ignore
                if attack not in ids:
                    msgs.append(
                        f"Attacked proposition with id '{attack}' in proposition '{shorten(str(proposition), 64)}' does not exist."
                    )
        
        is_valid = len(msgs) == 0
        message = " ".join(msgs) if msgs else None
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class AttributeValidityHandler(ArgannoHandler):
    """Handler that checks for unknown attributes in propositions."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")

        unknown_attrs = []
        for proposition in soup.find_all("proposition"):
            for attr in proposition.attrs:  # type: ignore
                if attr not in {
                    "id",
                    "supports",
                    "attacks",
                    "argument_label",
                    "ref_reco_label",
                }:
                    unknown_attrs.append(
                        f"Unknown attribute '{attr}' in proposition '{shorten(str(proposition), 64)}'"
                    )
        
        is_valid = len(unknown_attrs) == 0
        message = " ".join(unknown_attrs) if unknown_attrs else None
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class ElementValidityHandler(ArgannoHandler):
    """Handler that checks for unknown elements in the soup."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")

        unknown_elements = []
        for element in soup.find_all():
            element_name = element.name  # type: ignore
            if element_name not in {"proposition"}:
                unknown_elements.append(
                    f"Unknown element '{element_name}' at '{shorten(str(element), 64)}'"
                )
        
        is_valid = len(unknown_elements) == 0
        message = " ".join(unknown_elements) if unknown_elements else None
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class ArgumentLabelValidityHandler(ArgannoHandler):
    """Handler that checks that every argument label is one of the legal labels."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        legal_labels: Sequence[str] | None = None,
    ):
        super().__init__(name, logger)
        self.legal_labels = legal_labels or []

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")
        
        # If no legal labels are set, we can get them from context
        legal_labels = self.legal_labels

        if not legal_labels:
            # Skip validation if no legal labels are defined
            return None

        illegal_labels = []
        for proposition in soup.find_all("proposition"):
            argument_label = proposition.get("argument_label")  # type: ignore
            if argument_label is not None and argument_label not in legal_labels:
                illegal_labels.append(
                    f"Illegal argument label '{argument_label}' "
                    f"in proposition '{shorten(str(proposition), 64)}'"
                )
        
        is_valid = len(illegal_labels) == 0
        message = " ".join(illegal_labels) if illegal_labels else None
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class RefRecoLabelValidityHandler(ArgannoHandler):
    """Handler that checks that every ref_reco label is one of the legal labels."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        legal_labels: Sequence[str] | None = None,
    ):
        super().__init__(name, logger)
        self.legal_labels = legal_labels or []

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        soup = vdata.data
        if soup is None:
            return None
        if not isinstance(soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")
        
        # If no legal labels are set, we can get them from context
        legal_labels = self.legal_labels

        if not legal_labels:
            # Skip validation if no legal labels are defined
            return None

        illegal_labels = []
        for proposition in soup.find_all("proposition"):
            ref_reco_label = proposition.get("ref_reco_label")  # type: ignore
            if ref_reco_label is not None and ref_reco_label not in legal_labels:
                illegal_labels.append(
                    f"Illegal ref_reco label '{ref_reco_label}' "
                    f"in proposition '{shorten(str(proposition), 64)}'"
                )
        
        is_valid = len(illegal_labels) == 0
        message = " ".join(illegal_labels) if illegal_labels else None
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=is_valid,
            message=message,
        )


class ArgannoCompositeHandler(CompositeHandler[ArgannoHandler]):
    """A composite handler that groups all arganno verification handlers together."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        handlers: list[ArgannoHandler] | None = None,
    ):
        super().__init__(name, logger, handlers)
        
        # Initialize with default handlers if none provided
        if not handlers:
            self.handlers = [
                SourceTextIntegrityHandler(),
                NestedPropositionHandler(),
                PropositionIdPresenceHandler(),
                PropositionIdUniquenessHandler(),
                SupportReferenceValidityHandler(),
                AttackReferenceValidityHandler(),
                AttributeValidityHandler(),
                ElementValidityHandler(),
            ]
            
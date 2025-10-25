"""
Verifier registry using builder pattern.
"""

from abc import ABC, abstractmethod

from typing import Dict, List, Any, Type

from bs4 import BeautifulSoup
from pyargdown import Argdown

from ....verifiers.verification_request import PrimaryVerificationData, VDFilter, VerificationDType, VerificationRequest

from ....verifiers.base import CompositeHandler

from ...shared.models import ScoringResult, VerifierInfo, VerifiersList, VerifierConfigOption
from ...shared.filtering import FilterRoleType
from ...shared.exceptions import VerifierNotFoundError
#from ..services import verifier_registry


class BaseScorer(ABC):
    """Abstract base class for virtue scorers."""

    scorer_id: str

    def __init__(self, parent_name: str, vd_filters: Dict[FilterRoleType, VDFilter]) -> None:
        self.name = parent_name + "." + self.scorer_id
        self.vd_filters = vd_filters

    @abstractmethod
    def score(self, result: VerificationRequest) -> ScoringResult:
        """Score the given verification data and return scoring results."""
        pass

    def get_argdown(self, request: VerificationRequest, roles: List[FilterRoleType] = ["argmap", "infreco", "logreco"]) -> tuple[Argdown | None, str | None]:
        """Extract Argdown content from the verification request, if available.
        Applies scorer's filters for any of the roles (default=argmap, infreco, logreco), 
        and ensures the primary data type is Argdown."""

        def combined_filter(data: PrimaryVerificationData) -> bool:
            for role, vd_filter in self.vd_filters.items():
                if role in roles:
                    if not vd_filter(data):
                        return False
            return True

        filtered_verification_data = [
            (data.data, data.code_snippet)
            for data in request.verification_data
            if data.dtype == VerificationDType.argdown
            and data.data is not None
            and isinstance(data.data, Argdown)
            and combined_filter(data)
        ]

        if not filtered_verification_data:
            return None, None

        argdown_data, argdown_snippet = next(reversed(filtered_verification_data))

        return argdown_data, argdown_snippet


    def get_xml_soup(self, request: VerificationRequest) -> tuple[BeautifulSoup | None, str | None]:
        """Extract XML Annotation content (BeautifulSoup) from the verification request, if available.
        Applies scorer's filters for role=arganno, 
        and ensures the primary data type is Argdown."""

        filtered_verification_data = [
            (data.data, data.code_snippet)
            for data in request.verification_data
            if data.dtype == VerificationDType.xml
            and data.data is not None
            and isinstance(data.data, BeautifulSoup)
            and (self.vd_filters["arganno"](data) if "arganno" in self.vd_filters else True)
        ]

        if not filtered_verification_data:
            return None, None

        soup, xml_snippet = next(reversed(filtered_verification_data))

        return soup, xml_snippet

    def get_formalizations(self, request: VerificationRequest, roles: List[FilterRoleType] = ["logreco"]) -> tuple[Any,Any]:
        """Extract formalization expressions from the verification request, if available.
        Applies scorer's filters for any of the roles (default=logreco), 
        and ensures the primary data type is Argdown."""

        def combined_filter(data: PrimaryVerificationData) -> bool:
            for role, vd_filter in self.vd_filters.items():
                if role in roles:
                    if not vd_filter(data):
                        return False
            return True

        filtered_verification_data = [
            data
            for data in request.verification_data
            if data.dtype == VerificationDType.argdown
            and data.data is not None
            and isinstance(data.data, Argdown)
            and combined_filter(data)
        ]

        if not filtered_verification_data:
            return None, None

        argdown_vd_id = filtered_verification_data[-1].id

        # NOTE: formalizations are stored as details in result of WellFormedFormulasHandler
        wff_result = next(
            (
                result
                for result in request.results
                if "WellFormedFormulasHandler" in result.verifier_id  # NOTE: hacky way to get right VerificationResult
                and result.verification_data_references == [argdown_vd_id]
            ),
            None,
        )

        if wff_result is None:
            return None, None

        all_expressions = wff_result.details.get("all_expressions")
        all_declarations = wff_result.details.get("all_declarations")

        return all_expressions, all_declarations


class ScorerCompositeHandler(CompositeHandler):
    """Composite handler with extra virtue scoring method."""

    def __init__(self, **kwargs):
        scorers = kwargs.pop("scorers", [])
        super().__init__(**kwargs)
        self.scorers: List[BaseScorer] = scorers

    def score(self, result: VerificationRequest) -> List[ScoringResult]:
        """Run all virtue scorers on the given evaluation and collect results."""
        if not result.is_valid():
            return []
        all_scores: List[ScoringResult] = []
        for scorer in self.scorers:
            score = scorer.score(result)
            all_scores.append(score)
        return all_scores
    


class AbstractVerifierBuilder(ABC):
    """Interface for verifier builders."""

    name: str
    description: str
    input_types: List[str]
    allowed_filter_roles: List[str]
    scorer_classes: List[Type[BaseScorer]] = []
    config_options: List[VerifierConfigOption] = []
    is_coherence_verifier: bool = False

    def __init__(self) -> None:
        super().__init__()
        # for each scorer class, add a config option to enable/disable it
        for scorer_class in self.scorer_classes:
            option_name = f"enable_{scorer_class.scorer_id}"
            if not any(opt.name == option_name for opt in self.config_options):
                self.config_options.append(
                    VerifierConfigOption(
                        name=option_name,
                        type="bool",
                        default=False,
                        description=f"Enable scoring of {scorer_class.scorer_id}",
                        required=False
                    )
                )

    @abstractmethod
    def build(self, filters_spec: dict[FilterRoleType, Any], **kwargs) -> ScorerCompositeHandler:
        """Build complete verifier handler with validation."""        
    
    @abstractmethod
    def validate_filters(self, filter_roles: List[FilterRoleType]) -> List[str]:
        """Validate filter roles against allowed roles. Returns list of invalid roles."""
    
    @abstractmethod
    def validate_config(self, config: dict) -> List[str]:
        """Validate configuration options. Returns list of invalid options."""
    
    @abstractmethod
    def get_info(self) -> VerifierInfo:
        """Get verifier information for API responses."""


class VerifierRegistry:
    """Registry for managing available verifiers using builder pattern."""
    
    def __init__(self):
        self._builders: Dict[str, AbstractVerifierBuilder] = {}
    
    def register(self, name: str, builder: AbstractVerifierBuilder):
        """Register a verifier builder."""
        print(f"Registering builder: {name}")
        self._builders[name] = builder
        print(f"Current builders: {self._builders.keys()}")
    
    def get_builder(self, name: str) -> AbstractVerifierBuilder:
        """Get verifier builder by name."""
        if name not in self._builders:
            raise VerifierNotFoundError(name, list(self._builders.keys()))
        return self._builders[name]
    
    def create_handler(self, name: str, **kwargs) -> ScorerCompositeHandler:
        """Create a handler instance for a verifier with full preprocessing pipeline."""
        filters_spec = kwargs.pop("filters", None)
        filters_spec = filters_spec if filters_spec is not None else {}
        builder = self.get_builder(name)
        return builder.build(filters_spec, **kwargs)
    
    def validate_config_options(self, name: str, config: Dict[str, Any]) -> List[str]:
        """Validate configuration options for a verifier. Returns list of invalid options."""
        builder = self.get_builder(name)
        invalid_options = builder.validate_config(config)
        return invalid_options
    
    def validate_filter_roles(self, name: str, filter_roles: List[FilterRoleType]) -> List[str]:
        """Validate filter roles for a verifier. Returns list of invalid roles."""
        builder = self.get_builder(name)
        invalid_roles = builder.validate_filters(filter_roles)
        return invalid_roles

    def list_verifiers(self) -> List[str]:
        """List all available verifier names."""
        print(f"Listing verifiers: {list(self._builders.keys())}")
        return list(self._builders.keys())
    
    def get_verifier_info(self, name: str) -> VerifierInfo:
        """Get verifier information for API responses."""
        return self.get_builder(name).get_info()
    
    def get_all_verifiers_info(self) -> VerifiersList:
        """Get information about all verifiers grouped by category."""
        core = []
        coherence = []
        content_check = []
        
        for builder in self._builders.values():
            info = builder.get_info()
            
            if builder.name.startswith("has_"):
                content_check.append(info)
            elif builder.is_coherence_verifier:
                coherence.append(info)
            else:
                core.append(info)
        
        return VerifiersList(
            core=core,
            coherence=coherence,
            content_check=content_check
        )


#def verifier(name: str):
#    """Decorator for registering verifier builders."""
#    def decorator(builder_class):
#        print(f"Attempting to register verifier: {name}")
#        builder_instance = builder_class()
#        builder_instance.name = name
#        print(f"Verifier instance created: {builder_instance}")
#        print(f"Verifier registry before registration: {verifier_registry.list_verifiers()}")
#        verifier_registry.register(name, builder_instance)
#        print(f"Verifier registry after registration: {verifier_registry.list_verifiers()}")
#        return builder_class
#    return decorator


from abc import abstractmethod
from typing import List, Type

from argdown_feedback.verifiers.verification_request import PrimaryVerificationData, VDFilter, VerificationDType

from .....verifiers.base import BaseHandler
from ..verifier_registry import AbstractVerifierBuilder, BaseScorer, ScorerCompositeHandler
from ....shared.models import VerifierInfo



class VerifierBuilder(AbstractVerifierBuilder):
    """Abstract base class for verifier builders."""

    @abstractmethod
    def build_handlers_pipeline(self, filters_spec: dict, **kwargs) -> List[BaseHandler]:
        """Build the complete pipeline including preprocessing and main handler."""
        pass

    def build_scorers(self, filters_spec: dict, **kwargs) -> List[BaseScorer]:
        """Build the list of virtue scorers to be used."""
        scorers = [
            scorer_class(parent_name=self.name) 
            for scorer_class in self.scorer_classes
            if kwargs.get("enable_" + scorer_class.scorer_id, True)
        ]
        return scorers
        
    def build(self, filters_spec: dict, **kwargs) -> ScorerCompositeHandler:
        """Build complete verifier handler with validation."""        
        handlers = self.build_handlers_pipeline(filters_spec, **kwargs)
        scorers = self.build_scorers(filters_spec, **kwargs)
        return ScorerCompositeHandler(
            name=f"{self.name}_pipeline",
            handlers=handlers,
            scorers=scorers
        )
    
    def validate_filters(self, filter_roles: List[str]) -> List[str]:
        """Validate filter roles against allowed roles. Returns list of invalid roles."""
        invalid_roles = [role for role in filter_roles 
                        if role not in self.allowed_filter_roles]
        return invalid_roles
    
    def validate_config(self, config: dict) -> List[str]:
        """Validate configuration options. Returns list of invalid options."""
        valid_options = {opt.name for opt in self.config_options}
        valid_options.add("filters")  # Always allowed
        invalid_options = [key for key in config.keys() 
                          if key not in valid_options]
        return invalid_options
    
    def _create_vd_filters(self, filters_spec: dict) -> dict[str, VDFilter]:
        """Create VDFilter functions for given filter roles and criteria."""
        # TODO: Move this from existing registry - complex filter creation logic
        vd_filters: dict[str, VDFilter] = {}

        def make_filter(role: str, criteria_list: list[dict[str,str]]) -> VDFilter:
            def vd_filter(vdata: PrimaryVerificationData) -> bool:
                if vdata.dtype == VerificationDType.xml and role != "arganno":
                    return False
                elif vdata.dtype != VerificationDType.xml and role == "arganno":
                    return False
                for criteria in criteria_list:
                    if vdata.metadata is None:
                        return False
                    key = criteria.get("key")
                    value = criteria.get("value")
                    if key not in vdata.metadata or value is None:
                        return False
                    if not criteria.get("regex"):
                        if not vdata.metadata.get(key) == value:
                            return False
                    else:
                        regex_pattern = value
                        import re
                        if not re.match(regex_pattern, str(vdata.metadata.get(key))):
                            return False
                return True
            return vd_filter

        for role, criteria_list in filters_spec.items():            
            vd_filters[role] = make_filter(role, criteria_list)
        
        return vd_filters


    def get_info(self) -> VerifierInfo:
        """Get verifier information for API responses."""
        return VerifierInfo(
            name=self.name,
            description=self.description,
            input_types=self.input_types,
            allowed_filter_roles=self.allowed_filter_roles,
            config_options=self.config_options,
            is_coherence_verifier=self.is_coherence_verifier
        )


# def verifier(name: str):
#     """Decorator for registering verifier builders."""
#     def decorator(builder_class):
#         builder_instance = builder_class()
#         builder_instance.name = name  # Ensure name matches decorator
#         #verifier_registry.register(name, builder_instance)
#         return builder_class
#     return decorator

"""
Verifier registry using builder pattern.
"""

from abc import ABC, abstractmethod

from typing import Dict, List, Any

from ....verifiers.base import BaseHandler

from ...shared.models import VerifierInfo, VerifiersList, VerifierConfigOption
from ...shared.exceptions import VerifierNotFoundError
#from ..services import verifier_registry


class AbstractVerifierBuilder(ABC):
    """Interface for verifier builders."""

    name: str
    description: str
    input_types: List[str]
    allowed_filter_roles: List[str]
    config_options: List[VerifierConfigOption] = []
    is_coherence_verifier: bool = False

    @abstractmethod
    def build(self, filters_spec: dict, **kwargs) -> BaseHandler:
        """Build complete verifier handler with validation."""        
    
    @abstractmethod
    def validate_filters(self, filter_roles: List[str]) -> List[str]:
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
    
    def create_handler(self, name: str, **kwargs) -> BaseHandler:
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
    
    def validate_filter_roles(self, name: str, filter_roles: List[str]) -> List[str]:
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


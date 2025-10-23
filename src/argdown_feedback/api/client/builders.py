"""
Type-safe request builders for the argdown-feedback API.

Provides builder pattern for creating verification requests with compile-time
type safety for configuration options and filter roles.
"""

from typing import Literal, Optional, Any, Dict
from ..shared.models import VerificationRequest
from ..shared.filtering import (
    InfrecoFilterBuilder, ArgannoFilterBuilder, ArgmapFilterBuilder, LogrecoFilterBuilder,
    ArgmapInfrecoFilterBuilder, ArgannoArgmapFilterBuilder, ArgannoInfrecoFilterBuilder,
    ArgannoLogrecoFilterBuilder, ArgmapLogrecoFilterBuilder, ArgannoArgmapLogrecoFilterBuilder
)


# Core verifier request builders

class ArgannoRequestBuilder:
    """Type-safe request builder for arganno verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = ArgannoFilterBuilder()
    
    # ArgannoCompositeHandler doesn't accept any config options
    
    def add_filter(self, role: Literal["arganno"], key: str, value: Any, regex: bool = False) -> 'ArgannoRequestBuilder':
        """Only allow 'arganno' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


class ArgmapRequestBuilder:
    """Type-safe request builder for argmap verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = ArgmapFilterBuilder()
    
    def add_filter(self, role: Literal["argmap"], key: str, value: Any, regex: bool = False) -> 'ArgmapRequestBuilder':
        """Only allow 'argmap' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


class InfrecoRequestBuilder:
    """Type-safe request builder for infreco verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = InfrecoFilterBuilder()
    
    def config_option(self, key: Literal["from_key"], value: Any) -> 'InfrecoRequestBuilder':
        """Only allow valid config options for infreco."""
        self.config[key] = value
        return self
    
    def add_filter(self, role: Literal["infreco"], key: str, value: Any, regex: bool = False) -> 'InfrecoRequestBuilder':
        """Only allow 'infreco' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


class LogrecoRequestBuilder:
    """Type-safe request builder for logreco verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = LogrecoFilterBuilder()
    
    def config_option(self, key: Literal["from_key", "formalization_key", "declarations_key"], value: Any) -> 'LogrecoRequestBuilder':
        """Only allow valid config options for logreco."""
        self.config[key] = value
        return self
    
    def add_filter(self, role: Literal["logreco"], key: str, value: Any, regex: bool = False) -> 'LogrecoRequestBuilder':
        """Only allow 'logreco' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


class HasAnnotationsRequestBuilder:
    """Type-safe request builder for has_annotations verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=None
        )


class HasArgdownRequestBuilder:
    """Type-safe request builder for has_argdown verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=None
        )


# Coherence verifier request builders

class ArgannoArgmapRequestBuilder:
    """Type-safe request builder for arganno_argmap coherence verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = ArgannoArgmapFilterBuilder()
    
    def add_filter(self, role: Literal["arganno", "argmap"], key: str, value: Any, regex: bool = False) -> 'ArgannoArgmapRequestBuilder':
        """Allow 'arganno' and 'argmap' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


class ArgannoInfrecoRequestBuilder:
    """Type-safe request builder for arganno_infreco coherence verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = ArgannoInfrecoFilterBuilder()
    
    def config_option(self, key: Literal["from_key"], value: Any) -> 'ArgannoInfrecoRequestBuilder':
        """Only allow valid config options for arganno_infreco."""
        self.config[key] = value
        return self
    
    def add_filter(self, role: Literal["arganno", "infreco"], key: str, value: Any, regex: bool = False) -> 'ArgannoInfrecoRequestBuilder':
        """Allow 'arganno' and 'infreco' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


class ArgannoLogrecoRequestBuilder:
    """Type-safe request builder for arganno_logreco coherence verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = ArgannoLogrecoFilterBuilder()
    
    def config_option(self, key: Literal["from_key"], value: Any) -> 'ArgannoLogrecoRequestBuilder':
        """Only allow valid config options for arganno_logreco."""
        self.config[key] = value
        return self
    
    def add_filter(self, role: Literal["arganno", "logreco"], key: str, value: Any, regex: bool = False) -> 'ArgannoLogrecoRequestBuilder':
        """Allow 'arganno' and 'logreco' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


class ArgmapInfrecoRequestBuilder:
    """Type-safe request builder for argmap_infreco coherence verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = ArgmapInfrecoFilterBuilder()
    
    def config_option(self, key: Literal["from_key"], value: Any) -> 'ArgmapInfrecoRequestBuilder':
        """Only allow valid config options for argmap_infreco."""
        self.config[key] = value
        return self
    
    def add_filter(self, role: Literal["argmap", "infreco"], key: str, value: Any, regex: bool = False) -> 'ArgmapInfrecoRequestBuilder':
        """Allow 'argmap' and 'infreco' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


class ArgmapLogrecoRequestBuilder:
    """Type-safe request builder for argmap_logreco coherence verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = ArgmapLogrecoFilterBuilder()
    
    def config_option(self, key: Literal["from_key"], value: Any) -> 'ArgmapLogrecoRequestBuilder':
        """Only allow valid config options for argmap_logreco."""
        self.config[key] = value
        return self
    
    def add_filter(self, role: Literal["argmap", "logreco"], key: str, value: Any, regex: bool = False) -> 'ArgmapLogrecoRequestBuilder':
        """Allow 'argmap' and 'logreco' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


class ArgannoArgmapLogrecoRequestBuilder:
    """Type-safe request builder for arganno_argmap_logreco coherence verifier."""
    
    def __init__(self, inputs: str, source: Optional[str] = None):
        self.inputs = inputs
        self.source = source
        self.config: Dict[str, Any] = {}
        self.filter_builder = ArgannoArgmapLogrecoFilterBuilder()
    
    def config_option(self, key: Literal["from_key"], value: Any) -> 'ArgannoArgmapLogrecoRequestBuilder':
        """Only allow valid config options for arganno_argmap_logreco."""
        self.config[key] = value
        return self
    
    def add_filter(self, role: Literal["arganno", "argmap", "logreco"], key: str, value: Any, regex: bool = False) -> 'ArgannoArgmapLogrecoRequestBuilder':
        """Allow 'arganno', 'argmap', and 'logreco' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        """Build the verification request."""
        config_dict = self.config.copy()
        filters = self.filter_builder.build()
        if filters:
            config_dict["filters"] = filters
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )


# Factory functions for creating request builders

def create_arganno_request(inputs: str, source: Optional[str] = None) -> ArgannoRequestBuilder:
    """Create a type-safe request builder for arganno verifier."""
    return ArgannoRequestBuilder(inputs, source)


def create_argmap_request(inputs: str, source: Optional[str] = None) -> ArgmapRequestBuilder:
    """Create a type-safe request builder for argmap verifier."""
    return ArgmapRequestBuilder(inputs, source)


def create_infreco_request(inputs: str, source: Optional[str] = None) -> InfrecoRequestBuilder:
    """Create a type-safe request builder for infreco verifier."""
    return InfrecoRequestBuilder(inputs, source)


def create_logreco_request(inputs: str, source: Optional[str] = None) -> LogrecoRequestBuilder:
    """Create a type-safe request builder for logreco verifier."""
    return LogrecoRequestBuilder(inputs, source)


def create_has_annotations_request(inputs: str, source: Optional[str] = None) -> HasAnnotationsRequestBuilder:
    """Create a type-safe request builder for has_annotations verifier."""
    return HasAnnotationsRequestBuilder(inputs, source)


def create_has_argdown_request(inputs: str, source: Optional[str] = None) -> HasArgdownRequestBuilder:
    """Create a type-safe request builder for has_argdown verifier."""
    return HasArgdownRequestBuilder(inputs, source)


def create_arganno_argmap_request(inputs: str, source: Optional[str] = None) -> ArgannoArgmapRequestBuilder:
    """Create a type-safe request builder for arganno_argmap coherence verifier."""
    return ArgannoArgmapRequestBuilder(inputs, source)


def create_arganno_infreco_request(inputs: str, source: Optional[str] = None) -> ArgannoInfrecoRequestBuilder:
    """Create a type-safe request builder for arganno_infreco coherence verifier."""
    return ArgannoInfrecoRequestBuilder(inputs, source)


def create_arganno_logreco_request(inputs: str, source: Optional[str] = None) -> ArgannoLogrecoRequestBuilder:
    """Create a type-safe request builder for arganno_logreco coherence verifier."""
    return ArgannoLogrecoRequestBuilder(inputs, source)


def create_argmap_infreco_request(inputs: str, source: Optional[str] = None) -> ArgmapInfrecoRequestBuilder:
    """Create a type-safe request builder for argmap_infreco coherence verifier."""
    return ArgmapInfrecoRequestBuilder(inputs, source)


def create_argmap_logreco_request(inputs: str, source: Optional[str] = None) -> ArgmapLogrecoRequestBuilder:
    """Create a type-safe request builder for argmap_logreco coherence verifier."""
    return ArgmapLogrecoRequestBuilder(inputs, source)


def create_arganno_argmap_logreco_request(inputs: str, source: Optional[str] = None) -> ArgannoArgmapLogrecoRequestBuilder:
    """Create a type-safe request builder for arganno_argmap_logreco coherence verifier."""
    return ArgannoArgmapLogrecoRequestBuilder(inputs, source)
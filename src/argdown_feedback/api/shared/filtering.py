"""
Filter builder system for type-safe filter construction.

Provides builders for creating metadata filters with compile-time type safety
and runtime validation.
"""

from typing import Dict, List, Any, Literal
from dataclasses import dataclass

# Define allowed filter roles for each verifier using Literal types
InfrecoRoles = Literal["infreco"]
ArgannoRoles = Literal["arganno"] 
ArgmapRoles = Literal["argmap"]
LogrecoRoles = Literal["logreco"]
ArgmapInfrecoRoles = Literal["argmap", "infreco"]
ArgannoArgmapRoles = Literal["arganno", "argmap"]
ArgannoInfrecoRoles = Literal["arganno", "infreco"]
ArgannoLogrecoRoles = Literal["arganno", "logreco"]
ArgmapLogrecoRoles = Literal["argmap", "logreco"]
ArgannoArgmapLogrecoRoles = Literal["arganno", "argmap", "logreco"]


@dataclass
class FilterRule:
    """Single filter rule for metadata matching."""
    key: str
    value: Any
    regex: bool = False


class FilterBuilder:
    """Base filter builder for creating metadata filters."""
    
    def __init__(self):
        """Initialize empty filter builder."""
        self._filters: Dict[str, List[FilterRule]] = {}
    
    def add(self, role: str, key: str, value: Any, regex: bool = False) -> 'FilterBuilder':
        """
        Add a filter rule for a role.
        
        Args:
            role: Filter role (arganno, argmap, infreco, logreco)
            key: Metadata key to match
            value: Value to match against
            regex: Whether to use regex matching
            
        Returns:
            Self for method chaining
        """
        if role not in self._filters:
            self._filters[role] = []
        self._filters[role].append(FilterRule(key=key, value=value, regex=regex))
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build the final filters dictionary.
        
        Returns:
            Filters dictionary suitable for API requests
        """
        result: Dict[str, Any] = {}
        for role, rules in self._filters.items():
            if len(rules) == 1 and not rules[0].regex:
                # Simple format for single exact match
                result[role] = {rules[0].key: rules[0].value}
            else:
                # Advanced format for multiple rules or regex
                result[role] = [
                    {"key": rule.key, "value": rule.value, "regex": rule.regex}
                    for rule in rules
                ]
        return result


# Type-safe filter builders for specific verifier combinations

class InfrecoFilterBuilder:
    """Type-safe filter builder for infreco verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: InfrecoRoles, key: str, value: Any, regex: bool = False) -> 'InfrecoFilterBuilder':
        """Add filter rule for infreco role only."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()


class ArgannoFilterBuilder:
    """Type-safe filter builder for arganno verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: ArgannoRoles, key: str, value: Any, regex: bool = False) -> 'ArgannoFilterBuilder':
        """Add filter rule for arganno role only."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()


class ArgmapFilterBuilder:
    """Type-safe filter builder for argmap verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: ArgmapRoles, key: str, value: Any, regex: bool = False) -> 'ArgmapFilterBuilder':
        """Add filter rule for argmap role only."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()


class LogrecoFilterBuilder:
    """Type-safe filter builder for logreco verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: LogrecoRoles, key: str, value: Any, regex: bool = False) -> 'LogrecoFilterBuilder':
        """Add filter rule for logreco role only."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()


class ArgmapInfrecoFilterBuilder:
    """Type-safe filter builder for argmap_infreco coherence verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: ArgmapInfrecoRoles, key: str, value: Any, regex: bool = False) -> 'ArgmapInfrecoFilterBuilder':
        """Add filter rule for argmap or infreco roles."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()


class ArgannoArgmapFilterBuilder:
    """Type-safe filter builder for arganno_argmap coherence verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: ArgannoArgmapRoles, key: str, value: Any, regex: bool = False) -> 'ArgannoArgmapFilterBuilder':
        """Add filter rule for arganno or argmap roles."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()


class ArgannoInfrecoFilterBuilder:
    """Type-safe filter builder for arganno_infreco coherence verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: ArgannoInfrecoRoles, key: str, value: Any, regex: bool = False) -> 'ArgannoInfrecoFilterBuilder':
        """Add filter rule for arganno or infreco roles."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()


class ArgannoLogrecoFilterBuilder:
    """Type-safe filter builder for arganno_logreco coherence verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: ArgannoLogrecoRoles, key: str, value: Any, regex: bool = False) -> 'ArgannoLogrecoFilterBuilder':
        """Add filter rule for arganno or logreco roles."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()


class ArgmapLogrecoFilterBuilder:
    """Type-safe filter builder for argmap_logreco coherence verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: ArgmapLogrecoRoles, key: str, value: Any, regex: bool = False) -> 'ArgmapLogrecoFilterBuilder':
        """Add filter rule for argmap or logreco roles."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()


class ArgannoArgmapLogrecoFilterBuilder:
    """Type-safe filter builder for arganno_argmap_logreco coherence verifier."""
    
    def __init__(self):
        self._builder = FilterBuilder()
    
    def add(self, role: ArgannoArgmapLogrecoRoles, key: str, value: Any, regex: bool = False) -> 'ArgannoArgmapLogrecoFilterBuilder':
        """Add filter rule for arganno, argmap, or logreco roles."""
        self._builder.add(role, key, value, regex)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final filters dictionary."""
        return self._builder.build()
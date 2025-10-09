from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, List, TypeAlias
import logging
from bs4 import BeautifulSoup
from pyargdown import Argdown


class VerificationDType(Enum):
    """Types of primary verification data."""
    argdown = "argdown"
    xml = "xml"

@dataclass 
class VerificationConfig:
    """Global configuration for verification checks."""

@dataclass
class PrimaryVerificationData:
    """Primary verification data, parsed from fenced codeblocks"""
    id: str
    dtype: VerificationDType
    data: Argdown | BeautifulSoup | None = None
    code_snippet: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

# Define a custom type for data filters
VDFilter: TypeAlias = Callable[[PrimaryVerificationData], bool]


@dataclass
class VerificationResult:
    """Results of a verification check."""
    verifier_id: str
    verification_data_references: List[str]  # Coherence requests assess the alignment between multiple data items
    is_valid: bool
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerificationRequest:
    """
    Standard request format for verification handlers.
    Contains all data needed for verification operations.
    """
    # Raw input data, will be evaluated
    inputs: str

    # Original source text against which the verification is performed
    source: str | None = None

    # Primary verification data (parsed from inputs)
    verification_data: List[PrimaryVerificationData] = field(default_factory=list)
    
    # Verification state
    results: List[VerificationResult] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration options
    config: VerificationConfig = field(default_factory=VerificationConfig)
    
    # Processing flags
    continue_processing: bool = True
    
    # For tracking handler execution
    executed_handlers: List[str] = field(default_factory=list)
    
    def add_result(self, handler_name: str, verification_data_references: List[str], is_valid: bool, 
                   message: Optional[str] = None, details: Dict[str, Any] | None = None) -> None:
        """Add a verification result to the request."""
        if handler_name in self.results:
            logging.warning(f"Handler {handler_name} already executed. Overwriting result.")
        self.results.append(VerificationResult(
            verifier_id=handler_name,
            verification_data_references=verification_data_references,
            is_valid=is_valid,
            message=message,
            details=details or {}
        ))

    def add_result_record(self, vresult: VerificationResult) -> None:
        """Add a verification result to the request."""
        self.results.append(vresult)
        
    def merge_results(self, other_request: 'VerificationRequest') -> None:
        """Merge results from another request inplace."""
        self.results.extend(other_request.results)
        self.artifacts.update(other_request.artifacts)
        self.executed_handlers.extend(other_request.executed_handlers)
        self.continue_processing = self.continue_processing and other_request.continue_processing
                        
    def is_valid(self) -> bool:
        """Check if all verification results are valid."""
        return all(result.is_valid for result in self.results)
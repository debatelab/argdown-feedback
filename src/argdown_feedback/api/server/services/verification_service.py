"""
Core verification service that bridges API models to verifier handlers.

This service handles the async execution of CPU-intensive verification tasks
using ThreadPoolExecutor while maintaining FastAPI's async capabilities.
"""

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ...shared.models import VerificationRequest, VerificationResponse
from ...shared.exceptions import VerifierNotFoundError, VerificationError
from ....verifiers.verification_request import VerificationRequest as InternalRequest
from argdown_feedback.api.server.services import verifier_registry

logger = logging.getLogger(__name__)


class VerificationService:
    """
    Service for executing verification tasks asynchronously.
    
    Uses ThreadPoolExecutor to run CPU-intensive verification tasks in background
    threads while keeping FastAPI endpoints async.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize verification service.
        
        Args:
            max_workers: Maximum number of worker threads for verification tasks
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"VerificationService initialized with {max_workers} workers")
    
    async def verify_async(
        self, 
        verifier_name: str, 
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Execute verification asynchronously.
        
        Args:
            verifier_name: Name of the verifier to use
            request: API verification request
            
        Returns:
            Verification response with results
            
        Raises:
            VerifierNotFoundError: If verifier doesn't exist
            VerificationError: If verification fails
        """
        start_time = time.time()
        
        # Validate verifier exists
        try:
            verifier_registry.get_verifier_info(verifier_name)
        except KeyError:
            available = verifier_registry.list_verifiers()
            raise VerifierNotFoundError(verifier_name, available)
        
        try:
            # Run verification in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._verify_sync,
                verifier_name,
                request
            )
            
            # Add processing time to response
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            
            logger.info(
                f"Verification completed: {verifier_name} in {processing_time:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed for {verifier_name}: {e}", exc_info=True)
            raise VerificationError(f"Verification failed: {str(e)}")
    
    def _verify_sync(
        self, 
        verifier_name: str, 
        request: VerificationRequest
    ) -> VerificationResponse:
        """
        Execute verification synchronously in worker thread.
        
        Args:
            verifier_name: Name of the verifier to use
            request: API verification request
            
        Returns:
            Verification response with results
        """
        # Get verifier handler
        handler_kwargs = request.config if request.config else {}
        if "filters" not in handler_kwargs:
            handler_kwargs['filters'] = None
        handler = verifier_registry.create_handler(verifier_name, **handler_kwargs)
        
        # Build internal verification request
        internal_request = self._build_internal_request(request, verifier_name)
        
        # Execute verification
        result = handler.process(internal_request)
        
        # Remove `details` from results if present, as these might not be serializable`
        if hasattr(result, 'results'):
            for r in result.results:
                if hasattr(r, 'details'):
                    r.details = {}

        # Transform result to API format
        return self._build_response(verifier_name, result, request)
    
    def _build_internal_request(
        self, 
        request: VerificationRequest, 
        verifier_name: str
    ) -> InternalRequest:
        """
        Build internal verification request from API request.
        
        Args:
            request: API verification request
            verifier_name: Name of the verifier
            
        Returns:
            Internal verification request object
        """
        ## Extract verification data from inputs
        #verification_data = self._extract_verification_data(request.inputs)
        
        # Build internal request
        internal_request = InternalRequest(
            inputs=request.inputs,
            source=request.source,
            #verification_data=verification_data
        )
        
        # Add verifier-specific configuration
        if request.config:
            # Apply configuration to internal request
            for key, value in request.config.items():
                if hasattr(internal_request, key):
                    setattr(internal_request, key, value)
                # For filters and other config, they will be handled by the existing verification system
        
        return internal_request
    
    # def _extract_verification_data(self, inputs: str) -> List[PrimaryVerificationData]:
    #     """
    #     Extract verification data from input string.
        
    #     This is a simplified version - in the full implementation,
    #     this would use the existing code block extraction logic.
        
    #     Args:
    #         inputs: Input string containing code blocks
            
    #     Returns:
    #         List of verification data objects
    #     """
    #     # TODO: Integrate with existing FencedCodeBlockExtractor
    #     # For now, create a simple mock implementation
        
    #     verification_data = []
        
    #     # Simple regex-based extraction (placeholder)
    #     import re
        
    #     # Extract argdown code blocks
    #     argdown_pattern = r'```argdown\s*\n(.*?)\n```'
    #     argdown_matches = re.findall(argdown_pattern, inputs, re.DOTALL)
        
    #     for i, content in enumerate(argdown_matches):
    #         vdata = PrimaryVerificationData(
    #             id=f"argdown_{i}",
    #             dtype=VerificationDType.argdown,
    #             code_snippet=content.strip(),
    #             metadata={"filename": f"block_{i}.ad"}
    #         )
    #         verification_data.append(vdata)
        
    #     # Extract XML code blocks  
    #     xml_pattern = r'```xml\s*\n(.*?)\n```'
    #     xml_matches = re.findall(xml_pattern, inputs, re.DOTALL)
        
    #     for i, content in enumerate(xml_matches):
    #         vdata = PrimaryVerificationData(
    #             id=f"xml_{i}",
    #             dtype=VerificationDType.xml,
    #             code_snippet=content.strip(),
    #             metadata={"filename": f"annotations_{i}.xml"}
    #         )
    #         verification_data.append(vdata)
        
    #     if not verification_data:
    #         # If no code blocks found, treat entire input as argdown
    #         vdata = PrimaryVerificationData(
    #             id="input_0",
    #             dtype=VerificationDType.argdown,
    #             code_snippet=inputs.strip(),
    #             metadata={"filename": "input.ad"}
    #         )
    #         verification_data.append(vdata)
        
    #     return verification_data
    
    def _build_response(
        self, 
        verifier_name: str, 
        result: Any, 
        original_request: VerificationRequest
    ) -> VerificationResponse:
        """
        Build API response from internal verification result.
        
        Args:
            verifier_name: Name of the verifier used
            result: Internal verification result
            original_request: Original API request
            
        Returns:
            API verification response
        """
        # Extract result data
        is_valid = result.is_valid() if hasattr(result, 'is_valid') else True
        
        # Convert verification data to API format
        verification_data_objects = []
        if hasattr(result, 'verification_data'):
            for vd in result.verification_data:
                from ...shared.models import VerificationData
                vd_obj = VerificationData(
                    id=vd.id,
                    dtype=vd.dtype.value if hasattr(vd.dtype, 'value') else str(vd.dtype),
                    code_snippet=vd.code_snippet,
                    metadata=vd.metadata or {}
                )
                verification_data_objects.append(vd_obj)
        
        # Convert results to API format
        results_dicts = []
        if hasattr(result, 'results'):
            for r in result.results:
                if hasattr(r, '__dict__'):
                    results_dicts.append(r.__dict__)
                else:
                    results_dicts.append(str(r))
        
        # Get executed handlers
        executed_handlers = []
        if hasattr(result, 'executed_handlers'):
            executed_handlers = result.executed_handlers
        
        return VerificationResponse(
            verifier=verifier_name,
            is_valid=is_valid,
            verification_data=verification_data_objects,
            results=results_dicts,
            executed_handlers=executed_handlers,
            processing_time_ms=0.0  # Will be set by caller
        )
    
    def __del__(self):
        """Cleanup thread pool on service destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Global service instance
verification_service = VerificationService(max_workers=8)
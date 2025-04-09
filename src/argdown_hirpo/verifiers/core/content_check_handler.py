from typing import Optional
import logging



from argdown_hirpo.verifiers.verification_request import (
    VerificationRequest,
    VerificationDType,
    VerificationResult,
    VDFilter,
)
from argdown_hirpo.verifiers.base import BaseHandler


class HasAnnotationsHandler(BaseHandler):
    """Handler that checks if the input data has annotations."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name, logger)

    def handle(self, request: VerificationRequest) -> VerificationRequest:
        message = None
        if not any(
            vdata.dtype == VerificationDType.xml and vdata.code_snippet
            for vdata in request.verification_data
        ):
            error_msg = "Input data has no properly formatted fenced codeblocks with annotations."
            if request.inputs.count("```xml") == 0:
                error_msg += " No fenced code block starting with '```xml'."
            if "```\n" not in request.inputs:
                error_msg += " No closing '```'."
            message = error_msg
            
        vresult = VerificationResult(
            verifier_id=self.name,
            verification_data_references=[],
            is_valid=message is None,
            message=message,
        )
        request.add_result_record(vresult)

        return request



class HasArgdownHandler(BaseHandler):
    """Handler that checks if the input data has argdown snippets."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        filter: Optional[VDFilter] = None
    ):
        super().__init__(name, logger)
        if filter is None:
            filter = lambda x: True  # noqa: E731
        self.filter = filter
        
    def handle(self, request: VerificationRequest) -> VerificationRequest:
        message = None
        if not any(
            self.filter(vdata)
            and vdata.dtype == VerificationDType.argdown
            and vdata.code_snippet
            for vdata in request.verification_data
        ):
            error_msg = "Input data has no properly formatted fenced codeblocks with Argdown code."
            if request.inputs.count("```argdown") == 0:
                error_msg += " No fenced code block starting with '```argdown'."
            if "```\n" not in request.inputs:
                error_msg += " No closing '```'."
            message = error_msg
            
        vresult = VerificationResult(
            verifier_id=self.name,
            verification_data_references=[],
            is_valid=message is None,
            message=message,
        )
        request.add_result_record(vresult)

        return request

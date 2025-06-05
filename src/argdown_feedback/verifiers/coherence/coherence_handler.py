from abc import abstractmethod

from ..verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationResult,
)
from argdown_feedback.verifiers.base import BaseHandler


class CoherenceHandler(BaseHandler):
    """Base handler interface for evaluating coherence of different primary data instances."""

    @abstractmethod
    def evaluate(
        self,
        vdata1: PrimaryVerificationData,
        vdata2: PrimaryVerificationData,
        ctx: VerificationRequest,
    ) -> VerificationResult | None:
        """Evaluate the data and return a verification result."""

    @abstractmethod
    def is_applicable(
        self,
        vdata1: PrimaryVerificationData,
        vdata2: PrimaryVerificationData,
        ctx: VerificationRequest,
    ) -> bool:
        """Check if the handler is applicable to the given data pair. Needs to be customized in subclasses."""

    def handle(self, request: VerificationRequest) -> VerificationRequest:
        for i in range(len(request.verification_data)):
            for j in range(i + 1, len(request.verification_data)):
                vdata1 = request.verification_data[i]
                vdata2 = request.verification_data[j]
                if vdata1.data is None or vdata2.data is None:
                    continue
                if self.is_applicable(vdata1, vdata2, request):
                    vresult = self.evaluate(vdata1, vdata2, request)
                    if vresult is not None:
                        request.add_result_record(vresult)
        return request

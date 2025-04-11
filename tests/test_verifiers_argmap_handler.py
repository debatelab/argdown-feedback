from pprint import pprint
import pytest
import copy
from textwrap import dedent

from pyargdown import parse_argdown

from argdown_feedback.verifiers.core.argmap_handler import (
    ArgMapHandler,
    CompleteClaimsHandler,
    NoDuplicateLabelsHandler,
    NoPCSHandler,
    ArgMapCompositeHandler
)
from argdown_feedback.verifiers.verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
    VerificationResult
)

def parse_fenced_argdown(argdown_text: str):
    argdown_text = argdown_text.strip("\n ")
    argdown_text = "\n".join(argdown_text.splitlines()[1:-1])
    return parse_argdown(argdown_text)

@pytest.fixture
def valid_argdown_text():
    return dedent("""
    ```argdown
    [No meat]: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
        <+ <Climate change>: Animal farming causes climate change.
    ```
    """)


@pytest.fixture
def valid_argdown_graph(valid_argdown_text):
    # Parse the Argdown text into a graph
    return parse_fenced_argdown(valid_argdown_text)


@pytest.fixture
def incomplete_claims_argdown():
    return dedent("""
    ```argdown
    [No meat]: We should stop eating meat.
        <+ : Animals suffer.
        <+ <Climate change>: Animal farming causes climate change.
    ```
    """)


@pytest.fixture
def incomplete_claims_graph(incomplete_claims_argdown):
    return parse_fenced_argdown(incomplete_claims_argdown)


@pytest.fixture
def duplicate_labels_argdown():
    return dedent("""
    ```argdown
    [No meat]: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
        <+ <Suffering>: Animal farming causes climate change.
    ```
    """)


@pytest.fixture
def duplicate_labels_graph(duplicate_labels_argdown):
    return parse_fenced_argdown(duplicate_labels_argdown)


@pytest.fixture
def pcs_structure_argdown():
    return dedent("""
    ```argdown
    <Suffering>: Animals suffer.
    
    (1) Animals suffer.
    (2) If animals suffer, we should not eat them.
    -----
    (3) We should not eat animals.
    ```
    """)


@pytest.fixture
def pcs_structure_graph(pcs_structure_argdown):
    return parse_fenced_argdown(pcs_structure_argdown)


@pytest.fixture
def verification_request(valid_argdown_graph):
    source = "We should stop eating meat. Animals suffer. Animal farming causes climate change."
    verification_data = [
        PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_argdown_graph)
    ]
    request = VerificationRequest(inputs="", source=source, verification_data=verification_data)
    return request


def test_argmap_handler_is_applicable():
    handler = CompleteClaimsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=None)
    request = VerificationRequest(inputs="")
    
    assert handler.is_applicable(vdata, request) is True
    
    vdata.dtype = VerificationDType.xml
    assert handler.is_applicable(vdata, request) is False


def test_complete_claims_handler_valid(valid_argdown_graph):
    handler = CompleteClaimsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_argdown_graph)
    request = VerificationRequest(inputs="", verification_data=[vdata])
    
    result = handler.evaluate(vdata, request)
    print(valid_argdown_graph)
    pprint(result)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_complete_claims_handler_invalid(incomplete_claims_graph):
    handler = CompleteClaimsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=incomplete_claims_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Missing labels" in result.message


def test_no_duplicate_labels_handler_valid(valid_argdown_graph):
    handler = NoDuplicateLabelsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_argdown_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_no_duplicate_labels_handler_invalid(duplicate_labels_graph):
    handler = NoDuplicateLabelsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=duplicate_labels_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Duplicate labels" in result.message


def test_no_pcs_handler_valid(valid_argdown_graph):
    handler = NoPCSHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_argdown_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_no_pcs_handler_invalid(pcs_structure_graph):
    handler = NoPCSHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=pcs_structure_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Found detailed reconstruction" in result.message


def test_composite_handler(verification_request):
    # Create a handler with an invalid component
    incomplete_graph = parse_fenced_argdown("""
    ```argdown
    [No meat]: We should stop eating meat.
        <+ : Animals suffer.
    ```
    """)
    vdata_incomplete = PrimaryVerificationData(id="incomplete", dtype=VerificationDType.argdown, data=incomplete_graph)
    request = copy.deepcopy(verification_request)
    request.verification_data.append(vdata_incomplete)
    
    composite = ArgMapCompositeHandler()
    composite.process(request)
    
    # Should find issues in the incomplete claims data
    assert len(request.results) > 0
    # Check that the complete claims handler found the issue
    incomplete_results = [r for r in request.results if r.message is not None and "Missing labels" in r.message]
    assert len(incomplete_results) > 0


def test_handle_none_data():
    handlers = [
        CompleteClaimsHandler(),
        NoDuplicateLabelsHandler(),
        NoPCSHandler()
    ]
    
    for handler in handlers:
        vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=None)
        request = VerificationRequest(inputs="")
        result = handler.evaluate(vdata, request)
        assert result is None


def test_handle_invalid_data_type():
    handlers = [
        CompleteClaimsHandler(),
        NoDuplicateLabelsHandler(),
        NoPCSHandler()
    ]
    
    for handler in handlers:
        vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data="not a graph")
        request = VerificationRequest(inputs="")
        with pytest.raises(TypeError):
            handler.evaluate(vdata, request)


def test_argmap_handler_handle_method():
    request = VerificationRequest(inputs="")
    graph = parse_fenced_argdown("""
    ```argdown
    [No meat]: We should stop eating meat.
    ```
    """)
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=graph)
    request.verification_data.append(vdata)
    
    # Create a custom handler that's always valid
    class TestHandler(ArgMapHandler):
        def evaluate(self, vdata, ctx):
            return VerificationResult(
                verifier_id="test",
                verification_data_references=[vdata.id],
                is_valid=True,
                message=None
            )
    
    handler = TestHandler()
    handler.process(request)
    
    assert len(request.results) == 1
    assert request.results[0].is_valid is True


def test_real_world_example_argmap():
    argdown_text = dedent("""
    ```argdown
    [No meat]: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
        <+ <Climate change>: Animal farming causes climate change.
    ```
    """)
    
    graph = parse_fenced_argdown(argdown_text)
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=graph)
    source = "We should stop eating meat. Animals suffer. Animal farming causes climate change."
    request = VerificationRequest(inputs="", source=source)
    request.verification_data.append(vdata)
    
    composite = ArgMapCompositeHandler()
    composite.process(request)
    
    # All validations should pass
    invalid_results = [r for r in request.results if not r.is_valid]
    assert len(invalid_results) == 0
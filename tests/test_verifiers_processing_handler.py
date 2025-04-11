from pprint import pprint
import pytest
import copy
from textwrap import dedent
from bs4 import BeautifulSoup

from pyargdown import ArgdownMultiDiGraph

from argdown_feedback.verifiers.processing_handler import (
    ProcessingHandler,
    FencedCodeBlockExtractor,
    ArgdownParser,
    XMLParser,
    DefaultProcessingHandler,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationRequest,
    VerificationDType,
)


@pytest.fixture
def argdown_input_text():
    return dedent("""
    Here is an argument map:
    
    ```argdown
    [No meat]: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
        <+ <Climate change>: Animal farming causes climate change.
    ```
    """)


@pytest.fixture
def xml_input_text():
    return dedent("""
    Here is an XML annotation:
    
    ```xml
    <proposition id="1">We should stop eating meat.</proposition>
    <proposition id="2" supports="1">Animals suffer.</proposition>
    <proposition id="3" attacks="2">Some animals are raised humanely.</proposition>
    ```
    """)


@pytest.fixture
def mixed_input_text(argdown_input_text, xml_input_text):
    return argdown_input_text + "\n\nAnd here is some XML markup:\n\n" + xml_input_text


@pytest.fixture
def input_with_metadata():
    return dedent("""
    Here is an argument map with metadata:
    
    ```argdown {"type": "map", "author": "test"}
    [No meat]: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
    ```
    """)


@pytest.fixture
def invalid_argdown_input():
    return dedent("""
    Here is invalid Argdown:
    
    ```argdown
    [No meat: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
      - <Climate change>: Animal farming causes climate change.
    ```
    """)


@pytest.fixture
def invalid_xml_input():
    return dedent("""
    Here is invalid XML:
    
    ```xml
    <proposition id="1">We should stop eating meat.</proposition>
    <proposition id="2" supports="1">Animals suffer.
    ```
    """)


@pytest.fixture
def verification_request(argdown_input_text):
    return VerificationRequest(inputs=argdown_input_text, source=None)


def test_fenced_code_block_extractor_argdown(verification_request):
    handler = FencedCodeBlockExtractor()
    result_request = handler.process(verification_request)

    assert len(result_request.verification_data) == 1
    assert result_request.verification_data[0].dtype == VerificationDType.argdown
    assert result_request.verification_data[0].code_snippet is not None
    assert "```argdown" in result_request.verification_data[0].code_snippet
    assert result_request.verification_data[0].data is None


def test_fenced_code_block_extractor_xml():
    request = VerificationRequest(
        inputs=dedent("""
    ```xml
    <proposition id="1">Test</proposition>
    ```
    """),
        source=None,
    )

    handler = FencedCodeBlockExtractor()
    result_request = handler.process(request)

    assert len(result_request.verification_data) == 1
    assert result_request.verification_data[0].dtype == VerificationDType.xml
    assert result_request.verification_data[0].code_snippet is not None
    assert "```xml" in result_request.verification_data[0].code_snippet
    assert result_request.verification_data[0].data is None


def test_fenced_code_block_extractor_mixed(mixed_input_text):
    request = VerificationRequest(inputs=mixed_input_text, source=None)

    handler = FencedCodeBlockExtractor()
    result_request = handler.process(request)

    assert len(result_request.verification_data) == 2

    argdown_data = next(
        (
            d
            for d in result_request.verification_data
            if d.dtype == VerificationDType.argdown
        ),
        None,
    )
    xml_data = next(
        (
            d
            for d in result_request.verification_data
            if d.dtype == VerificationDType.xml
        ),
        None,
    )

    assert argdown_data is not None
    assert xml_data is not None

    assert "```argdown" in argdown_data.code_snippet
    assert "```xml" in xml_data.code_snippet


def test_fenced_code_block_extractor_with_metadata(input_with_metadata):
    request = VerificationRequest(inputs=input_with_metadata, source=None)

    handler = FencedCodeBlockExtractor()
    result_request = handler.process(request)

    assert len(result_request.verification_data) == 1
    assert result_request.verification_data[0].metadata is not None
    assert result_request.verification_data[0].metadata.get("type") == "map"
    assert result_request.verification_data[0].metadata.get("author") == "test"


def test_fenced_code_block_extractor_no_codeblocks():
    request = VerificationRequest(
        inputs="Just some plain text with no code blocks.", source=None
    )

    handler = FencedCodeBlockExtractor()
    result_request = handler.process(request)

    assert len(result_request.verification_data) == 0


def test_argdown_parser_valid(argdown_input_text):
    request = VerificationRequest(inputs=argdown_input_text, source=None)

    extractor = FencedCodeBlockExtractor()
    parser = ArgdownParser()

    request = extractor.process(request)
    request = parser.process(request)

    assert len(request.verification_data) == 1
    assert request.verification_data[0].data is not None
    assert isinstance(request.verification_data[0].data, ArgdownMultiDiGraph)
    assert len(request.results) == 0  # No errors


def test_argdown_parser_invalid(invalid_argdown_input):
    request = VerificationRequest(inputs=invalid_argdown_input, source=None)

    extractor = FencedCodeBlockExtractor()
    parser = ArgdownParser()

    request = extractor.process(request)
    request = parser.process(request)

    assert len(request.verification_data) == 1
    assert request.verification_data[0].data is None
    assert len(request.results) == 1  # Should have an error result
    assert not request.results[0].is_valid
    assert "Failed to parse argdown" in request.results[0].message


def test_xml_parser_valid(xml_input_text):
    request = VerificationRequest(inputs=xml_input_text, source=None)

    extractor = FencedCodeBlockExtractor()
    parser = XMLParser()

    request = extractor.process(request)
    request = parser.process(request)
    assert len(request.verification_data) == 1
    assert request.verification_data[0].data is not None
    assert isinstance(request.verification_data[0].data, BeautifulSoup)
    assert len(request.results) == 0  # No errors


def test_xml_parser_invalid(invalid_xml_input):
    """Even 'inmvalid' XML should be parsed to a BeautifulSoup object."""
    request = VerificationRequest(inputs=invalid_xml_input, source=None)

    extractor = FencedCodeBlockExtractor()
    parser = XMLParser()

    request = extractor.process(request)
    request = parser.process(request)

    assert len(request.verification_data) == 1
    assert request.verification_data[0].data is not None
    assert len(request.results) == 0  # Should have an error result


def test_parsers_skip_already_parsed():
    xml_text = dedent("""
    ```xml
    <proposition id="1">Test</proposition>
    ```
    """)
    request = VerificationRequest(inputs=xml_text, source=None)

    # First parse normally
    extractor = FencedCodeBlockExtractor()
    xml_parser = XMLParser()

    request = extractor.process(request)
    request = xml_parser.process(request)

    # Now create a deep copy and try to parse again
    request_copy = copy.deepcopy(request)
    request_copy = xml_parser.process(request_copy)

    # The data should be identical in both requests
    assert request.verification_data[0].data == request_copy.verification_data[0].data


def test_composite_processing_handler(mixed_input_text):
    request = VerificationRequest(inputs=mixed_input_text, source=None)

    composite = DefaultProcessingHandler()
    result_request = composite.process(request)

    # Should have extracted and parsed both code blocks
    assert len(result_request.verification_data) == 2

    argdown_data = next(
        (
            d
            for d in result_request.verification_data
            if d.dtype == VerificationDType.argdown
        ),
        None,
    )
    xml_data = next(
        (
            d
            for d in result_request.verification_data
            if d.dtype == VerificationDType.xml
        ),
        None,
    )

    assert argdown_data is not None and argdown_data.data is not None
    assert xml_data is not None and xml_data.data is not None
    assert isinstance(argdown_data.data, ArgdownMultiDiGraph)
    assert isinstance(xml_data.data, BeautifulSoup)


def test_composite_processing_handler_with_errors(invalid_argdown_input):
    request = VerificationRequest(inputs=invalid_argdown_input, source=None)

    composite = DefaultProcessingHandler()
    result_request = composite.process(request)

    # Should have extracted but failed to parse
    assert len(result_request.verification_data) == 1
    assert result_request.verification_data[0].data is None
    assert len(result_request.results) == 1
    assert not result_request.results[0].is_valid
    assert "Failed to parse" in result_request.results[0].message


def test_composite_handler_with_custom_handlers():
    # Create a custom handler that just logs its execution
    class CustomHandler(ProcessingHandler):
        def handle(self, request):
            request.executed_handlers.append("CustomHandler")
            return request

    custom_handler = CustomHandler()
    composite = DefaultProcessingHandler(handlers=[custom_handler])

    request = VerificationRequest(inputs="test", source=None)
    result_request = composite.process(request)

    assert "CustomHandler" in result_request.executed_handlers


def test_list_inputs_handling():
    inputs = ["First input", "Second input with ```argdown\n[P]: Test.\n```"]
    request = VerificationRequest(inputs=inputs, source=None)

    extractor = FencedCodeBlockExtractor()
    result_request = extractor.process(request)

    assert len(result_request.verification_data) == 1
    assert result_request.verification_data[0].dtype == VerificationDType.argdown

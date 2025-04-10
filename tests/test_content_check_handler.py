import copy
from pprint import pprint
import pytest
from textwrap import dedent
import uuid

from argdown_feedback.verifiers.core.content_check_handler import HasAnnotationsHandler, HasArgdownHandler
from argdown_feedback.verifiers.verification_request import (
    VerificationRequest, 
    VerificationResult,
    PrimaryVerificationData,
    VerificationDType,
)


@pytest.fixture
def xml_input():
    return dedent("""
    Some text before

    ```xml
    <proposition id="1">We should stop eating meat.</proposition>
    <proposition id="2" supports="1">Animals suffer.</proposition>
    ```

    Some text after
    """)


@pytest.fixture
def no_xml_input():
    return dedent("""
    Some text without any xml codeblocks.

    ```
    This is just a regular code block.
    ```
    """)


@pytest.fixture
def incomplete_xml_input():
    return dedent("""
    ```xml
    <proposition id="1">We should stop eating meat.</proposition>
    """)  # Missing closing ```


@pytest.fixture
def argdown_input():
    return dedent("""
    Some text before

    ```argdown
    [No meat]: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
    ```

    Some text after
    """)


@pytest.fixture
def no_argdown_input():
    return dedent("""
    Some text without any argdown codeblocks.

    ```
    This is just a regular code block.
    ```
    """)


@pytest.fixture
def incomplete_argdown_input():
    return dedent("""
    ```argdown
    [No meat]: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
    """)  # Missing closing ```


@pytest.fixture
def mixed_input():
    return dedent("""
    ```xml
    <proposition id="1">We should stop eating meat.</proposition>
    <proposition id="2" supports="1">Animals suffer.</proposition>
    ```

    ```argdown
    [No meat]: We should stop eating meat.
        <+ <Suffering>: Animals suffer.
    ```
    """)


def test_has_annotations_handler_with_xml():
    request = VerificationRequest(inputs=dedent("""
    ```xml
    <proposition id="1">We should stop eating meat.</proposition>
    ```
    """), source=None)
    
    # Add verification data as if it had been processed by the FencedCodeBlockExtractor
    request.verification_data.append(
        PrimaryVerificationData(
            id=str(uuid.uuid4()),
            dtype=VerificationDType.xml,
            code_snippet=dedent("""
            ```xml
            <proposition id="1">We should stop eating meat.</proposition>
            ```
            """),
            data=None
        )
    )
    
    handler = HasAnnotationsHandler(name="xml_checker")
    result = handler.handle(request)
    
    assert len(result.results) == 1
    assert result.results[0].is_valid
    assert result.results[0].message is None
    assert result.results[0].verifier_id == "xml_checker"


def test_has_annotations_handler_without_xml():
    request = VerificationRequest(inputs="No XML annotations here", source=None)
    
    # No verification data added
    
    handler = HasAnnotationsHandler(name="xml_checker")
    result = handler.handle(request)
    
    assert len(result.results) == 1
    assert not result.results[0].is_valid
    assert "no properly formatted fenced codeblocks with annotations" in result.results[0].message
    assert "No fenced code block starting with '```xml'" in result.results[0].message


def test_has_annotations_handler_with_incomplete_xml():
    request = VerificationRequest(inputs="```xml\n<proposition id=\"1\">Test</proposition>", source=None)
    
    # No verification data (extractor would fail due to incomplete block)
    
    handler = HasAnnotationsHandler(name="xml_checker")
    result = handler.handle(request)
    
    assert len(result.results) == 1
    assert not result.results[0].is_valid
    assert "No closing '```'" in result.results[0].message


def test_has_argdown_handler_with_argdown():
    request = VerificationRequest(inputs=dedent("""
    ```argdown
    [Claim]: This is a claim.
    ```
    """), source=None)
    
    # Add verification data as if it had been processed by the FencedCodeBlockExtractor
    request.verification_data.append(
        PrimaryVerificationData(
            id=str(uuid.uuid4()),
            dtype=VerificationDType.argdown,
            code_snippet=dedent("""
            ```argdown
            [Claim]: This is a claim.
            ```
            """),
            data=None
        )
    )
    
    handler = HasArgdownHandler(name="argdown_checker")
    result = handler.handle(request)
    
    assert len(result.results) == 1
    assert result.results[0].is_valid
    assert result.results[0].message is None
    assert result.results[0].verifier_id == "argdown_checker"


def test_has_argdown_handler_without_argdown():
    request = VerificationRequest(inputs="No Argdown code here", source=None)
    
    # No verification data added
    
    handler = HasArgdownHandler(name="argdown_checker")
    result = handler.handle(request)
    
    assert len(result.results) == 1
    assert not result.results[0].is_valid
    assert "no properly formatted fenced codeblocks with Argdown code" in result.results[0].message
    assert "No fenced code block starting with '```argdown'" in result.results[0].message


def test_has_argdown_handler_with_incomplete_argdown():
    request = VerificationRequest(inputs="```argdown\n[Claim]: This is a claim.", source=None)
    
    # No verification data (extractor would fail due to incomplete block)
    
    handler = HasArgdownHandler(name="argdown_checker")
    result = handler.handle(request)
    
    assert len(result.results) == 1
    assert not result.results[0].is_valid
    assert "No closing '```'" in result.results[0].message


def test_has_argdown_handler_with_filter():
    

    request = VerificationRequest(inputs=dedent("""
    ```argdown {filename="map.ad"}
    [Claim]: This is a claim.
    ```
                                                
    ```argdown {filename="reconstructions.ad"}
    [Claim]: This is a reconstructed claim.
    ```
    """), source=None)
    
    # Add verification data with metadata
    request.verification_data.append(
        PrimaryVerificationData(
            id=str(uuid.uuid4()),
            dtype=VerificationDType.argdown,
            code_snippet=dedent("""
            ```argdown {filename="map.ad"}
            [Claim]: This is a claim.
            ```
            """),
            data=None,
            metadata={"filename": "map.ad"}
        )
    )

    # Add another verification data with different metadata
    request.verification_data.append(
        PrimaryVerificationData(
            id=str(uuid.uuid4()),
            dtype=VerificationDType.argdown,
            code_snippet=dedent("""
            ```argdown {filename="reconstructions.ad"}
            [Claim]: This is a reconstructed claim.
            ```
            """),
            data=None,
            metadata={"filename": "reconstructions.ad"}
        )
    )
    
    # Test with a filter that only accepts argdown with map.ad filename
    filter_func = lambda vdata: vdata.metadata and vdata.metadata.get("filename") == "map.ad"  # noqa: E731
    handler = HasArgdownHandler(name="map_checker", filter=filter_func)
    result = handler.handle(copy.deepcopy(request))
    
    assert len(result.results) == 1
    assert result.results[0].is_valid
    
    # Test with a filter that only accepts argdown with different filename
    filter_func = lambda vdata: vdata.metadata and vdata.metadata.get("filename") == "non_existent.ad"  # noqa: E731
    handler = HasArgdownHandler(name="non_existent_checker", filter=filter_func)
    result = handler.handle(request)
    
    pprint(result.results)
    assert len(result.results) == 1
    assert not result.results[0].is_valid


def test_with_mixed_input():
    request = VerificationRequest(inputs=dedent("""
    ```xml
    <proposition id="1">Test proposition</proposition>
    ```
    
    ```argdown
    [Claim]: Test claim
    ```
    """), source=None)
    
    # Add verification data for both XML and Argdown
    request.verification_data.append(
        PrimaryVerificationData(
            id=str(uuid.uuid4()),
            dtype=VerificationDType.xml,
            code_snippet=dedent("""
            ```xml
            <proposition id="1">Test proposition</proposition>
            ```
            """),
            data=None
        )
    )
    
    request.verification_data.append(
        PrimaryVerificationData(
            id=str(uuid.uuid4()),
            dtype=VerificationDType.argdown,
            code_snippet=dedent("""
            ```argdown
            [Claim]: Test claim
            ```
            """),
            data=None
        )
    )
    
    # Test HasAnnotationsHandler
    annotations_handler = HasAnnotationsHandler(name="xml_checker")
    result = annotations_handler.handle(request)
    
    assert len(result.results) == 1
    assert result.results[0].is_valid
    
    # Test HasArgdownHandler
    argdown_handler = HasArgdownHandler(name="argdown_checker")
    result = argdown_handler.handle(result)  # Chain the handlers
    
    assert len(result.results) == 2
    assert result.results[1].is_valid


def test_with_empty_input():
    request = VerificationRequest(inputs="", source=None)
    
    annotations_handler = HasAnnotationsHandler(name="xml_checker")
    result = annotations_handler.handle(request)
    
    assert len(result.results) == 1
    assert not result.results[0].is_valid
    assert "No fenced code block starting with '```xml'" in result.results[0].message
    
    argdown_handler = HasArgdownHandler(name="argdown_checker")
    result = argdown_handler.handle(result)
    
    assert len(result.results) == 2
    assert not result.results[1].is_valid
    assert "No fenced code block starting with '```argdown'" in result.results[1].message
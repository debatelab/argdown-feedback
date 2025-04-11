from pprint import pprint
import pytest
from textwrap import dedent
from bs4 import BeautifulSoup
from pyargdown import parse_argdown, ArgdownMultiDiGraph



from argdown_feedback.verifiers.coherence.arganno_argmap_handler import (
    BaseArgannoArgmapCoherenceHandler,
    ArgannoArgmapElemCohereHandler,
    ArgannoArgmapDRelCohereHandler,
    ArgannoArgmapCoherenceHandler
)
from argdown_feedback.verifiers.verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
    VerificationResult
)
from argdown_feedback.verifiers.processing_handler import _MULTI_VALUED_ATTRIBUTES


def parse_fenced_argdown(argdown_text: str):
    argdown_text = argdown_text.strip("\n ")
    argdown_text = "\n".join(argdown_text.splitlines()[1:-1])
    return parse_argdown(argdown_text)


@pytest.fixture
def valid_argdown_text():
    return dedent("""
    ```argdown
    [A1]: First claim. {annotation_ids: ["prop1", "prop2"]}
    [A2]: Second claim. {annotation_ids: ["prop3"]}
    
    [A1]
        -> [A2]
    ```
    """)


@pytest.fixture
def valid_argdown_graph(valid_argdown_text):
    return parse_fenced_argdown(valid_argdown_text)


@pytest.fixture
def valid_xml_text():
    return """
    <proposition id="prop1" argument_label="A1">First claim part 1.</proposition>
    <proposition id="prop2" argument_label="A1">First claim part 2.</proposition>
    <proposition id="prop3" argument_label="A2" supports="prop1">Second claim.</proposition>
    """


@pytest.fixture
def valid_xml_soup(valid_xml_text):
    return BeautifulSoup(valid_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def invalid_label_xml_text():
    return """
    <proposition id="prop1" argument_label="A1">First claim part 1.</proposition>
    <proposition id="prop2" argument_label="A1">First claim part 2.</proposition>
    <proposition id="prop3" argument_label="NonExistent">Invalid label reference.</proposition>
    """


@pytest.fixture
def invalid_label_xml_soup(invalid_label_xml_text):
    return BeautifulSoup(invalid_label_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def missing_annotation_ids_argdown_text():
    return dedent("""
    ```argdown
    [A1]: First claim. {annotation_ids: ["prop1", "prop2"]}
    [A2]: Second claim.
    
    [A1]
        -> [A2]
    ```
    """)


@pytest.fixture
def missing_annotation_ids_graph(missing_annotation_ids_argdown_text):
    return parse_fenced_argdown(missing_annotation_ids_argdown_text)


@pytest.fixture
def invalid_annotation_id_argdown_text():
    return dedent("""
    ```argdown
    [A1]: First claim. {annotation_ids: ["prop1", "prop2"]}
    [A2]: Second claim. {annotation_ids: ["nonexistent"]}
    
    [A1]
        -> [A2]
    ```
    """)


@pytest.fixture
def invalid_annotation_id_graph(invalid_annotation_id_argdown_text):
    return parse_fenced_argdown(invalid_annotation_id_argdown_text)


@pytest.fixture
def mismatched_label_argdown_text():
    return dedent("""
    ```argdown
    [A1]: First claim. {annotation_ids: ["prop3"]}   // prop3 has label A2 in annotation
    [A2]: Second claim. {annotation_ids: ["prop1"]}  // prop1 has label A1 in annotation
    
    [A1]
        -> [A2]
    ```
    """)


@pytest.fixture
def mismatched_label_graph(mismatched_label_argdown_text):
    return parse_fenced_argdown(mismatched_label_argdown_text)


@pytest.fixture
def valid_drel_argdown_text():
    return dedent("""
    ```argdown
    [A1]: First claim. {annotation_ids: ["prop1"]}
    [A2]: Second claim. {annotation_ids: ["prop2"]}
    [A3]: Third claim. {annotation_ids: ["prop3"]}
    
    [A1] 
        +> [A2]
    [A3]
        -> [A2]
    ```
    """)


@pytest.fixture
def valid_drel_argdown_graph(valid_drel_argdown_text):
    return parse_fenced_argdown(valid_drel_argdown_text)


@pytest.fixture
def valid_drel_xml_text():
    return """
    <proposition id="prop1" argument_label="A1" supports="prop2">First claim.</proposition>
    <proposition id="prop2" argument_label="A2">Second claim.</proposition>
    <proposition id="prop3" argument_label="A3" attacks="prop2">Third claim.</proposition>
    """


@pytest.fixture
def valid_drel_xml_soup(valid_drel_xml_text):
    return BeautifulSoup(valid_drel_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def inconsistent_drel_argdown_text():
    return dedent("""
    ```argdown
    [A1]: First claim. {annotation_ids: ["prop1"]}
    [A2]: Second claim. {annotation_ids: ["prop2"]}
    [A3]: Third claim. {annotation_ids: ["prop3"]}
    
    [A1] 
        -> [A2]
    [A1] 
        -> [A3]  // This relation is not in annotation
    ```
    """)


@pytest.fixture
def inconsistent_drel_argdown_graph(inconsistent_drel_argdown_text):
    return parse_fenced_argdown(inconsistent_drel_argdown_text)


@pytest.fixture
def inconsistent_drel_xml_text():
    return """
    <proposition id="prop1" argument_label="A1" supports="prop2">First claim.</proposition>
    <proposition id="prop2" argument_label="A2">Second claim.</proposition>
    <proposition id="prop3" argument_label="A3" supports="prop1">Third claim supports first.</proposition>
    """


@pytest.fixture
def inconsistent_drel_xml_soup(inconsistent_drel_xml_text):
    return BeautifulSoup(inconsistent_drel_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def valid_argdown_vdata(valid_argdown_graph):
    return PrimaryVerificationData(
        id="argdown_test", dtype=VerificationDType.argdown, data=valid_argdown_graph
    )


@pytest.fixture
def valid_xml_vdata(valid_xml_soup):
    return PrimaryVerificationData(
        id="xml_test", dtype=VerificationDType.xml, data=valid_xml_soup
    )


@pytest.fixture
def verification_request_with_valid_data(valid_argdown_vdata, valid_xml_vdata):
    return VerificationRequest(
        inputs="test", source="test source", verification_data=[valid_argdown_vdata, valid_xml_vdata]
    )


@pytest.fixture
def invalid_label_xml_vdata(invalid_label_xml_soup):
    return PrimaryVerificationData(
        id="invalid_label_xml", dtype=VerificationDType.xml, data=invalid_label_xml_soup
    )


@pytest.fixture
def missing_annotation_ids_vdata(missing_annotation_ids_graph):
    return PrimaryVerificationData(
        id="missing_anno_ids", dtype=VerificationDType.argdown, data=missing_annotation_ids_graph
    )


@pytest.fixture
def invalid_annotation_id_vdata(invalid_annotation_id_graph):
    return PrimaryVerificationData(
        id="invalid_anno_id", dtype=VerificationDType.argdown, data=invalid_annotation_id_graph
    )


@pytest.fixture
def mismatched_label_vdata(mismatched_label_graph):
    return PrimaryVerificationData(
        id="mismatched_label", dtype=VerificationDType.argdown, data=mismatched_label_graph
    )


@pytest.fixture
def valid_drel_argdown_vdata(valid_drel_argdown_graph):
    return PrimaryVerificationData(
        id="valid_drel_ad", dtype=VerificationDType.argdown, data=valid_drel_argdown_graph
    )


@pytest.fixture
def valid_drel_xml_vdata(valid_drel_xml_soup):
    return PrimaryVerificationData(
        id="valid_drel_xml", dtype=VerificationDType.xml, data=valid_drel_xml_soup
    )


@pytest.fixture
def verification_request_with_valid_drels(valid_drel_argdown_vdata, valid_drel_xml_vdata):
    return VerificationRequest(
        inputs="test", source="test source", 
        verification_data=[valid_drel_argdown_vdata, valid_drel_xml_vdata]
    )


@pytest.fixture
def inconsistent_drel_argdown_vdata(inconsistent_drel_argdown_graph):
    return PrimaryVerificationData(
        id="inconsistent_drel_ad", dtype=VerificationDType.argdown, data=inconsistent_drel_argdown_graph
    )


@pytest.fixture
def inconsistent_drel_xml_vdata(inconsistent_drel_xml_soup):
    return PrimaryVerificationData(
        id="inconsistent_drel_xml", dtype=VerificationDType.xml, data=inconsistent_drel_xml_soup
    )


def test_get_labels(valid_argdown_graph, valid_xml_soup):
    all_argmap_labels, all_annotation_ids, argument_label_map = BaseArgannoArgmapCoherenceHandler.get_labels(
        valid_argdown_graph, valid_xml_soup
    )
    
    assert set(all_argmap_labels) == {"A1", "A2"}
    assert set(all_annotation_ids) == {"prop1", "prop2", "prop3"}
    assert argument_label_map == {"prop1": "A1", "prop2": "A1", "prop3": "A2"}


def test_elem_cohere_handler_valid(verification_request_with_valid_data, valid_argdown_vdata, valid_xml_vdata):
    handler = ArgannoArgmapElemCohereHandler()
    result = handler.evaluate(valid_argdown_vdata, valid_xml_vdata, verification_request_with_valid_data)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None
    assert result.verification_data_references == ["argdown_test", "xml_test"]


def test_elem_cohere_handler_illegal_label(valid_argdown_vdata, invalid_label_xml_vdata):
    handler = ArgannoArgmapElemCohereHandler()
    result = handler.evaluate(valid_argdown_vdata, invalid_label_xml_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Illegal 'argument_label' reference" in result.message


def test_elem_cohere_handler_missing_annotation_ids(missing_annotation_ids_vdata, valid_xml_vdata):
    handler = ArgannoArgmapElemCohereHandler()
    result = handler.evaluate(missing_annotation_ids_vdata, valid_xml_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Missing 'annotation_ids' attribute" in result.message


def test_elem_cohere_handler_invalid_annotation_id(invalid_annotation_id_vdata, valid_xml_vdata):
    handler = ArgannoArgmapElemCohereHandler()
    result = handler.evaluate(invalid_annotation_id_vdata, valid_xml_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Illegal 'annotation_ids' reference" in result.message


def test_elem_cohere_handler_mismatched_label(mismatched_label_vdata, valid_xml_vdata):
    handler = ArgannoArgmapElemCohereHandler()
    result = handler.evaluate(mismatched_label_vdata, valid_xml_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Label reference mismatch" in result.message


def test_drel_cohere_handler_valid(verification_request_with_valid_drels, valid_drel_argdown_vdata, valid_drel_xml_vdata):
    handler = ArgannoArgmapDRelCohereHandler()
    result = handler.evaluate(valid_drel_argdown_vdata, valid_drel_xml_vdata, verification_request_with_valid_drels)

    pprint(valid_drel_argdown_vdata.data.dialectical_relations)
    pprint(valid_drel_xml_vdata.data.find_all("proposition"))
    pprint(result)

    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_drel_cohere_handler_inconsistent(inconsistent_drel_argdown_vdata, inconsistent_drel_xml_vdata):
    handler = ArgannoArgmapDRelCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        inconsistent_drel_argdown_vdata, inconsistent_drel_xml_vdata
    ])
    result = handler.evaluate(inconsistent_drel_argdown_vdata, inconsistent_drel_xml_vdata, request)
    
    assert result is not None
    assert result.is_valid is False
    assert "is not matched by any" in result.message


def test_composite_handler():
    composite = ArgannoArgmapCoherenceHandler()
    
    # Check that default handlers are initialized
    assert len(composite.handlers) == 2
    assert any(isinstance(h, ArgannoArgmapElemCohereHandler) for h in composite.handlers)
    assert any(isinstance(h, ArgannoArgmapDRelCohereHandler) for h in composite.handlers)


def test_composite_handler_with_custom_filters():
    custom_filter1 = lambda vd: vd.dtype == VerificationDType.argdown and "test" in vd.id  # noqa: E731
    custom_filter2 = lambda vd: vd.dtype == VerificationDType.xml and "test" in vd.id  # noqa: E731
    
    composite = ArgannoArgmapCoherenceHandler(filters=(custom_filter1, custom_filter2))
    
    # Check that filters were passed to child handlers
    for handler in composite.handlers:
        assert handler.filters == (custom_filter1, custom_filter2)


def test_composite_handler_process_request(verification_request_with_valid_data, valid_argdown_vdata, valid_xml_vdata):
    composite = ArgannoArgmapCoherenceHandler()
    
    # Mock evaluation results for child handlers
    class MockHandler(BaseArgannoArgmapCoherenceHandler):
        def __init__(self, name, result_value):
            super().__init__(name)
            self.result_value = result_value
            
        def evaluate(self, vdata1, vdata2, ctx):
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata1.id, vdata2.id],
                is_valid=self.result_value,
                message=None if self.result_value else "Mock error"
            )
    
    composite.handlers = [
        MockHandler("TestHandler1", True),
        MockHandler("TestHandler2", False)
    ]
    
    result_request = composite.process(verification_request_with_valid_data)
    
    # Should have results for both handlers
    assert len(result_request.results) == 2
    assert any(r.verifier_id == "TestHandler1" and r.is_valid for r in result_request.results)
    assert any(r.verifier_id == "TestHandler2" and not r.is_valid for r in result_request.results)


def test_handle_wrong_data_types():
    handler = ArgannoArgmapElemCohereHandler()
    
    # Test with wrong data type for argdown
    with pytest.raises(AssertionError):
        wrong_argdown_vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data="not a graph")
        handler.evaluate(
            wrong_argdown_vdata,
            PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=BeautifulSoup("<a></a>", "html.parser")),
            VerificationRequest(inputs="test")
        )
    
    # Test with wrong data type for XML
    with pytest.raises(AssertionError):
        handler.evaluate(
            PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=ArgdownMultiDiGraph()),
            PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data="not a soup"),
            VerificationRequest(inputs="test")
        )


def test_real_world_example():
    # A more complex real-world example with multiple nodes and relations
    argdown_text = dedent("""
    ```argdown
    [Claim1]: The earth is round. {annotation_ids: ["p1"]}
    [Claim2]: We can observe ships disappearing hull-first over the horizon. {annotation_ids: ["p2"]}
    [Claim3]: The shadow cast on the moon during lunar eclipses is round. {annotation_ids: ["p3"]}
    [Objection]: Ancient people believed the earth was flat. {annotation_ids: ["p4"]}
    
    [Claim2]
        -> [Claim1]
    [Claim3]
        +> [Claim1]
    [Objection]
        -> [Claim1]
    ```
    """)
    
    xml_text = """
    <proposition id="p1" argument_label="Claim1">The earth is round.</proposition>
    <proposition id="p2" argument_label="Claim2" attacks="p1">We can observe ships disappearing hull-first over the horizon.</proposition>
    <proposition id="p3" argument_label="Claim3" supports="p1">The shadow cast on the moon during lunar eclipses is round.</proposition>
    <proposition id="p4" argument_label="Objection" attacks="p1">Ancient people believed the earth was flat.</proposition>
    """
    
    argdown_graph = parse_fenced_argdown(argdown_text)
    xml_soup = BeautifulSoup(xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)
    
    argdown_vdata = PrimaryVerificationData(
        id="real_argdown", dtype=VerificationDType.argdown, data=argdown_graph
    )
    xml_vdata = PrimaryVerificationData(
        id="real_xml", dtype=VerificationDType.xml, data=xml_soup
    )
    
    request = VerificationRequest(
        inputs="test", source="real world example", 
        verification_data=[argdown_vdata, xml_vdata]
    )
    
    # Test with composite handler
    composite = ArgannoArgmapCoherenceHandler()
    result_request = composite.process(request)
    
    # Should have results for both handlers
    assert len(result_request.results) == 2
    
    # Both should be valid for this well-formed example
    for result in result_request.results:
        assert result.is_valid is True, f"Handler {result.verifier_id} failed with: {result.message}"
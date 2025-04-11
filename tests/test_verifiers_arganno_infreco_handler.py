from pprint import pprint  # noqa: F401
import pytest
from textwrap import dedent
from bs4 import BeautifulSoup
from pyargdown import parse_argdown, ArgdownMultiDiGraph

from argdown_feedback.verifiers.coherence.arganno_infreco_handler import (
    BaseArgannoInfrecoCoherenceHandler,
    ArgannoInfrecoElemCohereHandler,
    ArgannoInfrecoRelationCohereHandler,
    ArgannoInfrecoCoherenceHandler
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
def valid_infreco_text():
    return dedent("""
    ```argdown
    <Argument 1>: Animals suffer.

    (1) Animals suffer. {annotation_ids: ["prop1"]}
    (2) Suffering is bad. {annotation_ids: ["prop2"]}
    -- {from: ["1", "2"]} --
    (3) We should minimize animal suffering. {annotation_ids: ["prop3"]}
    ```
    """)


@pytest.fixture
def valid_infreco_graph(valid_infreco_text):
    return parse_fenced_argdown(valid_infreco_text)


@pytest.fixture
def valid_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="1">Animals suffer.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="2">Suffering is bad.</proposition>
    <proposition id="prop3" argument_label="Argument 1" ref_reco_label="3" supports="prop1">We should minimize animal suffering.</proposition>
    """


@pytest.fixture
def valid_xml_soup(valid_xml_text):
    return BeautifulSoup(valid_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def invalid_argument_label_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="1">Animals suffer.</proposition>
    <proposition id="prop2" argument_label="NonExistent" ref_reco_label="2">Suffering is bad.</proposition>
    <proposition id="prop3" argument_label="Argument 1" ref_reco_label="3">We should minimize animal suffering.</proposition>
    """


@pytest.fixture
def invalid_argument_label_xml_soup(invalid_argument_label_xml_text):
    return BeautifulSoup(invalid_argument_label_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def invalid_ref_reco_label_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="1">Animals suffer.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="NonExistent">Suffering is bad.</proposition>
    <proposition id="prop3" argument_label="Argument 1" ref_reco_label="3">We should minimize animal suffering.</proposition>
    """


@pytest.fixture
def invalid_ref_reco_label_xml_soup(invalid_ref_reco_label_xml_text):
    return BeautifulSoup(invalid_ref_reco_label_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def missing_annotation_ids_infreco_text():
    return dedent("""
    ```argdown
    <Argument 1>: Animals suffer.

    (1) Animals suffer. {annotation_ids: ["prop1"]}
    (2) Suffering is bad.
    -- {from: ["1", "2"]} --
    (3) We should minimize animal suffering. {annotation_ids: ["prop3"]}
    ```
    """)


@pytest.fixture
def missing_annotation_ids_graph(missing_annotation_ids_infreco_text):
    return parse_fenced_argdown(missing_annotation_ids_infreco_text)


@pytest.fixture
def invalid_annotation_id_infreco_text():
    return dedent("""
    ```argdown
    <Argument 1>: Animals suffer.

    (1) Animals suffer. {annotation_ids: ["prop1"]}
    (2) Suffering is bad. {annotation_ids: ["prop2"]}
    -- {from: ["1", "2"]} --
    (3) We should minimize animal suffering. {annotation_ids: ["nonexistent"]}
    ```
    """)


@pytest.fixture
def invalid_annotation_id_graph(invalid_annotation_id_infreco_text):
    return parse_fenced_argdown(invalid_annotation_id_infreco_text)


@pytest.fixture
def overlapping_annotation_ids_text():
    return dedent("""
    ```argdown
    <Argument 1>: Animals suffer.

    (1) Animals suffer. {annotation_ids: ["prop1", "prop2"]}
    (2) Suffering is bad. {annotation_ids: ["prop2"]}
    -- {from: ["1", "2"]} --
    (3) We should minimize animal suffering. {annotation_ids: ["prop3"]}
    ```
    """)


@pytest.fixture
def overlapping_annotation_ids_graph(overlapping_annotation_ids_text):
    return parse_fenced_argdown(overlapping_annotation_ids_text)


@pytest.fixture
def valid_relation_infreco_text():
    return dedent("""
    ```argdown
    <Argument 1>: First argument.

    (1) Animals suffer. {annotation_ids: ["prop1"]}
    -- {from: ["1"]} --
    (2) We should minimize animal suffering. {annotation_ids: ["prop2"]}

    <Argument 2>: Second argument.

    (1) Factory farming increases animal suffering. {annotation_ids: ["prop3"]}
    -- {from: ["1"]} --
    (2) Factory farming is wrong. {annotation_ids: ["prop4"]}
    ```
    """)


@pytest.fixture
def valid_relation_infreco_graph(valid_relation_infreco_text):
    return parse_fenced_argdown(valid_relation_infreco_text)


@pytest.fixture
def valid_relation_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="1" supports="prop2">Animals suffer.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="2">We should minimize animal suffering.</proposition>
    <proposition id="prop3" argument_label="Argument 2" ref_reco_label="1" supports="prop4">Factory farming increases animal suffering.</proposition>
    <proposition id="prop4" argument_label="Argument 2" ref_reco_label="2">Factory farming is wrong.</proposition>
    """


@pytest.fixture
def valid_relation_xml_soup(valid_relation_xml_text):
    return BeautifulSoup(valid_relation_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def inconsistent_relation_text():
    return dedent("""
    ```argdown
    <Argument 1>: First argument.

    (1) Animals suffer. {annotation_ids: ["prop1"]}
    -- {from: ["1"]} --
    (2) We should minimize animal suffering. {annotation_ids: ["prop2"]}

    <Argument 2>: Second argument.

    (1) Factory farming increases animal suffering. {annotation_ids: ["prop3"]}
    -- {from: ["1"]} --
    (2) Factory farming is wrong. {annotation_ids: ["prop4"]}
    ```
    """)


@pytest.fixture
def inconsistent_relation_graph(inconsistent_relation_text):
    return parse_fenced_argdown(inconsistent_relation_text)


@pytest.fixture
def inconsistent_relation_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="1">Animals suffer.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="2">We should minimize animal suffering.</proposition>
    <proposition id="prop3" argument_label="Argument 2" ref_reco_label="1">Factory farming increases animal suffering.</proposition>
    <proposition id="prop4" argument_label="Argument 2" ref_reco_label="2" supports="prop1">Factory farming is wrong supports animals suffer.</proposition>
    """


@pytest.fixture
def inconsistent_relation_xml_soup(inconsistent_relation_xml_text):
    return BeautifulSoup(inconsistent_relation_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def attack_between_same_argument_text():
    return dedent("""
    ```argdown
    <Argument 1>: First argument.

    (1) Animals suffer. {annotation_ids: ["prop1"]}
    -- {from: ["1"]} --
    (2) We should minimize animal suffering. {annotation_ids: ["prop2"]}
    ```
    """)


@pytest.fixture
def attack_between_same_argument_graph(attack_between_same_argument_text):
    return parse_fenced_argdown(attack_between_same_argument_text)


@pytest.fixture
def attack_between_same_argument_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="1">Animals suffer.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="2" attacks="prop1">We should minimize animal suffering attacks animals suffer.</proposition>
    """


@pytest.fixture
def attack_between_same_argument_xml_soup(attack_between_same_argument_xml_text):
    return BeautifulSoup(attack_between_same_argument_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def valid_infreco_vdata(valid_infreco_graph):
    return PrimaryVerificationData(
        id="infreco_test", dtype=VerificationDType.argdown, data=valid_infreco_graph
    )


@pytest.fixture
def valid_xml_vdata(valid_xml_soup):
    return PrimaryVerificationData(
        id="xml_test", dtype=VerificationDType.xml, data=valid_xml_soup
    )


@pytest.fixture
def verification_request_with_valid_data(valid_infreco_vdata, valid_xml_vdata):
    return VerificationRequest(
        inputs="test", source="test source", verification_data=[valid_infreco_vdata, valid_xml_vdata]
    )


@pytest.fixture
def invalid_argument_label_xml_vdata(invalid_argument_label_xml_soup):
    return PrimaryVerificationData(
        id="invalid_arg_label_xml", dtype=VerificationDType.xml, data=invalid_argument_label_xml_soup
    )


@pytest.fixture
def invalid_ref_reco_label_xml_vdata(invalid_ref_reco_label_xml_soup):
    return PrimaryVerificationData(
        id="invalid_ref_reco_xml", dtype=VerificationDType.xml, data=invalid_ref_reco_label_xml_soup
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
def overlapping_annotation_ids_vdata(overlapping_annotation_ids_graph):
    return PrimaryVerificationData(
        id="overlapping_anno_ids", dtype=VerificationDType.argdown, data=overlapping_annotation_ids_graph
    )


@pytest.fixture
def valid_relation_infreco_vdata(valid_relation_infreco_graph):
    return PrimaryVerificationData(
        id="valid_relation_infreco", dtype=VerificationDType.argdown, data=valid_relation_infreco_graph
    )


@pytest.fixture
def valid_relation_xml_vdata(valid_relation_xml_soup):
    return PrimaryVerificationData(
        id="valid_relation_xml", dtype=VerificationDType.xml, data=valid_relation_xml_soup
    )


@pytest.fixture
def verification_request_with_valid_relations(valid_relation_infreco_vdata, valid_relation_xml_vdata):
    return VerificationRequest(
        inputs="test", source="test source", 
        verification_data=[valid_relation_infreco_vdata, valid_relation_xml_vdata]
    )


@pytest.fixture
def inconsistent_relation_infreco_vdata(inconsistent_relation_graph):
    return PrimaryVerificationData(
        id="inconsistent_relation_infreco", dtype=VerificationDType.argdown, data=inconsistent_relation_graph
    )


@pytest.fixture
def inconsistent_relation_xml_vdata(inconsistent_relation_xml_soup):
    return PrimaryVerificationData(
        id="inconsistent_relation_xml", dtype=VerificationDType.xml, data=inconsistent_relation_xml_soup
    )


@pytest.fixture
def attack_between_same_argument_infreco_vdata(attack_between_same_argument_graph):
    return PrimaryVerificationData(
        id="attack_same_arg_infreco", dtype=VerificationDType.argdown, data=attack_between_same_argument_graph
    )


@pytest.fixture
def attack_between_same_argument_xml_vdata(attack_between_same_argument_xml_soup):
    return PrimaryVerificationData(
        id="attack_same_arg_xml", dtype=VerificationDType.xml, data=attack_between_same_argument_xml_soup
    )


def test_get_labels(valid_infreco_graph, valid_xml_soup):
    all_argument_labels, all_annotation_ids, argument_label_map, refreco_map, proposition_label_map = (
        BaseArgannoInfrecoCoherenceHandler.get_labels(valid_infreco_graph, valid_xml_soup)
    )
    
    assert set(all_argument_labels) == {"Argument 1"}
    assert set(all_annotation_ids) == {"prop1", "prop2", "prop3"}
    assert argument_label_map == {"prop1": "Argument 1", "prop2": "Argument 1", "prop3": "Argument 1"}
    assert refreco_map == {"prop1": "1", "prop2": "2", "prop3": "3"}
    # Note: proposition_label_map would be empty because we don't actually have proposition labels in this example


def test_elem_cohere_handler_valid(verification_request_with_valid_data, valid_infreco_vdata, valid_xml_vdata):
    handler = ArgannoInfrecoElemCohereHandler()
    result = handler.evaluate(valid_infreco_vdata, valid_xml_vdata, verification_request_with_valid_data)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None
    assert result.verification_data_references == ["infreco_test", "xml_test"]


def test_elem_cohere_handler_illegal_argument_label(valid_infreco_vdata, invalid_argument_label_xml_vdata):
    handler = ArgannoInfrecoElemCohereHandler()
    result = handler.evaluate(valid_infreco_vdata, invalid_argument_label_xml_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Illegal 'argument_label' reference" in result.message


def test_elem_cohere_handler_illegal_ref_reco_label(valid_infreco_vdata, invalid_ref_reco_label_xml_vdata):
    handler = ArgannoInfrecoElemCohereHandler()
    result = handler.evaluate(valid_infreco_vdata, invalid_ref_reco_label_xml_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Illegal 'ref_reco_label' reference" in result.message


def test_elem_cohere_handler_missing_annotation_ids(missing_annotation_ids_vdata, valid_xml_vdata):
    handler = ArgannoInfrecoElemCohereHandler()
    result = handler.evaluate(missing_annotation_ids_vdata, valid_xml_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Missing 'annotation_ids'" in result.message


def test_elem_cohere_handler_invalid_annotation_id(invalid_annotation_id_vdata, valid_xml_vdata):
    handler = ArgannoInfrecoElemCohereHandler()
    result = handler.evaluate(invalid_annotation_id_vdata, valid_xml_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Illegal 'annotation_ids' reference" in result.message


def test_elem_cohere_handler_overlapping_annotation_ids(overlapping_annotation_ids_vdata, valid_xml_vdata):
    handler = ArgannoInfrecoElemCohereHandler()
    result = handler.evaluate(overlapping_annotation_ids_vdata, valid_xml_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Label reference mismatch" in result.message


def test_relation_cohere_handler_valid(verification_request_with_valid_relations, 
                                      valid_relation_infreco_vdata, 
                                      valid_relation_xml_vdata):
    handler = ArgannoInfrecoRelationCohereHandler()
    result = handler.evaluate(valid_relation_infreco_vdata, 
                            valid_relation_xml_vdata, 
                            verification_request_with_valid_relations)
    pprint(result)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_relation_cohere_handler_inconsistent(inconsistent_relation_infreco_vdata, 
                                            inconsistent_relation_xml_vdata):
    handler = ArgannoInfrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        inconsistent_relation_infreco_vdata, inconsistent_relation_xml_vdata
    ])
    result = handler.evaluate(inconsistent_relation_infreco_vdata, 
                            inconsistent_relation_xml_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is False
    assert "supports" in result.message


def test_relation_cohere_handler_attack_same_argument(attack_between_same_argument_infreco_vdata,
                                                    attack_between_same_argument_xml_vdata):
    handler = ArgannoInfrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        attack_between_same_argument_infreco_vdata, attack_between_same_argument_xml_vdata
    ])
    result = handler.evaluate(attack_between_same_argument_infreco_vdata, 
                            attack_between_same_argument_xml_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is False
    assert "Text segments assigned to the same argument cannot attack each other" in result.message


def test_composite_handler():
    composite = ArgannoInfrecoCoherenceHandler()
    
    # Check that default handlers are initialized
    assert len(composite.handlers) == 2
    assert any(isinstance(h, ArgannoInfrecoElemCohereHandler) for h in composite.handlers)
    assert any(isinstance(h, ArgannoInfrecoRelationCohereHandler) for h in composite.handlers)


def test_composite_handler_with_custom_filters():
    custom_filter1 = lambda vd: vd.dtype == VerificationDType.argdown and "test" in vd.id  # noqa: E731
    custom_filter2 = lambda vd: vd.dtype == VerificationDType.xml and "test" in vd.id  # noqa: E731
    custom_from_key = "premises"
    
    composite = ArgannoInfrecoCoherenceHandler(filters=(custom_filter1, custom_filter2), from_key=custom_from_key)
    
    # Check that filters were passed to child handlers
    for handler in composite.handlers:
        assert handler.filters == (custom_filter1, custom_filter2)
        assert handler.from_key == custom_from_key


def test_composite_handler_process_request(verification_request_with_valid_data, 
                                          valid_infreco_vdata, 
                                          valid_xml_vdata):
    composite = ArgannoInfrecoCoherenceHandler()
    
    # Mock evaluation results for child handlers
    class MockHandler(BaseArgannoInfrecoCoherenceHandler):
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
    handler = ArgannoInfrecoElemCohereHandler()
    
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
    # A more complex real-world example with multiple arguments and relations
    argdown_text = dedent("""
    ```argdown
    <MoralArg>: We should not eat animals.

    (P1) Causing unnecessary suffering is wrong. {annotation_ids: ["p1"]}
    (P2) Eating animals causes unnecessary suffering. {annotation_ids: ["p2"]}
    -- {from: ["P1", "P2"]} --
    (C1) Therefore, we should not eat animals. {annotation_ids: ["p3"]}

    <EnvArg>: Factory farming harms the environment.

    (P1) Factory farming produces significant greenhouse gas emissions. {annotation_ids: ["p4"]}
    (P2) Activities that significantly contribute to climate change are harmful. {annotation_ids: ["p5"]}
    -- {from: ["P1", "P2"]} --
    (C1) Therefore, factory farming harms the environment. {annotation_ids: ["p6"]}

    <MoralArg>
        +> <EnvArg>
    ```
    """)
    
    xml_text = """
    <proposition id="p1" argument_label="MoralArg" ref_reco_label="P1">Causing unnecessary suffering is wrong.</proposition>
    <proposition id="p2" argument_label="MoralArg" ref_reco_label="P2">Eating animals causes unnecessary suffering.</proposition>
    <proposition id="p3" argument_label="MoralArg" ref_reco_label="C" supports="p6">Therefore, we should not eat animals.</proposition>
    <proposition id="p4" argument_label="EnvArg" ref_reco_label="P1">Factory farming produces significant greenhouse gas emissions.</proposition>
    <proposition id="p5" argument_label="EnvArg" ref_reco_label="P2">Activities that significantly contribute to climate change are harmful.</proposition>
    <proposition id="p6" argument_label="EnvArg" ref_reco_label="C">Therefore, factory farming harms the environment.</proposition>
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
    composite = ArgannoInfrecoCoherenceHandler()
    result_request = composite.process(request)
    
    # Should have results for both handlers
    assert len(result_request.results) == 2
    
    # Both should be valid for this well-formed example
    for result in result_request.results:
        assert result.is_valid is True, f"Handler {result.verifier_id} failed with: {result.message}"
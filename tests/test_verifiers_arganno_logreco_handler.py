from pprint import pprint  # noqa: F401
import pytest
from textwrap import dedent
from bs4 import BeautifulSoup
from pyargdown import parse_argdown, ArgdownMultiDiGraph

from argdown_feedback.verifiers.coherence.arganno_logreco_handler import (
    ArgannoLogrecoCoherenceHandler
)
from argdown_feedback.verifiers.coherence.arganno_infreco_handler import (
    ArgannoInfrecoElemCohereHandler,
    ArgannoInfrecoRelationCohereHandler,
    BaseArgannoInfrecoCoherenceHandler
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
def valid_logreco_text():
    return dedent("""
    ```argdown
    <Argument 1>: Socrates is mortal.

    (P1) All men are mortal. {formalization: "all x.(M(x) -> D(x))", declarations: {"M": "Man", "D": "Mortal"}, annotation_ids: ["prop1"]}
    (P2) Socrates is a man. {formalization: "M(s)", declarations: {"s": "socrates"}, annotation_ids: ["prop2"]}
    -- {from: ["P1", "P2"]} --
    (C1) Socrates is mortal. {formalization: "D(s)", annotation_ids: ["prop3"]}
    ```
    """)


@pytest.fixture
def valid_logreco_graph(valid_logreco_text):
    return parse_fenced_argdown(valid_logreco_text)


@pytest.fixture
def valid_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="P1">All men are mortal.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="P2">Socrates is a man.</proposition>
    <proposition id="prop3" argument_label="Argument 1" ref_reco_label="C1" supports="prop1">Socrates is mortal.</proposition>
    """


@pytest.fixture
def valid_xml_soup(valid_xml_text):
    return BeautifulSoup(valid_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def invalid_argument_label_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="P1">All men are mortal.</proposition>
    <proposition id="prop2" argument_label="NonExistent" ref_reco_label="P2">Socrates is a man.</proposition>
    <proposition id="prop3" argument_label="Argument 1" ref_reco_label="C1">Socrates is mortal.</proposition>
    """


@pytest.fixture
def invalid_argument_label_xml_soup(invalid_argument_label_xml_text):
    return BeautifulSoup(invalid_argument_label_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def invalid_ref_reco_label_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="P1">All men are mortal.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="NonExistent">Socrates is a man.</proposition>
    <proposition id="prop3" argument_label="Argument 1" ref_reco_label="C1">Socrates is mortal.</proposition>
    """


@pytest.fixture
def invalid_ref_reco_label_xml_soup(invalid_ref_reco_label_xml_text):
    return BeautifulSoup(invalid_ref_reco_label_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def missing_annotation_ids_logreco_text():
    return dedent("""
    ```argdown
    <Argument 1>: Socrates is mortal.

    (P1) All men are mortal. {formalization: "all x.(M(x) -> D(x))", declarations: {"M": "Man", "D": "Mortal"}, annotation_ids: ["prop1"]}
    (P2) Socrates is a man. {formalization: "M(s)", declarations: {"s": "socrates"}}
    -- {from: ["P1", "P2"]} --
    (C1) Socrates is mortal. {formalization: "D(s)", annotation_ids: ["prop3"]}
    ```
    """)


@pytest.fixture
def missing_annotation_ids_graph(missing_annotation_ids_logreco_text):
    return parse_fenced_argdown(missing_annotation_ids_logreco_text)


@pytest.fixture
def invalid_annotation_id_logreco_text():
    return dedent("""
    ```argdown
    <Argument 1>: Socrates is mortal.

    (P1) All men are mortal. {formalization: "all x.(M(x) -> D(x))", declarations: {"M": "Man", "D": "Mortal"}, annotation_ids: ["prop1"]}
    (P2) Socrates is a man. {formalization: "M(s)", declarations: {"s": "socrates"}, annotation_ids: ["prop2"]}
    -- {from: ["P1", "P2"]} --
    (C1) Socrates is mortal. {formalization: "D(s)", annotation_ids: ["nonexistent"]}
    ```
    """)


@pytest.fixture
def invalid_annotation_id_graph(invalid_annotation_id_logreco_text):
    return parse_fenced_argdown(invalid_annotation_id_logreco_text)


@pytest.fixture
def valid_relation_logreco_text():
    return dedent("""
    ```argdown
    <Argument 1>: First argument.

    (P1) All men are mortal. {formalization: "all x.(M(x) -> D(x))", declarations: {"M": "Man", "D": "Mortal"}, annotation_ids: ["prop1"]}
    -- {from: ["P1"]} --
    (C1) Socrates is mortal. {formalization: "D(s)", declarations: {"s": "socrates"}, annotation_ids: ["prop2"]}

    <Argument 2>: Second argument.

    (P1) All animals feel pain. {formalization: "all x.(A(x) -> F(x))", declarations: {"A": "Animal", "F": "FeelsPain"}, annotation_ids: ["prop3"]}
    -- {from: ["P1"]} --
    (C1) We should not harm animals. {formalization: "all x.(A(x) -> -H(x))", declarations: {"H": "ShouldHarm"}, annotation_ids: ["prop4"]}
    ```
    """)


@pytest.fixture
def valid_relation_logreco_graph(valid_relation_logreco_text):
    return parse_fenced_argdown(valid_relation_logreco_text)


@pytest.fixture
def valid_relation_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="P1" supports="prop2">All men are mortal.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="C1">Socrates is mortal.</proposition>
    <proposition id="prop3" argument_label="Argument 2" ref_reco_label="P1" supports="prop4">All animals feel pain.</proposition>
    <proposition id="prop4" argument_label="Argument 2" ref_reco_label="C1">We should not harm animals.</proposition>
    """


@pytest.fixture
def valid_relation_xml_soup(valid_relation_xml_text):
    return BeautifulSoup(valid_relation_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def inconsistent_relation_text():
    return dedent("""
    ```argdown
    <Argument 1>: First argument.

    (P1) All men are mortal. {formalization: "all x.(M(x) -> D(x))", declarations: {"M": "Man", "D": "Mortal"}, annotation_ids: ["prop1"]}
    -- {from: ["P1"]} --
    (C1) Socrates is mortal. {formalization: "D(s)", declarations: {"s": "socrates"}, annotation_ids: ["prop2"]}

    <Argument 2>: Second argument.

    (P1) All animals feel pain. {formalization: "all x.(A(x) -> F(x))", declarations: {"A": "Animal", "F": "FeelsPain"}, annotation_ids: ["prop3"]}
    -- {from: ["P1"]} --
    (C1) We should not harm animals. {formalization: "all x.(A(x) -> -H(x))", declarations: {"H": "ShouldHarm"}, annotation_ids: ["prop4"]}
    ```
    """)


@pytest.fixture
def inconsistent_relation_graph(inconsistent_relation_text):
    return parse_fenced_argdown(inconsistent_relation_text)


@pytest.fixture
def inconsistent_relation_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="P1">All men are mortal.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="C1">Socrates is mortal.</proposition>
    <proposition id="prop3" argument_label="Argument 2" ref_reco_label="P1">All animals feel pain.</proposition>
    <proposition id="prop4" argument_label="Argument 2" ref_reco_label="C1" supports="prop1">We should not harm animals supports All men are mortal.</proposition>
    """


@pytest.fixture
def inconsistent_relation_xml_soup(inconsistent_relation_xml_text):
    return BeautifulSoup(inconsistent_relation_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def attack_between_same_argument_text():
    return dedent("""
    ```argdown
    <Argument 1>: First logical argument.

    (P1) All men are mortal. {formalization: "all x.(M(x) -> D(x))", declarations: {"M": "Man", "D": "Mortal"}, annotation_ids: ["prop1"]}
    -- {from: ["P1"]} --
    (C1) Socrates is mortal. {formalization: "D(s)", declarations: {"s": "socrates"}, annotation_ids: ["prop2"]}
    ```
    """)


@pytest.fixture
def attack_between_same_argument_graph(attack_between_same_argument_text):
    return parse_fenced_argdown(attack_between_same_argument_text)


@pytest.fixture
def attack_between_same_argument_xml_text():
    return """
    <proposition id="prop1" argument_label="Argument 1" ref_reco_label="P1">All men are mortal.</proposition>
    <proposition id="prop2" argument_label="Argument 1" ref_reco_label="C1" attacks="prop1">Socrates is mortal attacks All men are mortal.</proposition>
    """


@pytest.fixture
def attack_between_same_argument_xml_soup(attack_between_same_argument_xml_text):
    return BeautifulSoup(attack_between_same_argument_xml_text, "html.parser", multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES)


@pytest.fixture
def valid_logreco_vdata(valid_logreco_graph):
    return PrimaryVerificationData(
        id="logreco_test", dtype=VerificationDType.argdown, data=valid_logreco_graph
    )


@pytest.fixture
def valid_xml_vdata(valid_xml_soup):
    return PrimaryVerificationData(
        id="xml_test", dtype=VerificationDType.xml, data=valid_xml_soup
    )


@pytest.fixture
def verification_request_with_valid_data(valid_logreco_vdata, valid_xml_vdata):
    return VerificationRequest(
        inputs="test", source="test source", verification_data=[valid_logreco_vdata, valid_xml_vdata]
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
def valid_relation_logreco_vdata(valid_relation_logreco_graph):
    return PrimaryVerificationData(
        id="valid_relation_logreco", dtype=VerificationDType.argdown, data=valid_relation_logreco_graph
    )


@pytest.fixture
def valid_relation_xml_vdata(valid_relation_xml_soup):
    return PrimaryVerificationData(
        id="valid_relation_xml", dtype=VerificationDType.xml, data=valid_relation_xml_soup
    )


@pytest.fixture
def verification_request_with_valid_relations(valid_relation_logreco_vdata, valid_relation_xml_vdata):
    return VerificationRequest(
        inputs="test", source="test source", 
        verification_data=[valid_relation_logreco_vdata, valid_relation_xml_vdata]
    )


@pytest.fixture
def inconsistent_relation_logreco_vdata(inconsistent_relation_graph):
    return PrimaryVerificationData(
        id="inconsistent_relation_logreco", dtype=VerificationDType.argdown, data=inconsistent_relation_graph
    )


@pytest.fixture
def inconsistent_relation_xml_vdata(inconsistent_relation_xml_soup):
    return PrimaryVerificationData(
        id="inconsistent_relation_xml", dtype=VerificationDType.xml, data=inconsistent_relation_xml_soup
    )


@pytest.fixture
def attack_between_same_argument_logreco_vdata(attack_between_same_argument_graph):
    return PrimaryVerificationData(
        id="attack_same_arg_logreco", dtype=VerificationDType.argdown, data=attack_between_same_argument_graph
    )


@pytest.fixture
def attack_between_same_argument_xml_vdata(attack_between_same_argument_xml_soup):
    return PrimaryVerificationData(
        id="attack_same_arg_xml", dtype=VerificationDType.xml, data=attack_between_same_argument_xml_soup
    )


def test_inheritance_and_init():
    """Test that ArgannoLogrecoCoherenceHandler inherits from ArgannoInfrecoCoherenceHandler."""
    handler = ArgannoLogrecoCoherenceHandler()
    
    # Check inheritance
    assert isinstance(handler, ArgannoLogrecoCoherenceHandler)
    
    # Check default handlers are initialized
    assert len(handler.handlers) == 2
    assert any(isinstance(h, ArgannoInfrecoElemCohereHandler) for h in handler.handlers)
    assert any(isinstance(h, ArgannoInfrecoRelationCohereHandler) for h in handler.handlers)


def test_custom_init_parameters():
    """Test that custom init parameters are passed correctly."""
    custom_filter1 = lambda vd: vd.dtype == VerificationDType.argdown and "test" in vd.id  # noqa: E731
    custom_filter2 = lambda vd: vd.dtype == VerificationDType.xml and "test" in vd.id  # noqa: E731
    custom_from_key = "premises"
    custom_name = "CustomLogrecoHandler"
    
    handler = ArgannoLogrecoCoherenceHandler(
        name=custom_name,
        filters=(custom_filter1, custom_filter2),
        from_key=custom_from_key
    )
    
    assert handler.name == custom_name
    
    # Check that filters were passed to child handlers
    for child_handler in handler.handlers:
        assert child_handler.filters == (custom_filter1, custom_filter2)
        assert child_handler.from_key == custom_from_key


def test_elem_cohere_handler_valid(verification_request_with_valid_data):
    """Test elem coherence with valid logreco data."""
    handler = ArgannoLogrecoCoherenceHandler()
    result_request = handler.handle(verification_request_with_valid_data)
    
    assert len(result_request.results) == 2
    
    # Get the result from the element coherence handler
    elem_result = next((r for r in result_request.results if "Elem" in r.verifier_id), None)
    assert elem_result is not None
    assert elem_result.is_valid is True
    assert elem_result.message is None


def test_elem_cohere_handler_illegal_argument_label(valid_logreco_vdata, invalid_argument_label_xml_vdata):
    """Test elem coherence with illegal argument label."""
    handler = ArgannoLogrecoCoherenceHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        valid_logreco_vdata, invalid_argument_label_xml_vdata
    ])
    
    result_request = handler.handle(request)
    
    # Get the result from the element coherence handler
    elem_result = next((r for r in result_request.results if "Elem" in r.verifier_id), None)
    assert elem_result is not None
    assert elem_result.is_valid is False
    assert "Illegal 'argument_label' reference" in elem_result.message


def test_elem_cohere_handler_illegal_ref_reco_label(valid_logreco_vdata, invalid_ref_reco_label_xml_vdata):
    """Test elem coherence with illegal ref_reco label."""
    handler = ArgannoLogrecoCoherenceHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        valid_logreco_vdata, invalid_ref_reco_label_xml_vdata
    ])
    
    result_request = handler.handle(request)
    
    # Get the result from the element coherence handler
    elem_result = next((r for r in result_request.results if "Elem" in r.verifier_id), None)
    assert elem_result is not None
    assert elem_result.is_valid is False
    assert "Illegal 'ref_reco_label' reference" in elem_result.message


def test_elem_cohere_handler_missing_annotation_ids(missing_annotation_ids_vdata, valid_xml_vdata):
    """Test elem coherence with missing annotation_ids attribute."""
    handler = ArgannoLogrecoCoherenceHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        missing_annotation_ids_vdata, valid_xml_vdata
    ])
    
    result_request = handler.handle(request)
    
    # Get the result from the element coherence handler
    elem_result = next((r for r in result_request.results if "Elem" in r.verifier_id), None)
    assert elem_result is not None
    assert elem_result.is_valid is False
    assert "Missing 'annotation_ids'" in elem_result.message


def test_elem_cohere_handler_invalid_annotation_id(invalid_annotation_id_vdata, valid_xml_vdata):
    """Test elem coherence with invalid annotation_id reference."""
    handler = ArgannoLogrecoCoherenceHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        invalid_annotation_id_vdata, valid_xml_vdata
    ])
    
    result_request = handler.handle(request)
    
    # Get the result from the element coherence handler
    elem_result = next((r for r in result_request.results if "Elem" in r.verifier_id), None)
    assert elem_result is not None
    assert elem_result.is_valid is False
    assert "Illegal 'annotation_ids' reference" in elem_result.message


def test_relation_cohere_handler_valid(verification_request_with_valid_relations):
    """Test relation coherence with valid relations."""
    handler = ArgannoLogrecoCoherenceHandler()
    result_request = handler.handle(verification_request_with_valid_relations)
    
    # Get the result from the relation coherence handler
    relation_result = next((r for r in result_request.results if "Relation" in r.verifier_id), None)
    assert relation_result is not None
    assert relation_result.is_valid is True
    assert relation_result.message is None


def test_relation_cohere_handler_inconsistent(inconsistent_relation_logreco_vdata, inconsistent_relation_xml_vdata):
    """Test relation coherence with inconsistent relations."""
    handler = ArgannoLogrecoCoherenceHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        inconsistent_relation_logreco_vdata, inconsistent_relation_xml_vdata
    ])
    
    result_request = handler.handle(request)
    
    # Get the result from the relation coherence handler
    relation_result = next((r for r in result_request.results if "Relation" in r.verifier_id), None)
    assert relation_result is not None
    assert relation_result.is_valid is False
    assert "supports" in relation_result.message


def test_relation_cohere_handler_attack_same_argument(attack_between_same_argument_logreco_vdata, attack_between_same_argument_xml_vdata):
    """Test relation coherence with attack between segments in the same argument."""
    handler = ArgannoLogrecoCoherenceHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        attack_between_same_argument_logreco_vdata, attack_between_same_argument_xml_vdata
    ])
    
    result_request = handler.handle(request)
    
    # Get the result from the relation coherence handler
    relation_result = next((r for r in result_request.results if "Relation" in r.verifier_id), None)
    assert relation_result is not None
    assert relation_result.is_valid is False
    assert "Text segments assigned to the same argument cannot attack each other" in relation_result.message


def test_handle_wrong_data_types():
    """Test handling of wrong data types."""
    handler = ArgannoLogrecoCoherenceHandler()
    
    # Create a request with wrong data types
    request = VerificationRequest(inputs="test", verification_data=[
        PrimaryVerificationData(id="wrong_ad", dtype=VerificationDType.argdown, data="not a graph"),
        PrimaryVerificationData(id="wrong_xml", dtype=VerificationDType.xml, data="not a soup")
    ])
    
    result = handler.handle(request)
    pprint(result)
    assert all("Internal error:" in vr.message for vr in result.results)


def test_real_world_example():
    """Test with a realistic complex example containing multiple arguments with logical formalizations."""
    argdown_text = dedent("""
    ```argdown
    <Argument 1>: Modus Ponens Example

    (P1) If P then Q. {formalization: "P -> Q", annotation_ids: ["p1"]}
    (P2) P. {formalization: "P", annotation_ids: ["p2"]}
    -- {from: ["P1", "P2"]} --
    (C1) Therefore, Q. {formalization: "Q", annotation_ids: ["p3"]}

    <Argument 2>: Modus Tollens Example

    (P1) If P then Q. {formalization: "P -> Q", annotation_ids: ["p4"]}
    (P2) Not Q. {formalization: "-Q", annotation_ids: ["p5"]}
    -- {from: ["P1", "P2"]} --
    (C1) Therefore, not P. {formalization: "-P", annotation_ids: ["p6"]}

    <Argument 1>
        +> <Argument 2>
    ```
    """)
    
    xml_text = """
    <proposition id="p1" argument_label="Argument 1" ref_reco_label="P1">If P then Q.</proposition>
    <proposition id="p2" argument_label="Argument 1" ref_reco_label="P2">P.</proposition>
    <proposition id="p3" argument_label="Argument 1" ref_reco_label="C1" supports="p6">Therefore, Q.</proposition>
    <proposition id="p4" argument_label="Argument 2" ref_reco_label="P1">If P then Q.</proposition>
    <proposition id="p5" argument_label="Argument 2" ref_reco_label="P2">Not Q.</proposition>
    <proposition id="p6" argument_label="Argument 2" ref_reco_label="C1">Therefore, not P.</proposition>
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
    
    # Test with the handler
    handler = ArgannoLogrecoCoherenceHandler()
    result_request = handler.handle(request)
    
    # Should have results for both handlers
    assert len(result_request.results) == 2
    
    # Both should be valid for this well-formed example
    for result in result_request.results:
        assert result.is_valid is True, f"Handler {result.verifier_id} failed with: {result.message}"


def test_custom_handlers():
    """Test with custom handlers."""
    
    # Create mock handlers
    class MockHandler(BaseArgannoInfrecoCoherenceHandler):
        def __init__(self, name, result_value):
            super().__init__(name)
            self.result_value = result_value
            self.called = False
            
        def evaluate(self, vdata1, vdata2, ctx):
            self.called = True
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata1.id, vdata2.id],
                is_valid=self.result_value,
                message=None if self.result_value else "Mock error"
            )
    
    handler1 = MockHandler("TestHandler1", True)
    handler2 = MockHandler("TestHandler2", False)
    
    # Create our handler with custom handlers
    logreco_handler = ArgannoLogrecoCoherenceHandler(handlers=[handler1, handler2])
    
    # Create a simple valid request
    request = VerificationRequest(
        inputs="test", source="test", 
        verification_data=[
            PrimaryVerificationData(id="ad", dtype=VerificationDType.argdown, data=ArgdownMultiDiGraph()),
            PrimaryVerificationData(id="xml", dtype=VerificationDType.xml, data=BeautifulSoup("<a></a>", "html.parser"))
        ]
    )
    
    # Process the request
    result_request = logreco_handler.handle(request)
    
    # Check that both handlers were called and their results added
    assert len(result_request.results) == 2
    assert any(r.verifier_id == "TestHandler1" and r.is_valid for r in result_request.results)
    assert any(r.verifier_id == "TestHandler2" and not r.is_valid for r in result_request.results)
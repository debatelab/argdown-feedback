from pprint import pprint  # noqa: F401
import pytest
from textwrap import dedent
from pyargdown import parse_argdown, ArgdownMultiDiGraph

from argdown_feedback.verifiers.coherence.argmap_infreco_handler import (
    BaseArgmapInfrecoCoherenceHandler,
    ArgmapInfrecoElemCohereHandler,
    ArgmapInfrecoRelationCohereHandler,
    ArgmapInfrecoCoherenceHandler
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
def valid_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}
    [A1]: First claim.
    [A2]: Second claim.
    [A3]: Third claim.
    
    <Argument1>
        + [A1]
        +> [A2] // Support relation
    <Argument2>
        + [A3]
        -> [A2] // Attack relation
    ```
    """)


@pytest.fixture
def valid_map_graph(valid_map_text):
    return parse_fenced_argdown(valid_map_text)


@pytest.fixture
def valid_reco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim.
    -- {from: ["P1"]} --
    (C1) [A2]: Second claim.
    
    <Argument2>: Second argument.
    
    (P1) [A3]: Third claim.
    (P2) Implicit premise.
    -- {from: ["P1", "P2"]} --
    (C1) NOT: Second claim.
        -> [A2] // Attack relation
    ```
    """)


@pytest.fixture
def valid_reco_graph(valid_reco_text):
    return parse_fenced_argdown(valid_reco_text)


@pytest.fixture
def missing_argument_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}
    [A1]: First claim.
    [A3]: Third claim.
    
    <Argument1>
        + [A1]
    ```
    """)


@pytest.fixture
def missing_argument_map_graph(missing_argument_map_text):
    return parse_fenced_argdown(missing_argument_map_text)


@pytest.fixture
def missing_argument_reco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim.
    -- {from: ["P1"]} --
    (C1) First conclusion.
    ```
    """)


@pytest.fixture
def missing_argument_reco_graph(missing_argument_reco_text):
    return parse_fenced_argdown(missing_argument_reco_text)


@pytest.fixture
def missing_claim_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}
    [A1]: First claim.
    [A3]: Third claim.
    // <Argument2> Second arg. -- Missing this
    // ...    

    <Argument1>
        + [A1]
        +> [A3]
                  
    

    ```
    """)


@pytest.fixture
def missing_claim_map_graph(missing_claim_map_text):
    return parse_fenced_argdown(missing_claim_map_text)


@pytest.fixture
def valid_relation_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}
    [A1]: First claim.
    [A2]: Second claim.
    
    <Argument1>
        + [A1]
        +> [A2]
    ```
    """)


@pytest.fixture
def valid_relation_map_graph(valid_relation_map_text):
    return parse_fenced_argdown(valid_relation_map_text)


@pytest.fixture
def valid_relation_reco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim.
    -- {from: ["P1"]} --
    (C1) [A2]: Second claim.
    ```
    """)


@pytest.fixture
def valid_relation_reco_graph(valid_relation_reco_text):
    return parse_fenced_argdown(valid_relation_reco_text)


@pytest.fixture
def invalid_support_relation_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}
    [A1]: First claim.
    [A2]: Second claim.
    
    <Argument1>
        + [A1]
        +> [A2]
    ```
    """)


@pytest.fixture
def invalid_support_relation_map_graph(invalid_support_relation_map_text):
    return parse_fenced_argdown(invalid_support_relation_map_text)


@pytest.fixture
def invalid_support_relation_reco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim.
    -- {from: ["P1"]} --
    (C1) Different conclusion.
        -> [A2] // Attack relation, unlike in map
    ```
    """)


@pytest.fixture
def invalid_support_relation_reco_graph(invalid_support_relation_reco_text):
    return parse_fenced_argdown(invalid_support_relation_reco_text)


@pytest.fixture
def invalid_attack_relation_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}
    [A1]: First claim.
    [A2]: Second claim.
    
    <Argument1>
        + [A1]
        -> [A2]
    ```
    """)


@pytest.fixture
def invalid_attack_relation_map_graph(invalid_attack_relation_map_text):
    return parse_fenced_argdown(invalid_attack_relation_map_text)


@pytest.fixture
def invalid_attack_relation_reco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim.
    -- {from: ["P1"]} --
    (C1) Some conclusion.
        +> [A2] // Support relation, unlike in map
    ```
    """)


@pytest.fixture
def invalid_attack_relation_reco_graph(invalid_attack_relation_reco_text):
    return parse_fenced_argdown(invalid_attack_relation_reco_text)


@pytest.fixture
def prop_arg_relation_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}
    [A1]: First claim.
    [P1]: Proposition claim.
    
    [P1]
        +> <Argument1>
        
    <Argument1>
        +> [A1]
    ```
    """)


@pytest.fixture
def prop_arg_relation_map_graph(prop_arg_relation_map_text):
    return parse_fenced_argdown(prop_arg_relation_map_text)


@pytest.fixture
def prop_arg_relation_reco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [P1]: Proposition claim.
    (P2) Some premise.
    -- {from: ["P1", "P2"]} --
    (C1) [A1]: First claim.
    ```
    """)


@pytest.fixture
def prop_arg_relation_reco_graph(prop_arg_relation_reco_text):
    return parse_fenced_argdown(prop_arg_relation_reco_text)


@pytest.fixture
def arg_prop_relation_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}
    [A1]: First claim.
    [P1]: Proposition claim.
    
    <Argument1>
        + [A1]
        +> [P1]
    ```
    """)


@pytest.fixture
def arg_prop_relation_map_graph(arg_prop_relation_map_text):
    return parse_fenced_argdown(arg_prop_relation_map_text)


@pytest.fixture
def arg_prop_relation_reco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) Some premise.
    -- {from: ["P1"]} --
    (C1) [P1]: Proposition claim.
    ```
    """)


@pytest.fixture
def arg_prop_relation_reco_graph(arg_prop_relation_reco_text):
    return parse_fenced_argdown(arg_prop_relation_reco_text)


@pytest.fixture
def valid_map_vdata(valid_map_graph):
    return PrimaryVerificationData(
        id="map_test", 
        dtype=VerificationDType.argdown, 
        data=valid_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def valid_reco_vdata(valid_reco_graph):
    return PrimaryVerificationData(
        id="reco_test", 
        dtype=VerificationDType.argdown, 
        data=valid_reco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def verification_request_with_valid_data(valid_map_vdata, valid_reco_vdata):
    return VerificationRequest(
        inputs="test", 
        source="test source", 
        verification_data=[valid_map_vdata, valid_reco_vdata]
    )


@pytest.fixture
def missing_argument_map_vdata(missing_argument_map_graph):
    return PrimaryVerificationData(
        id="missing_arg_map", 
        dtype=VerificationDType.argdown, 
        data=missing_argument_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def missing_argument_reco_vdata(missing_argument_reco_graph):
    return PrimaryVerificationData(
        id="missing_arg_reco", 
        dtype=VerificationDType.argdown, 
        data=missing_argument_reco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def missing_claim_map_vdata(missing_claim_map_graph):
    return PrimaryVerificationData(
        id="missing_claim_map", 
        dtype=VerificationDType.argdown, 
        data=missing_claim_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def valid_relation_map_vdata(valid_relation_map_graph):
    return PrimaryVerificationData(
        id="valid_relation_map", 
        dtype=VerificationDType.argdown, 
        data=valid_relation_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def valid_relation_reco_vdata(valid_relation_reco_graph):
    return PrimaryVerificationData(
        id="valid_relation_reco", 
        dtype=VerificationDType.argdown, 
        data=valid_relation_reco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def verification_request_with_valid_relations(valid_relation_map_vdata, valid_relation_reco_vdata):
    return VerificationRequest(
        inputs="test", 
        source="test source", 
        verification_data=[valid_relation_map_vdata, valid_relation_reco_vdata]
    )


@pytest.fixture
def invalid_support_relation_map_vdata(invalid_support_relation_map_graph):
    return PrimaryVerificationData(
        id="invalid_support_relation_map", 
        dtype=VerificationDType.argdown, 
        data=invalid_support_relation_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def invalid_support_relation_reco_vdata(invalid_support_relation_reco_graph):
    return PrimaryVerificationData(
        id="invalid_support_relation_reco", 
        dtype=VerificationDType.argdown, 
        data=invalid_support_relation_reco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def invalid_attack_relation_map_vdata(invalid_attack_relation_map_graph):
    return PrimaryVerificationData(
        id="invalid_attack_relation_map", 
        dtype=VerificationDType.argdown, 
        data=invalid_attack_relation_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def invalid_attack_relation_reco_vdata(invalid_attack_relation_reco_graph):
    return PrimaryVerificationData(
        id="invalid_attack_relation_reco", 
        dtype=VerificationDType.argdown, 
        data=invalid_attack_relation_reco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def prop_arg_relation_map_vdata(prop_arg_relation_map_graph):
    return PrimaryVerificationData(
        id="prop_arg_relation_map", 
        dtype=VerificationDType.argdown, 
        data=prop_arg_relation_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def prop_arg_relation_reco_vdata(prop_arg_relation_reco_graph):
    return PrimaryVerificationData(
        id="prop_arg_relation_reco", 
        dtype=VerificationDType.argdown, 
        data=prop_arg_relation_reco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def arg_prop_relation_map_vdata(arg_prop_relation_map_graph):
    return PrimaryVerificationData(
        id="arg_prop_relation_map", 
        dtype=VerificationDType.argdown, 
        data=arg_prop_relation_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def arg_prop_relation_reco_vdata(arg_prop_relation_reco_graph):
    return PrimaryVerificationData(
        id="arg_prop_relation_reco", 
        dtype=VerificationDType.argdown, 
        data=arg_prop_relation_reco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


def test_elem_cohere_handler_valid(verification_request_with_valid_data, valid_map_vdata, valid_reco_vdata):
    handler = ArgmapInfrecoElemCohereHandler()
    result = handler.evaluate(valid_map_vdata, valid_reco_vdata, verification_request_with_valid_data)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None
    assert result.verification_data_references == ["map_test", "reco_test"]


def test_elem_cohere_handler_missing_argument_in_reco(missing_argument_reco_vdata, valid_map_vdata):
    handler = ArgmapInfrecoElemCohereHandler()
    result = handler.evaluate(valid_map_vdata, missing_argument_reco_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "is not reconstructed" in result.message


def test_elem_cohere_handler_missing_argument_in_map(missing_argument_map_vdata, valid_reco_vdata):
    handler = ArgmapInfrecoElemCohereHandler()
    result = handler.evaluate(missing_argument_map_vdata, valid_reco_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "is not in the map" in result.message


def test_elem_cohere_handler_missing_claim(missing_claim_map_vdata, valid_reco_vdata):
    handler = ArgmapInfrecoElemCohereHandler()
    result = handler.evaluate(missing_claim_map_vdata, valid_reco_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "Reconstructed argument <Argument2> is not in the map" in result.message


def test_relation_cohere_handler_valid(verification_request_with_valid_relations, 
                                      valid_relation_map_vdata, 
                                      valid_relation_reco_vdata):
    handler = ArgmapInfrecoRelationCohereHandler()
    result = handler.evaluate(valid_relation_map_vdata, 
                            valid_relation_reco_vdata, 
                            verification_request_with_valid_relations)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_relation_cohere_handler_invalid_support(invalid_support_relation_map_vdata, 
                                               invalid_support_relation_reco_vdata):
    handler = ArgmapInfrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        invalid_support_relation_map_vdata, invalid_support_relation_reco_vdata
    ])
    result = handler.evaluate(invalid_support_relation_map_vdata, 
                            invalid_support_relation_reco_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is False
    assert "not grounded in the argument reconstruction" in result.message
    assert "support relation" in result.message


def test_relation_cohere_handler_invalid_attack(invalid_attack_relation_map_vdata, 
                                               invalid_attack_relation_reco_vdata):
    handler = ArgmapInfrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        invalid_attack_relation_map_vdata, invalid_attack_relation_reco_vdata
    ])
    result = handler.evaluate(invalid_attack_relation_map_vdata, 
                            invalid_attack_relation_reco_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is False
    assert "not grounded in the argument reconstruction" in result.message
    assert "attack relation" in result.message


def test_relation_cohere_handler_prop_arg_relation(prop_arg_relation_map_vdata, 
                                                 prop_arg_relation_reco_vdata):
    handler = ArgmapInfrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        prop_arg_relation_map_vdata, prop_arg_relation_reco_vdata
    ])
    result = handler.evaluate(prop_arg_relation_map_vdata, 
                            prop_arg_relation_reco_vdata, 
                            request)
    
    pprint(result)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_relation_cohere_handler_arg_prop_relation(arg_prop_relation_map_vdata, 
                                                 arg_prop_relation_reco_vdata):
    handler = ArgmapInfrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        arg_prop_relation_map_vdata, arg_prop_relation_reco_vdata
    ])
    result = handler.evaluate(arg_prop_relation_map_vdata, 
                            arg_prop_relation_reco_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_composite_handler():
    composite = ArgmapInfrecoCoherenceHandler()
    
    # Check that default handlers are initialized
    assert len(composite.handlers) == 2
    assert any(isinstance(h, ArgmapInfrecoElemCohereHandler) for h in composite.handlers)
    assert any(isinstance(h, ArgmapInfrecoRelationCohereHandler) for h in composite.handlers)


def test_composite_handler_with_custom_filters():
    custom_filter1 = lambda vd: vd.dtype == VerificationDType.argdown and "test_map" in vd.id  # noqa: E731
    custom_filter2 = lambda vd: vd.dtype == VerificationDType.argdown and "test_reco" in vd.id  # noqa: E731
    custom_from_key = "premises"
    
    composite = ArgmapInfrecoCoherenceHandler(filters=(custom_filter1, custom_filter2), from_key=custom_from_key)
    
    # Check that filters were passed to child handlers
    for handler in composite.handlers:
        assert handler.filters == (custom_filter1, custom_filter2)
        assert handler.from_key == custom_from_key


def test_composite_handler_process_request(verification_request_with_valid_data, valid_map_vdata, valid_reco_vdata):
    composite = ArgmapInfrecoCoherenceHandler()
    
    # Mock evaluation results for child handlers
    class MockHandler(BaseArgmapInfrecoCoherenceHandler):
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
    handler = ArgmapInfrecoElemCohereHandler()
    
    # Test with wrong data type for argdown map
    with pytest.raises(AssertionError):
        wrong_map_vdata = PrimaryVerificationData(
            id="test", 
            dtype=VerificationDType.argdown, 
            data="not a graph",
            metadata={"filename": "map.ad"}
        )
        handler.evaluate(
            wrong_map_vdata,
            PrimaryVerificationData(
                id="test", 
                dtype=VerificationDType.argdown, 
                data=ArgdownMultiDiGraph(),
                metadata={"filename": "reconstructions.ad"}
            ),
            VerificationRequest(inputs="test")
        )
    
    # Test with wrong data type for argdown reco
    with pytest.raises(AssertionError):
        handler.evaluate(
            PrimaryVerificationData(
                id="test", 
                dtype=VerificationDType.argdown, 
                data=ArgdownMultiDiGraph(),
                metadata={"filename": "map.ad"}
            ),
            PrimaryVerificationData(
                id="test", 
                dtype=VerificationDType.argdown, 
                data="not a graph",
                metadata={"filename": "reconstructions.ad"}
            ),
            VerificationRequest(inputs="test")
        )


def test_is_applicable():
    handler = ArgmapInfrecoElemCohereHandler()
    
    # Create test data
    map_vdata = PrimaryVerificationData(
        id="map", 
        dtype=VerificationDType.argdown, 
        data=ArgdownMultiDiGraph(),
        metadata={"filename": "map.ad"}
    )
    reco_vdata = PrimaryVerificationData(
        id="reco", 
        dtype=VerificationDType.argdown, 
        data=ArgdownMultiDiGraph(),
        metadata={"filename": "reconstructions.ad"}
    )
    xml_vdata = PrimaryVerificationData(
        id="xml", 
        dtype=VerificationDType.xml, 
        data="xml data"
    )
    
    # Create request with all data
    request = VerificationRequest(
        inputs="test", 
        verification_data=[map_vdata, reco_vdata, xml_vdata]
    )
    
    # Test applicable case
    assert handler.is_applicable(map_vdata, reco_vdata, request) is True
    
    # Test non-applicable cases
    assert handler.is_applicable(map_vdata, xml_vdata, request) is False
    assert handler.is_applicable(reco_vdata, map_vdata, request) is False
    
    # Test with custom filters
    custom_handler = ArgmapInfrecoElemCohereHandler(
        filters=(
            lambda vd: vd.id == "custom_map",
            lambda vd: vd.id == "custom_reco"
        )
    )
    
    assert custom_handler.is_applicable(map_vdata, reco_vdata, request) is False
    
    custom_map_vdata = PrimaryVerificationData(
        id="custom_map", 
        dtype=VerificationDType.argdown, 
        data=ArgdownMultiDiGraph()
    )
    custom_reco_vdata = PrimaryVerificationData(
        id="custom_reco", 
        dtype=VerificationDType.argdown, 
        data=ArgdownMultiDiGraph()
    )
    
    request_with_custom = VerificationRequest(
        inputs="test", 
        verification_data=[custom_map_vdata, custom_reco_vdata]
    )
    
    assert custom_handler.is_applicable(custom_map_vdata, custom_reco_vdata, request_with_custom) is True


def test_real_world_example():
    # A more complex real-world example with multiple arguments and relations
    map_text = dedent("""
    ```argdown {filename="map.ad"}

    <Free Will>: Humans have free will.
    <Determinism>: All events are determined by prior causes.
    <Compatibilism>: Free will and determinism are compatible.
    
    [Autonomy]: Humans have autonomy.
        - <Determinism>
        + <Free Will>

    <Compatibilism>
        -> <Determinism>
    ```
    """)
    
    reco_text = dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Free Will>: Humans have free will.
    
    (P1) We experience making choices.
    (P2) We feel responsible for our actions.
    -- {from: ["P1", "P2"]} --
    (C1) [Autonomy]: Humans have free will.
    
    <Determinism>: All events are determined by prior causes.
    
    (P1) Physics shows the universe follows deterministic laws.
    (P2) Human decisions are physical processes.
    -- {from: ["P1", "P2"]} --
    (C1) All events, including human decisions, are determined by prior causes.
    (P3) If all events are determined, there is no free choice.
    -- {from: ["C1", "P3"]} --
    (C2) NOT: Humans have free will.
    
    <Compatibilism>: Free will and determinism are compatible.
    
    (P1) Free will means the ability to act according to one's own desires.
    (P2) Deterministic causation doesn't prevent acting on desires.
    -- {from: ["P1", "P2"]} --
    (C1) Free will is compatible with determinism.
    (P3) Determinism assumes a simplistic view of causation.
    -- {from: ["P3", "C1"]} --
    (C2) NOT: If all events are determined, there is no free choice.
    ```
    """)
    
    map_graph = parse_fenced_argdown(map_text)
    reco_graph = parse_fenced_argdown(reco_text)
    
    map_vdata = PrimaryVerificationData(
        id="real_map", 
        dtype=VerificationDType.argdown, 
        data=map_graph,
        metadata={"filename": "map.ad"}
    )
    reco_vdata = PrimaryVerificationData(
        id="real_reco", 
        dtype=VerificationDType.argdown, 
        data=reco_graph,
        metadata={"filename": "reconstructions.ad"}
    )
    
    request = VerificationRequest(
        inputs="test", 
        source="real world example", 
        verification_data=[map_vdata, reco_vdata]
    )
    
    # Test with composite handler
    composite = ArgmapInfrecoCoherenceHandler()
    result_request = composite.process(request)
    
    # Should have results for both handlers
    assert len(result_request.results) == 2
    
    # Both should be valid for this well-formed example
    for result in result_request.results:
        assert result.is_valid is True, f"Handler {result.verifier_id} failed with message: {result.message}"
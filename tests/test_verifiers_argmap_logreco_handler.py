from pprint import pprint  # noqa: F401
import pytest
from textwrap import dedent
from pyargdown import parse_argdown, ArgdownMultiDiGraph

from argdown_feedback.verifiers.coherence.argmap_logreco_handler import (
    BaseArgmapLogrecoCoherenceHandler,
    ArgmapLogrecoElemCohereHandler,
    ArgmapLogrecoRelationCohereHandler,
    ArgmapLogrecoCoherenceHandler
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
def valid_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim. {formalization: "P", declarations: {"P": "FirstClaim"}}
    -- {from: ["P1"]} --
    (C1) [A2]: Second claim. {formalization: "Q", declarations: {"Q": "SecondClaim"}}
    
    <Argument2>: Second argument.
    
    (P1) [A3]: Third claim. {formalization: "R", declarations: {"R": "ThirdClaim"}}
    (P2) Implicit premise. {formalization: "R -> -Q", declarations: {}}
    -- {from: ["P1", "P2"]} --
    (C1) NOT: Second claim. {formalization: "-Q"}
        -> [A2] // Attack relation
    ```
    """)


@pytest.fixture
def valid_logreco_graph(valid_logreco_text):
    return parse_fenced_argdown(valid_logreco_text)


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
def missing_argument_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim. {formalization: "P", declarations: {"P": "FirstClaim"}}
    -- {from: ["P1"]} --
    (C1) First conclusion. {formalization: "Q", declarations: {"Q": "Conclusion"}}
    ```
    """)


@pytest.fixture
def missing_argument_logreco_graph(missing_argument_logreco_text):
    return parse_fenced_argdown(missing_argument_logreco_text)


@pytest.fixture
def missing_claim_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim. {formalization: "P", declarations: {"P": "FirstClaim"}}
    -- {from: ["P1"]} --
    (C1) Second claim. {formalization: "Q", declarations: {"Q": "SecondClaim"}}
    
    <Argument2>: Second argument.
    
    (P1) [A3]: Third claim. {formalization: "R", declarations: {"R": "ThirdClaim"}}
    (P2) Implicit premise. {formalization: "R -> -Q", declarations: {}}
    -- {from: ["P1", "P2"]} --
    (C1) NOT: Second claim. {formalization: "-Q"}
        // -> [A2] Completely missing claim [A2]
    ```
    """)


@pytest.fixture
def missing_claim_logreco_graph(missing_claim_logreco_text):
    return parse_fenced_argdown(missing_claim_logreco_text)


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
def valid_relation_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim. {formalization: "P", declarations: {"P": "FirstClaim"}}
    -- {from: ["P1"]} --
    (C1) [A2]: Second claim. {formalization: "Q", declarations: {"Q": "SecondClaim"}}
    ```
    """)


@pytest.fixture
def valid_relation_logreco_graph(valid_relation_logreco_text):
    return parse_fenced_argdown(valid_relation_logreco_text)


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
def invalid_support_relation_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim. {formalization: "P", declarations: {"P": "FirstClaim"}}
    -- {from: ["P1"]} --
    (C1) Different conclusion. {formalization: "R", declarations: {"R": "Different"}}
        -> [A2] // Attack relation, unlike in map
    ```
    """)


@pytest.fixture
def invalid_support_relation_logreco_graph(invalid_support_relation_logreco_text):
    return parse_fenced_argdown(invalid_support_relation_logreco_text)


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
def invalid_attack_relation_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim. {formalization: "P", declarations: {"P": "FirstClaim"}}
    -- {from: ["P1"]} --
    (C1) Some conclusion. {formalization: "Q", declarations: {"Q": "Some"}}
        +> [A2] // Support relation, unlike in map
    ```
    """)


@pytest.fixture
def invalid_attack_relation_logreco_graph(invalid_attack_relation_logreco_text):
    return parse_fenced_argdown(invalid_attack_relation_logreco_text)


@pytest.fixture
def grounded_relation_map_text():
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
def grounded_relation_map_graph(grounded_relation_map_text):
    return parse_fenced_argdown(grounded_relation_map_text)


@pytest.fixture
def grounded_relation_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim. {formalization: "P", declarations: {"P": "FirstClaim"}}
    -- {from: ["P1"]} --
    (C1) [A2]: Second claim. {formalization: "Q", declarations: {"Q": "SecondClaim"}}
    ```
    """)


@pytest.fixture
def grounded_relation_logreco_graph(grounded_relation_logreco_text):
    return parse_fenced_argdown(grounded_relation_logreco_text)


@pytest.fixture
def ungrounded_relation_map_text():
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
def ungrounded_relation_map_graph(ungrounded_relation_map_text):
    return parse_fenced_argdown(ungrounded_relation_map_text)


@pytest.fixture
def ungrounded_relation_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) [A1]: First claim. {formalization: "P", declarations: {"P": "FirstClaim"}}
    -- {from: ["P1"]} --
    (C1) Unrelated conclusion. {formalization: "R", declarations: {"R": "Unrelated"}}

    <Argument1>
        +> [A2] // Support relation not grounded in first argument's conclusion
    ```
    """)


@pytest.fixture
def ungrounded_relation_logreco_graph(ungrounded_relation_logreco_text):
    return parse_fenced_argdown(ungrounded_relation_logreco_text)


@pytest.fixture
def indirectly_supported_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}    
    <Argument1>
        + [A1]
            - <Argument2
    ```
    """)


@pytest.fixture
def indirectly_supported_map_graph(indirectly_supported_map_text):
    return parse_fenced_argdown(indirectly_supported_map_text)


@pytest.fixture
def indirectly_supported_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument2>: Second argument.

    (P1) Premise. {formalization: "S", declarations: {"S": "SomePremise"}}
    -- {from: ["P1"]} --
    (C1) First  claim. {formalization: "P", declarations: {"P": "FirstClaim"}}

    <Argument1>: First argument.
                                 
    (P2) Not first claim. {formalization: "-P", declarations: {"P": "FirstClaim"}} 
    -- {from: ["P2"]} --
    (C2) Second claim. {formalization: "Q", declarations: {"Q": "SecondClaim"}}
    ```
    """)


@pytest.fixture
def indirectly_supported_logreco_graph(indirectly_supported_logreco_text):
    return parse_fenced_argdown(indirectly_supported_logreco_text)


@pytest.fixture
def formal_contradiction_map_text():
    return dedent("""
    ```argdown {filename="map.ad"}
    [A1]: First claim.
    [A2]: Second claim.
    
    <Argument1>
        +> [A1]
    <Argument2>
        +> [A2]
    
    [A2]
        >< [A1]
    ```
    """)


@pytest.fixture
def formal_contradiction_map_graph(formal_contradiction_map_text):
    return parse_fenced_argdown(formal_contradiction_map_text)


@pytest.fixture
def formal_contradiction_logreco_text():
    return dedent("""
    ```argdown {filename="reconstructions.ad"}
    <Argument1>: First argument.

    (P1) Some premise. {formalization: "S", declarations: {"S": "SomePremise"}}
    -- {from: ["P1"]} --
    (C1) [A1]: First claim. {formalization: "P", declarations: {"P": "FirstClaim"}}

    <Argument2>: Second argument.
    
    (P1) Another premise. {formalization: "T", declarations: {"T": "AnotherPremise"}}
    -- {from: ["P1"]} --
    (C1) [A2]: Second claim. {formalization: "-P", declarations: {}}
    ```
    """)


@pytest.fixture
def formal_contradiction_logreco_graph(formal_contradiction_logreco_text):
    return parse_fenced_argdown(formal_contradiction_logreco_text)


@pytest.fixture
def valid_map_vdata(valid_map_graph):
    return PrimaryVerificationData(
        id="map_test", 
        dtype=VerificationDType.argdown, 
        data=valid_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def valid_logreco_vdata(valid_logreco_graph):
    return PrimaryVerificationData(
        id="logreco_test", 
        dtype=VerificationDType.argdown, 
        data=valid_logreco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def verification_request_with_valid_data(valid_map_vdata, valid_logreco_vdata):
    return VerificationRequest(
        inputs="test", 
        source="test source", 
        verification_data=[valid_map_vdata, valid_logreco_vdata]
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
def missing_argument_logreco_vdata(missing_argument_logreco_graph):
    return PrimaryVerificationData(
        id="missing_arg_logreco", 
        dtype=VerificationDType.argdown, 
        data=missing_argument_logreco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def missing_claim_logreco_vdata(missing_claim_logreco_graph):
    return PrimaryVerificationData(
        id="missing_claim_logreco", 
        dtype=VerificationDType.argdown, 
        data=missing_claim_logreco_graph,
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
def valid_relation_logreco_vdata(valid_relation_logreco_graph):
    return PrimaryVerificationData(
        id="valid_relation_logreco", 
        dtype=VerificationDType.argdown, 
        data=valid_relation_logreco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def verification_request_with_valid_relations(valid_relation_map_vdata, valid_relation_logreco_vdata):
    return VerificationRequest(
        inputs="test", 
        source="test source", 
        verification_data=[valid_relation_map_vdata, valid_relation_logreco_vdata]
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
def invalid_support_relation_logreco_vdata(invalid_support_relation_logreco_graph):
    return PrimaryVerificationData(
        id="invalid_support_relation_logreco", 
        dtype=VerificationDType.argdown, 
        data=invalid_support_relation_logreco_graph,
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
def invalid_attack_relation_logreco_vdata(invalid_attack_relation_logreco_graph):
    return PrimaryVerificationData(
        id="invalid_attack_relation_logreco", 
        dtype=VerificationDType.argdown, 
        data=invalid_attack_relation_logreco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def grounded_relation_map_vdata(grounded_relation_map_graph):
    return PrimaryVerificationData(
        id="grounded_relation_map", 
        dtype=VerificationDType.argdown, 
        data=grounded_relation_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def grounded_relation_logreco_vdata(grounded_relation_logreco_graph):
    return PrimaryVerificationData(
        id="grounded_relation_logreco", 
        dtype=VerificationDType.argdown, 
        data=grounded_relation_logreco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def ungrounded_relation_map_vdata(ungrounded_relation_map_graph):
    return PrimaryVerificationData(
        id="ungrounded_relation_map", 
        dtype=VerificationDType.argdown, 
        data=ungrounded_relation_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def ungrounded_relation_logreco_vdata(ungrounded_relation_logreco_graph):
    return PrimaryVerificationData(
        id="ungrounded_relation_logreco", 
        dtype=VerificationDType.argdown, 
        data=ungrounded_relation_logreco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def indirectly_supported_map_vdata(indirectly_supported_map_graph):
    return PrimaryVerificationData(
        id="indirectly_supported_map", 
        dtype=VerificationDType.argdown, 
        data=indirectly_supported_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def indirectly_supported_logreco_vdata(indirectly_supported_logreco_graph):
    return PrimaryVerificationData(
        id="indirectly_supported_logreco", 
        dtype=VerificationDType.argdown, 
        data=indirectly_supported_logreco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


@pytest.fixture
def formal_contradiction_map_vdata(formal_contradiction_map_graph):
    return PrimaryVerificationData(
        id="formal_contradiction_map", 
        dtype=VerificationDType.argdown, 
        data=formal_contradiction_map_graph,
        metadata={"filename": "map.ad"}
    )


@pytest.fixture
def formal_contradiction_logreco_vdata(formal_contradiction_logreco_graph):
    return PrimaryVerificationData(
        id="formal_contradiction_logreco", 
        dtype=VerificationDType.argdown, 
        data=formal_contradiction_logreco_graph,
        metadata={"filename": "reconstructions.ad"}
    )


def test_get_labels(valid_map_graph, valid_logreco_graph):
    map_alabels, reco_alabels, map_prop_labels, reco_prop_labels = BaseArgmapLogrecoCoherenceHandler.get_labels(
        valid_map_graph, valid_logreco_graph
    )
    
    assert set(map_alabels) == {"Argument1", "Argument2"}
    assert set(reco_alabels) == {"Argument1", "Argument2"}
    assert set(map_prop_labels) == {"A1", "A2", "A3"}
    assert {"A1", "A2", "A3"}.issubset(set(reco_prop_labels))
    

def test_elem_cohere_handler_valid(verification_request_with_valid_data, valid_map_vdata, valid_logreco_vdata):
    handler = ArgmapLogrecoElemCohereHandler()
    result = handler.evaluate(valid_map_vdata, valid_logreco_vdata, verification_request_with_valid_data)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None
    assert result.verification_data_references == ["map_test", "logreco_test"]


def test_elem_cohere_handler_missing_argument_in_logreco(missing_argument_logreco_vdata, valid_map_vdata):
    handler = ArgmapLogrecoElemCohereHandler()
    result = handler.evaluate(valid_map_vdata, missing_argument_logreco_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "is not reconstructed" in result.message


def test_elem_cohere_handler_missing_argument_in_map(missing_argument_map_vdata, valid_logreco_vdata):
    handler = ArgmapLogrecoElemCohereHandler()
    result = handler.evaluate(missing_argument_map_vdata, valid_logreco_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "is not in the map" in result.message


def test_elem_cohere_handler_missing_claim(missing_claim_logreco_vdata, valid_map_vdata):
    handler = ArgmapLogrecoElemCohereHandler()
    result = handler.evaluate(valid_map_vdata, missing_claim_logreco_vdata, 
                            VerificationRequest(inputs="test"))
    
    assert result is not None
    assert result.is_valid is False
    assert "has no corresponding proposition" in result.message


def test_relation_cohere_handler_valid(verification_request_with_valid_relations, 
                                      valid_relation_map_vdata, 
                                      valid_relation_logreco_vdata):
    handler = ArgmapLogrecoRelationCohereHandler()
    result = handler.evaluate(valid_relation_map_vdata, 
                            valid_relation_logreco_vdata, 
                            verification_request_with_valid_relations)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_relation_cohere_handler_invalid_support(invalid_support_relation_map_vdata, 
                                               invalid_support_relation_logreco_vdata):
    handler = ArgmapLogrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        invalid_support_relation_map_vdata, invalid_support_relation_logreco_vdata
    ])
    result = handler.evaluate(invalid_support_relation_map_vdata, 
                            invalid_support_relation_logreco_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is False
    assert "not matched by any relation" in result.message
    assert "support relation" in result.message.lower()


def test_relation_cohere_handler_invalid_attack(invalid_attack_relation_map_vdata, 
                                               invalid_attack_relation_logreco_vdata):
    handler = ArgmapLogrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        invalid_attack_relation_map_vdata, invalid_attack_relation_logreco_vdata
    ])
    result = handler.evaluate(invalid_attack_relation_map_vdata, 
                            invalid_attack_relation_logreco_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is False
    assert "not matched by any relation" in result.message
    assert "attack relation" in result.message.lower()


def test_relation_cohere_handler_grounded_relation(grounded_relation_map_vdata, 
                                                 grounded_relation_logreco_vdata):
    handler = ArgmapLogrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        grounded_relation_map_vdata, grounded_relation_logreco_vdata
    ])
    result = handler.evaluate(grounded_relation_map_vdata, 
                            grounded_relation_logreco_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_relation_cohere_handler_ungrounded_relation(ungrounded_relation_map_vdata, 
                                                   ungrounded_relation_logreco_vdata):
    handler = ArgmapLogrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        ungrounded_relation_map_vdata, ungrounded_relation_logreco_vdata
    ])
    result = handler.evaluate(ungrounded_relation_map_vdata, 
                            ungrounded_relation_logreco_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is False
    assert "not grounded in logical argument reconstructions" in result.message


def test_relation_cohere_handler_indirectly_supported(indirectly_supported_map_vdata, 
                                                    indirectly_supported_logreco_vdata):
    handler = ArgmapLogrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        indirectly_supported_map_vdata, indirectly_supported_logreco_vdata
    ])
    result = handler.evaluate(indirectly_supported_map_vdata, 
                            indirectly_supported_logreco_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_relation_cohere_handler_formal_contradiction(formal_contradiction_map_vdata, 
                                                    formal_contradiction_logreco_vdata):
    handler = ArgmapLogrecoRelationCohereHandler()
    request = VerificationRequest(inputs="test", verification_data=[
        formal_contradiction_map_vdata, formal_contradiction_logreco_vdata
    ])
    result = handler.evaluate(formal_contradiction_map_vdata, 
                            formal_contradiction_logreco_vdata, 
                            request)
    
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_composite_handler():
    composite = ArgmapLogrecoCoherenceHandler()
    
    # Check that default handlers are initialized
    assert len(composite.handlers) == 2
    assert any(isinstance(h, ArgmapLogrecoElemCohereHandler) for h in composite.handlers)
    assert any(isinstance(h, ArgmapLogrecoRelationCohereHandler) for h in composite.handlers)


def test_composite_handler_with_custom_filters():
    custom_filter1 = lambda vd: vd.dtype == VerificationDType.argdown and "test_map" in vd.id  # noqa: E731
    custom_filter2 = lambda vd: vd.dtype == VerificationDType.argdown and "test_logreco" in vd.id  # noqa: E731
    custom_from_key = "premises"
    
    composite = ArgmapLogrecoCoherenceHandler(filters=(custom_filter1, custom_filter2), from_key=custom_from_key)
    
    # Check that filters were passed to child handlers
    for handler in composite.handlers:
        assert handler.filters == (custom_filter1, custom_filter2)
        assert handler.from_key == custom_from_key


def test_composite_handler_process_request(verification_request_with_valid_data, valid_map_vdata, valid_logreco_vdata):
    composite = ArgmapLogrecoCoherenceHandler()
    
    # Mock evaluation results for child handlers
    class MockHandler(BaseArgmapLogrecoCoherenceHandler):
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
    
    result_request = composite.handle(verification_request_with_valid_data)
    
    # Should have results for both handlers
    assert len(result_request.results) == 2
    assert any(r.verifier_id == "TestHandler1" and r.is_valid for r in result_request.results)
    assert any(r.verifier_id == "TestHandler2" and not r.is_valid for r in result_request.results)


def test_handle_wrong_data_types():
    handler = ArgmapLogrecoElemCohereHandler()
    
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
    
    # Test with wrong data type for argdown logreco
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
    handler = ArgmapLogrecoElemCohereHandler()
    
    # Create test data
    map_vdata = PrimaryVerificationData(
        id="map", 
        dtype=VerificationDType.argdown, 
        data=ArgdownMultiDiGraph(),
        metadata={"filename": "map.ad"}
    )
    logreco_vdata = PrimaryVerificationData(
        id="logreco", 
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
        verification_data=[map_vdata, logreco_vdata, xml_vdata]
    )
    
    # Test applicable case
    assert handler.is_applicable(map_vdata, logreco_vdata, request) is True
    
    # Test non-applicable cases
    assert handler.is_applicable(map_vdata, xml_vdata, request) is False
    assert handler.is_applicable(logreco_vdata, map_vdata, request) is False
    
    # Test with custom filters
    custom_handler = ArgmapLogrecoElemCohereHandler(
        filters=(
            lambda vd: vd.id == "custom_map",
            lambda vd: vd.id == "custom_logreco"
        )
    )
    
    assert custom_handler.is_applicable(map_vdata, logreco_vdata, request) is False
    
    custom_map_vdata = PrimaryVerificationData(
        id="custom_map", 
        dtype=VerificationDType.argdown, 
        data=ArgdownMultiDiGraph()
    )
    custom_logreco_vdata = PrimaryVerificationData(
        id="custom_logreco", 
        dtype=VerificationDType.argdown, 
        data=ArgdownMultiDiGraph()
    )
    
    request_with_custom = VerificationRequest(
        inputs="test", 
        verification_data=[custom_map_vdata, custom_logreco_vdata]
    )
    
    assert custom_handler.is_applicable(custom_map_vdata, custom_logreco_vdata, request_with_custom) is True


def test_real_world_example():
    # A more complex real-world example with multiple arguments and relations
    map_text = dedent("""
    ```argdown {filename="map.ad"}
                      
    [Free Will]: Humans have free will.
    [Determinism]: All events are determined by prior causes.
    [Compatibilism]: Free will and determinism are compatible.
    
    <FreeWill_A>
        +> [Free Will]
    <Determinism_A>
        <+ [Determinism]
        -> [Free Will]
    <Compatibilism_A>
        + [Compatibilism]
        -> <Determinism_A>
    ```
    """).strip("\n ")
    
    logreco_text = dedent("""
    ```argdown {filename="reconstructions.ad"}
    <FreeWill_A>: Humans have free will.
    
    (P1) We experience making choices. {formalization: "E(c)", declarations: {"E": "Experience", "c": "choices"}}
    (P2) We feel responsible for our actions. {formalization: "F(r)", declarations: {"F": "Feel", "r": "responsibility"}}
    -- {from: ["P1", "P2"]} --
    (C1) [Free Will]: Humans have free will. {formalization: "W", declarations: {"W": "FreeWill"}}
    
    <Determinism_A>: All events are determined by prior causes.
    
    (P1) [Determinism]: Physics shows the universe follows deterministic laws. {formalization: "D(u)", declarations: {"D": "Deterministic", "u": "universe"}}
    (P2) Human decisions are physical processes. {formalization: "P(d)", declarations: {"P": "Physical", "d": "decisions"}}
    -- {from: ["P1", "P2"]} --
    (C1) All events, including human decisions, are determined by prior causes. {formalization: "A(d)", declarations: {"A": "AllDetermined"}}
    (P3) [Connecting Premise]: If all events are determined, there is no free choice. {formalization: "A(d) -> -W", declarations: {}}
    -- {from: ["C1", "P3"]} --
    (C2) Humans do not have free will. {formalization: "-W"}
        >< [Free Will] // Attack relation
    
    <Compatibilism_A>: Free will and determinism are compatible.
    
    (P1) [Compatibilism]: Free will means the ability to act according to one's own desires. {formalization: "W <-> D(a)", declarations: {"D": "Desires", "a": "actions"}}
    (P2) Deterministic causation doesn't prevent acting on desires. {formalization: "A(d) & D(a)", declarations: {}}
    -- {from: ["P1", "P2"]} --
    (C1) Free will is compatible with determinism. {formalization: "C(w,d)", declarations: {"C": "Compatible", "w": "will", "d": "determinism"}}
    (P3) Determinism assumes a simplistic view of causation. {formalization: "S(d)", declarations: {"S": "Simplistic"}}
    -- {from: ["P3", "C1"]} --
    (C2) The deterministic argument against free will is flawed. {formalization: "-((A(d) -> -W))", declarations: {}}
        >< [Connecting Premise]
    ```
    """)
    
    map_graph = parse_fenced_argdown(map_text)
    pprint(map_graph.arguments)
    logreco_graph = parse_fenced_argdown(logreco_text)
    pprint(logreco_graph.arguments)

    map_vdata = PrimaryVerificationData(
        id="real_map", 
        dtype=VerificationDType.argdown, 
        data=map_graph,
        metadata={"filename": "map.ad"}
    )
    logreco_vdata = PrimaryVerificationData(
        id="real_logreco", 
        dtype=VerificationDType.argdown, 
        data=logreco_graph,
        metadata={"filename": "reconstructions.ad"}
    )
    
    request = VerificationRequest(
        inputs="test", 
        source="real world example", 
        verification_data=[map_vdata, logreco_vdata]
    )
    
    # Test with composite handler
    composite = ArgmapLogrecoCoherenceHandler()
    result_request = composite.handle(request)
    
    # Should have results for both handlers
    assert len(result_request.results) == 2
    
    # Both should be valid for this well-formed example
    for result in result_request.results:
        assert result.is_valid is True, f"Handler {result.verifier_id} failed with message: {result.message}"
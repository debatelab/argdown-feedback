from pprint import pprint
import pytest
from textwrap import dedent

from nltk.sem.logic import Expression  # type: ignore
from pyargdown import parse_argdown

from argdown_feedback.verifiers.core.logreco_handler import (
    WellFormedFormulasHandler,
    GlobalDeductiveValidityHandler,
    LocalDeductiveValidityHandler,
    AllPremisesRelevantHandler,
    PremisesConsistentHandler,
    FormallyGroundedRelationsHandler,
    LogRecoCompositeHandler,
)
from argdown_feedback.verifiers.verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
)


def parse_fenced_argdown(argdown_text: str):
    argdown_text = argdown_text.strip("\n ")
    argdown_text = "\n".join(argdown_text.splitlines()[1:-1])
    return parse_argdown(argdown_text)


@pytest.fixture
def valid_logreco_text():
    return dedent("""
    ```argdown
    <Socrates>: Socrates is mortal.

    (P1) All men are mortal. {formalization: "all x.(F(x) -> Go(x))", declarations: {"F": "Man", "Go": "Mortal"}}
    (P2) Socrates is a man. {formalization: "F(a)", declarations: {"a": "socrates"}}
    -- {from: ["P1", "P2"]} --
    (C1) Socrates is mortal. {formalization: "Go(a)"}
    ```
    """)


@pytest.fixture
def valid_logreco_graph(valid_logreco_text):
    return parse_fenced_argdown(valid_logreco_text)


@pytest.fixture
def valid_vdata(valid_logreco_graph):
    return PrimaryVerificationData(
        id="test_logreco_1",
        dtype=VerificationDType.argdown,
        data=valid_logreco_graph,
        code_snippet="test",
    )


@pytest.fixture
def verification_request_with_valid_logreco(valid_vdata):
    return VerificationRequest(
        inputs="test", source=None, verification_data=[valid_vdata]
    )


@pytest.fixture
def invalid_formalization_text():
    return dedent("""
    ```argdown
    <Socrates>: Socrates is mortal.

    (P1) All men are mortal. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "Man", "G": "Mortal"}}
    (P2) Socrates is a man. {formalization: "F(a", declarations: {"a": "socrates"}}
    -- {from: ["P1", "P2"]} --
    (C1) Socrates is mortal. {formalization: "Mortal(socrates)"}
    ```
    """)


@pytest.fixture
def invalid_formalization_graph(invalid_formalization_text):
    return parse_fenced_argdown(invalid_formalization_text)


@pytest.fixture
def invalid_formalization_vdata(invalid_formalization_graph):
    return PrimaryVerificationData(
        id="test_logreco_2",
        dtype=VerificationDType.argdown,
        data=invalid_formalization_graph,
        code_snippet="test",
    )


@pytest.fixture
def deductively_invalid_text():
    return dedent("""
    ```argdown
    <Invalid>: Invalid deduction.

    (P1) All men are mortal. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "man", "G": "mortal"}}
    (P2) Socrates is a man. {formalization: "F(a)", declarations: {"a": "socrates"}}
    -- {from: ["P1", "P2"]} --
    (C1) All entities are mortal. {formalization: "all x.F(x)"}
    ```
    """)


@pytest.fixture
def deductively_invalid_graph(deductively_invalid_text):
    return parse_fenced_argdown(deductively_invalid_text)


@pytest.fixture
def deductively_invalid_vdata(deductively_invalid_graph):
    return PrimaryVerificationData(
        id="test_logreco_3",
        dtype=VerificationDType.argdown,
        data=deductively_invalid_graph,
        code_snippet="test",
    )


@pytest.fixture
def inconsistent_premises_text():
    return dedent("""
    ```argdown
    <Inconsistent>: Premises are inconsistent.

    (P1) All men are mortal. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "Man", "G": "Mortal"}}
    (P2) Socrates is a man. {formalization: "F(a)", declarations: {"a": "socrates"}}
    (P3) Socrates is not mortal. {formalization: "-G(a)"}
    -- {from: ["P1", "P2", "P3"]} --
    (C1) Contradiction. {formalization: "all x.(F(x) | -F(x))"}
    ```
    """)


@pytest.fixture
def inconsistent_premises_graph(inconsistent_premises_text):
    return parse_fenced_argdown(inconsistent_premises_text)


@pytest.fixture
def inconsistent_premises_vdata(inconsistent_premises_graph):
    return PrimaryVerificationData(
        id="test_logreco_4",
        dtype=VerificationDType.argdown,
        data=inconsistent_premises_graph,
        code_snippet="test",
    )


@pytest.fixture
def irrelevant_premise_text():
    return dedent("""
    ```argdown
    <Irrelevant>: Contains an irrelevant premise.

    (P1) All men are mortal. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "Man", "G": "Mortal"}}
    (P2) Socrates is a man. {formalization: "F(a)", declarations: {"a": "socrates"}}
    (P3) Athens is in Greece. {formalization: "H(b)", declarations: {"H": "InGreece", "b": "athens"}}
    -- {from: ["P1", "P2", "P3"]} --
    (C1) Socrates is mortal. {formalization: "G(a)"}
    ```
    """)


@pytest.fixture
def irrelevant_premise_graph(irrelevant_premise_text):
    return parse_fenced_argdown(irrelevant_premise_text)


@pytest.fixture
def irrelevant_premise_vdata(irrelevant_premise_graph):
    return PrimaryVerificationData(
        id="test_logreco_5",
        dtype=VerificationDType.argdown,
        data=irrelevant_premise_graph,
        code_snippet="test",
    )


@pytest.fixture
def dialectical_relations_text():
    return dedent("""
    ```argdown
    <Argument 1>: First argument.

    (P1) All men are mortal. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "Man", "G": "Mortal"}}
    (P2) Socrates is a man. {formalization: "F(a)", declarations: {"a": "socrates"}}
    -- {from: ["P1", "P2"]} --
    (C1) Socrates is mortal. {formalization: "G(a)"}

    [Immortal]: Socrates is immortal. {formalization: "-G(a)"}

    [Immortal]
        >< [C1]
    ```
    """)


@pytest.fixture
def dialectical_relations_graph(dialectical_relations_text):
    return parse_fenced_argdown(dialectical_relations_text)


@pytest.fixture
def dialectical_relations_vdata(dialectical_relations_graph):
    return PrimaryVerificationData(
        id="test_logreco_6",
        dtype=VerificationDType.argdown,
        data=dialectical_relations_graph,
        code_snippet="test",
    )


@pytest.fixture
def missing_formalization_text():
    return dedent("""
    ```argdown
    <Socrates>: Socrates is mortal.

    (P1) All men are mortal.
    (P2) Socrates is a man. {formalization: "F(a)", declarations: {"F": "Man", "a": "socrates"}}
    -- {from: ["P1", "P2"]} --
    (C1) Socrates is mortal. {formalization: "G(a)", declarations: {"G": "Mortal"}}
    ```
    """)


@pytest.fixture
def missing_formalization_graph(missing_formalization_text):
    return parse_fenced_argdown(missing_formalization_text)


@pytest.fixture
def missing_formalization_vdata(missing_formalization_graph):
    return PrimaryVerificationData(
        id="test_logreco_7",
        dtype=VerificationDType.argdown,
        data=missing_formalization_graph,
        code_snippet="test",
    )


@pytest.fixture
def local_validity_text():
    return dedent("""
    ```argdown
    <LocalInvalid>: Has a locally invalid inference.

    (P1) All men are mortal. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "Man", "G": "Mortal"}}
    (P2) Socrates is a man. {formalization: "F(a)", declarations: {"a": "socrates"}}
    -- {from: ["P1"]} --
    (C1) Socrates is mortal. {formalization: "all x.F(x)"}
    ```
    """)


@pytest.fixture
def local_validity_graph(local_validity_text):
    return parse_fenced_argdown(local_validity_text)


@pytest.fixture
def local_validity_vdata(local_validity_graph):
    return PrimaryVerificationData(
        id="test_logreco_8",
        dtype=VerificationDType.argdown,
        data=local_validity_graph,
        code_snippet="test",
    )


def test_wellformed_formulas_handler_valid(
    verification_request_with_valid_logreco, valid_vdata
):
    handler = WellFormedFormulasHandler()
    result = handler.evaluate(valid_vdata, verification_request_with_valid_logreco)
    pprint(result)

    assert result is not None
    assert result.is_valid
    assert result.verifier_id == "WellFormedFormulasHandler"
    assert result.details is not None
    assert "all_expressions" in result.details
    assert "all_declarations" in result.details
    assert len(result.details["all_expressions"].keys()) == 3
    assert len(result.details["all_declarations"].keys()) == 3
    assert isinstance(
        next(iter(result.details["all_expressions"].values())), Expression
    )


def test_wellformed_formulas_handler_invalid(invalid_formalization_vdata):
    handler = WellFormedFormulasHandler()
    result = handler.evaluate(
        invalid_formalization_vdata, VerificationRequest(inputs="test")
    )

    assert result is not None
    assert not result.is_valid
    assert "is not a well-formed first-order logic formula" in result.message


def test_wellformed_formulas_handler_missing_formalization(missing_formalization_vdata):
    handler = WellFormedFormulasHandler()
    result = handler.evaluate(
        missing_formalization_vdata, VerificationRequest(inputs="test")
    )

    assert result is not None
    assert not result.is_valid
    assert "lacks inline yaml" in result.message


def test_global_deductive_validity_handler_valid(
    verification_request_with_valid_logreco, valid_vdata
):
    # First run the formulas handler to get the expressions cached
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(
        valid_vdata, verification_request_with_valid_logreco
    )
    verification_request_with_valid_logreco.add_result_record(wff_result)

    handler = GlobalDeductiveValidityHandler()
    result = handler.evaluate(valid_vdata, verification_request_with_valid_logreco)

    assert result is not None
    assert result.is_valid


def test_global_deductive_validity_handler_invalid(deductively_invalid_vdata):
    # Setup verification request with cached formulas
    request = VerificationRequest(
        inputs="test", verification_data=[deductively_invalid_vdata]
    )
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(deductively_invalid_vdata, request)
    request.add_result_record(wff_result)

    handler = GlobalDeductiveValidityHandler()
    result = handler.evaluate(deductively_invalid_vdata, request)

    assert result is not None
    assert not result.is_valid
    assert "not deductively valid" in result.message


def test_local_deductive_validity_handler_valid(
    verification_request_with_valid_logreco, valid_vdata
):
    # First run the formulas handler to get the expressions cached
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(
        valid_vdata, verification_request_with_valid_logreco
    )
    verification_request_with_valid_logreco.add_result_record(wff_result)

    handler = LocalDeductiveValidityHandler()
    result = handler.evaluate(valid_vdata, verification_request_with_valid_logreco)

    assert result is not None
    assert result.is_valid


def test_local_deductive_validity_handler_invalid(local_validity_vdata):
    # Setup verification request with cached formulas
    request = VerificationRequest(
        inputs="test", verification_data=[local_validity_vdata]
    )
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(local_validity_vdata, request)
    request.add_result_record(wff_result)

    handler = LocalDeductiveValidityHandler()
    result = handler.evaluate(local_validity_vdata, request)

    assert result is not None
    assert not result.is_valid
    assert "sub-inference to conclusion" in result.message


def test_all_premises_relevant_handler_irrelevant_premise(irrelevant_premise_vdata):
    # Setup verification request with cached formulas
    request = VerificationRequest(
        inputs="test", verification_data=[irrelevant_premise_vdata]
    )
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(irrelevant_premise_vdata, request)
    request.add_result_record(wff_result)

    handler = AllPremisesRelevantHandler()
    result = handler.evaluate(irrelevant_premise_vdata, request)

    assert result is not None
    assert not result.is_valid
    assert "not required to logically infer" in result.message


def test_all_premises_relevant_handler_valid(
    verification_request_with_valid_logreco, valid_vdata
):
    # First run the formulas handler to get the expressions cached
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(
        valid_vdata, verification_request_with_valid_logreco
    )
    verification_request_with_valid_logreco.add_result_record(wff_result)

    handler = AllPremisesRelevantHandler()
    result = handler.evaluate(valid_vdata, verification_request_with_valid_logreco)

    assert result is not None
    assert result.is_valid


def test_premises_consistent_handler_valid(
    verification_request_with_valid_logreco, valid_vdata
):
    # First run the formulas handler to get the expressions cached
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(
        valid_vdata, verification_request_with_valid_logreco
    )
    verification_request_with_valid_logreco.add_result_record(wff_result)

    handler = PremisesConsistentHandler()
    result = handler.evaluate(valid_vdata, verification_request_with_valid_logreco)

    assert result is not None
    assert result.is_valid


def test_premises_consistent_handler_inconsistent(inconsistent_premises_vdata):
    # Setup verification request with cached formulas
    request = VerificationRequest(
        inputs="test", verification_data=[inconsistent_premises_vdata]
    )
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(inconsistent_premises_vdata, request)
    request.add_result_record(wff_result)

    handler = PremisesConsistentHandler()
    result = handler.evaluate(inconsistent_premises_vdata, request)

    assert result is not None
    assert not result.is_valid
    assert "NOT logically consistent" in result.message


def test_formally_grounded_relations_handler_valid(
    verification_request_with_valid_logreco, valid_vdata
):
    # First run the formulas handler to get the expressions cached
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(
        valid_vdata, verification_request_with_valid_logreco
    )
    verification_request_with_valid_logreco.add_result_record(wff_result)

    handler = FormallyGroundedRelationsHandler()
    result = handler.evaluate(valid_vdata, verification_request_with_valid_logreco)

    assert result is not None
    assert result.is_valid


def test_formally_grounded_relations_handler_with_dialectical_relations(
    dialectical_relations_vdata,
):
    # Setup verification request with cached formulas
    request = VerificationRequest(
        inputs="test", verification_data=[dialectical_relations_vdata]
    )
    wff_handler = WellFormedFormulasHandler()
    wff_result = wff_handler.evaluate(dialectical_relations_vdata, request)
    request.add_result_record(wff_result)

    handler = FormallyGroundedRelationsHandler()
    result = handler.evaluate(dialectical_relations_vdata, request)

    assert result is not None
    assert result.is_valid


def test_logreco_composite_handler():
    composite = LogRecoCompositeHandler()

    # Check that all handlers are initialized
    assert len(composite.handlers) == 6

    # Check handler types
    assert any(isinstance(h, WellFormedFormulasHandler) for h in composite.handlers)
    assert any(
        isinstance(h, GlobalDeductiveValidityHandler) for h in composite.handlers
    )
    assert any(isinstance(h, LocalDeductiveValidityHandler) for h in composite.handlers)
    assert any(isinstance(h, AllPremisesRelevantHandler) for h in composite.handlers)
    assert any(isinstance(h, PremisesConsistentHandler) for h in composite.handlers)
    assert any(
        isinstance(h, FormallyGroundedRelationsHandler) for h in composite.handlers
    )


def test_composite_handler_process_all(verification_request_with_valid_logreco):
    composite = LogRecoCompositeHandler()
    result_request = composite.handle(verification_request_with_valid_logreco)

    # Should have at least one result for valid data
    assert len(result_request.results) > 0


def test_handle_none_data():
    handler = WellFormedFormulasHandler()
    result = handler.evaluate(
        PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=None),
        VerificationRequest(inputs="test"),
    )

    assert result is None


def test_handle_wrong_data_type():
    handler = WellFormedFormulasHandler()

    with pytest.raises(TypeError):
        handler.evaluate(
            PrimaryVerificationData(
                id="test", dtype=VerificationDType.argdown, data="not a graph"
            ),
            VerificationRequest(inputs="test"),
        )


def test_real_world_example():
    # Example from the logreco task test
    text = dedent("""
    ```argdown
    <Argument>: All persons must die.

    (P1) All humans are mortal. {formalization: "all x.(F(x) -> G(x))", declarations: {"F": "Human", "G": "Mortal"}}
    (P2) All persons are human. {formalization: "all x.(H(x) -> F(x))", declarations: {"H": "Person"}}
    -- {from: ["P1", "P2"]} --
    (C1) All persons are mortal. {formalization: "all x.(H(x) -> G(x))"}
    ```
    """)

    argdown_graph = parse_fenced_argdown(text)
    vdata = PrimaryVerificationData(
        id="real_world", dtype=VerificationDType.argdown, data=argdown_graph
    )

    request = VerificationRequest(inputs="test", verification_data=[vdata])

    # Test with the composite handler
    composite = LogRecoCompositeHandler()
    result_request = composite.handle(request)

    # Should have results for each handler
    assert len(result_request.results) == 6

    # WFF handler should be valid for this example
    wff_result = next(
        (
            r
            for r in result_request.results
            if r.verifier_id == "WellFormedFormulasHandler"
        ),
        None,
    )
    assert wff_result is not None
    assert wff_result.is_valid
    assert "all_expressions" in wff_result.details

    # Print results for debugging
    for result in result_request.results:
        print(f"{result.verifier_id}: {result.is_valid}")
        if not result.is_valid and result.message:
            print(f"  Message: {result.message}")

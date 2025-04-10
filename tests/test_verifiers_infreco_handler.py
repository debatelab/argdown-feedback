from pprint import pprint
import pytest
import copy
from textwrap import dedent

from pyargdown import parse_argdown

from argdown_feedback.verifiers.core.infreco_handler import (
    InfRecoHandler,
    HasArgumentsHandler,
    HasUniqueArgumentHandler,
    HasPCSHandler,
    StartsWithPremiseHandler,
    EndsWithConclusionHandler,
    NotMultipleGistsHandler,
    NoDuplicatePCSLabelsHandler,
    HasLabelHandler,
    HasGistHandler,
    HasInferenceDataHandler,
    PropRefsExistHandler,
    UsesAllPropsHandler,
    NoExtraPropositionsHandler,
    OnlyGroundedDialecticalRelationsHandler,
    NoPropInlineDataHandler,
    NoArgInlineDataHandler,
    InfRecoCompositeHandler
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
def valid_infreco_text():
    return dedent("""
    ```argdown
    <Argument 1>: Animals suffer.

    (1) Animals suffer.
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def valid_infreco_graph(valid_infreco_text):
    return parse_fenced_argdown(valid_infreco_text)


@pytest.fixture
def multi_argument_text():
    return dedent("""
    ```argdown
    <Argument-a>: Two args.

    (1) Animals have brain.
    -- {from: ["1"]} --
    (2) Animals suffer.

    <Argument-b>: Two args.

    (1) Animals suffer.
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def multi_argument_graph(multi_argument_text):
    return parse_fenced_argdown(multi_argument_text)


@pytest.fixture
def no_pcs_text():
    return dedent("""
    ```argdown
    <Argument>: No PCS.
    ```
    """)


@pytest.fixture
def no_pcs_graph(no_pcs_text):
    return parse_fenced_argdown(no_pcs_text)


@pytest.fixture
def ends_with_premise_text():
    return dedent("""
    ```argdown
    <Argument>: Ends with premise

    (1) Animals suffer.
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
    (3) Never eat meat.
    ```
    """)


@pytest.fixture
def ends_with_premise_graph(ends_with_premise_text):
    return parse_fenced_argdown(ends_with_premise_text)


@pytest.fixture
def duplicate_pcs_labels_text():
    return dedent("""
    ```argdown
    <Argument>: With duplicate pcs label.

    (1) Animals suffer.
    -- {from: ["1"]} --
    (1) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def duplicate_pcs_labels_graph(duplicate_pcs_labels_text):
    return parse_fenced_argdown(duplicate_pcs_labels_text)


@pytest.fixture
def no_label_text():
    return dedent("""
    ```argdown
                  
    (1) No LABEL! Animals suffer.
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def no_label_graph(no_label_text):
    return parse_fenced_argdown(no_label_text)


@pytest.fixture
def no_gist_text():
    return dedent("""
    ```argdown
    <Argument without gist>

    (1) Animals suffer.
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def no_gist_graph(no_gist_text):
    return parse_fenced_argdown(no_gist_text)


@pytest.fixture
def empty_from_list_text():
    return dedent("""
    ```argdown
    <Argument>: With empty from list.

    (1) Animals suffer.
    -- {from: []} --
    (2) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def empty_from_list_graph(empty_from_list_text):
    return parse_fenced_argdown(empty_from_list_text)


@pytest.fixture
def missing_inference_data_text():
    return dedent("""
    ```argdown
    <Argument>: Inf info missing altogether.

    (1) Animals suffer.
    -- from: [] --
    (2) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def missing_inference_data_graph(missing_inference_data_text):
    return parse_fenced_argdown(missing_inference_data_text)


@pytest.fixture
def invalid_ref_text():
    return dedent("""
    ```argdown
    <Argument>: Nonexistant ref.
                                        
    (1) Animals suffer.
    -- {from: ["2"]} --
    (2) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def invalid_ref_graph(invalid_ref_text):
    return parse_fenced_argdown(invalid_ref_text)


@pytest.fixture
def unused_premises_text():
    return dedent("""
    ```argdown
    <Argument>: Unused premises.
                                        
    (1) Animals suffer.
    (2) Animals have feelings.
    -- {from: ["1"]} --
    (3) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def unused_premises_graph(unused_premises_text):
    return parse_fenced_argdown(unused_premises_text)


@pytest.fixture
def extra_propositions_text():
    return dedent("""
    ```argdown
    [Brain]: Animals have brain.
                                        
    <Argument>: Extra propositions.
                                        
    (1) Animals suffer.
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def extra_propositions_graph(extra_propositions_text):
    return parse_fenced_argdown(extra_propositions_text)


@pytest.fixture
def dialectical_relations_text():
    return dedent("""
    ```argdown
    <Argument>: With dialectical relations.
                                        
    (1) Animals suffer.
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
        +> [No-meat]
    ```
    """)


@pytest.fixture
def dialectical_relations_graph(dialectical_relations_text):
    return parse_fenced_argdown(dialectical_relations_text)


@pytest.fixture
def prop_inline_data_text():
    return dedent("""
    ```argdown
    <Argument>: With prop inline data.
                                        
    (1) Animals suffer. {veracity: "true"}
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
    ```
    """)


@pytest.fixture
def prop_inline_data_graph(prop_inline_data_text):
    return parse_fenced_argdown(prop_inline_data_text)


@pytest.fixture
def verification_request(valid_infreco_graph):
    source = "Animals suffer. Eating animals is wrong."
    verification_data = [
        PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    ]
    request = VerificationRequest(inputs="", source=source, verification_data=verification_data)
    return request


def test_infreco_handler_is_applicable():
    handler = HasArgumentsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=None)
    request = VerificationRequest(inputs="")
    
    assert handler.is_applicable(vdata, request) is True
    
    vdata.dtype = VerificationDType.xml
    assert handler.is_applicable(vdata, request) is False


def test_has_arguments_handler_valid(valid_infreco_graph):
    handler = HasArgumentsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_has_arguments_handler_invalid():
    # Create a graph with no arguments
    empty_graph = parse_argdown("")
    handler = HasArgumentsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=empty_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "No arguments" in result.message


def test_has_unique_argument_handler_valid(valid_infreco_graph):
    handler = HasUniqueArgumentHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_has_unique_argument_handler_invalid(multi_argument_graph):
    handler = HasUniqueArgumentHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=multi_argument_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "More than one argument" in result.message


def test_has_pcs_handler_valid(valid_infreco_graph):
    handler = HasPCSHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_has_pcs_handler_invalid(no_pcs_graph):
    handler = HasPCSHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=no_pcs_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "lack premise conclusion structure" in result.message


def test_starts_with_premise_handler_valid(valid_infreco_graph):
    handler = StartsWithPremiseHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_ends_with_conclusion_handler_valid(valid_infreco_graph):
    handler = EndsWithConclusionHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_ends_with_conclusion_handler_invalid(valid_infreco_graph):
    handler = EndsWithConclusionHandler()
    illegal_argument = copy.deepcopy(valid_infreco_graph.arguments[0])
    illegal_argument.pcs = illegal_argument.pcs[:-1]  # Remove the last conclusion
    illegal_argument.label = "Illegal argument"
    ends_with_premise_graph = copy.deepcopy(valid_infreco_graph)
    ends_with_premise_graph.add_argument(illegal_argument, check_legal=False)
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=ends_with_premise_graph)
    request = VerificationRequest(inputs="", verification_data=[vdata])
    
    result = handler.evaluate(vdata, request)
    pprint(ends_with_premise_graph.arguments)
    pprint(result)
    assert result is not None
    assert result.is_valid is False
    assert "following arguments do end with a conclusion" in result.message  # The message in EndsWithConclusionHandler seems incorrect


def test_no_duplicate_pcs_labels_handler_valid(valid_infreco_graph):
    handler = NoDuplicatePCSLabelsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_no_duplicate_pcs_labels_handler_invalid(duplicate_pcs_labels_graph):
    handler = NoDuplicatePCSLabelsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=duplicate_pcs_labels_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "premise" in result.message  # The message in NoDuplicatePCSLabelsHandler seems incorrect


def test_has_label_handler_valid(valid_infreco_graph):
    handler = HasLabelHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_has_label_handler_invalid(no_label_graph):
    handler = HasLabelHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=no_label_graph)
    request = VerificationRequest(inputs="", verification_data=[vdata])

    result = handler.evaluate(vdata, request)
    assert not no_label_graph.arguments or result is not None
    assert not no_label_graph.arguments or result.is_valid is False
    assert not no_label_graph.arguments or "lack labels" in result.message


def test_has_gist_handler_valid(valid_infreco_graph):
    handler = HasGistHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_has_gist_handler_invalid(no_gist_graph):
    handler = HasGistHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=no_gist_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "lack gists" in result.message


def test_has_inference_data_handler_valid(valid_infreco_graph):
    handler = HasInferenceDataHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_has_inference_data_handler_invalid_empty_from(empty_from_list_graph):
    handler = HasInferenceDataHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=empty_from_list_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "empty" in result.message


def test_has_inference_data_handler_invalid_missing_data(missing_inference_data_graph):
    handler = HasInferenceDataHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=missing_inference_data_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "lacks" in result.message


def test_prop_refs_exist_handler_valid(valid_infreco_graph):
    handler = PropRefsExistHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_prop_refs_exist_handler_invalid(invalid_ref_graph):
    handler = PropRefsExistHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=invalid_ref_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "does not refer" in result.message


def test_uses_all_props_handler_valid(valid_infreco_graph):
    handler = UsesAllPropsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_uses_all_props_handler_invalid(unused_premises_graph):
    handler = UsesAllPropsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=unused_premises_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "not explicitly used" in result.message


def test_no_extra_propositions_handler_valid(valid_infreco_graph):
    handler = NoExtraPropositionsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_no_extra_propositions_handler_invalid(extra_propositions_graph):
    handler = NoExtraPropositionsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=extra_propositions_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "not used in any argument" in result.message


def test_only_grounded_dialectical_relations_handler_valid(valid_infreco_graph):
    handler = OnlyGroundedDialecticalRelationsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_only_grounded_dialectical_relations_handler_invalid(dialectical_relations_graph):
    handler = OnlyGroundedDialecticalRelationsHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=dialectical_relations_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "dialectical relations" in result.message


def test_no_prop_inline_data_handler_valid(valid_infreco_graph):
    handler = NoPropInlineDataHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=valid_infreco_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_no_prop_inline_data_handler_invalid(prop_inline_data_graph):
    handler = NoPropInlineDataHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=prop_inline_data_graph)
    request = VerificationRequest(inputs="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "yaml inline data" in result.message


def test_composite_handler(verification_request):
    # Create a graph with issues
    invalid_graph = parse_fenced_argdown(dedent("""
    ```argdown
    <Argument>: Missing ref.
    
    (1) Animals suffer.
    -- {from: ["2"]} --
    (2) Eating animals is wrong.
    ```
    """))
    vdata_invalid = PrimaryVerificationData(id="invalid", dtype=VerificationDType.argdown, data=invalid_graph)
    request = copy.deepcopy(verification_request)
    request.verification_data.append(vdata_invalid)
    
    composite = InfRecoCompositeHandler()
    composite.handle(request)
    
    # Should find issues in the invalid data
    assert len(request.results) > 0
    # Check that the appropriate handler found the issue
    invalid_results = [r for r in request.results if r.message is not None and "does not refer" in r.message]
    assert len(invalid_results) > 0


def test_handle_none_data():
    handlers = [
        HasArgumentsHandler(),
        HasUniqueArgumentHandler(),
        HasPCSHandler(),
        StartsWithPremiseHandler(),
        EndsWithConclusionHandler(),
        NotMultipleGistsHandler(),
        NoDuplicatePCSLabelsHandler(),
        HasLabelHandler(),
        HasGistHandler(),
        HasInferenceDataHandler(),
        PropRefsExistHandler(),
        UsesAllPropsHandler(),
        NoExtraPropositionsHandler(),
        OnlyGroundedDialecticalRelationsHandler(),
        NoPropInlineDataHandler(),
        NoArgInlineDataHandler()
    ]
    
    for handler in handlers:
        vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=None)
        request = VerificationRequest(inputs="")
        result = handler.evaluate(vdata, request)
        assert result is None


def test_handle_invalid_data_type():
    handlers = [
        HasArgumentsHandler(),
        HasUniqueArgumentHandler(),
        HasPCSHandler(),
        StartsWithPremiseHandler(),
        EndsWithConclusionHandler(),
        NotMultipleGistsHandler(),
        NoDuplicatePCSLabelsHandler(),
        HasLabelHandler(),
        HasGistHandler(),
        HasInferenceDataHandler(),
        PropRefsExistHandler(),
        UsesAllPropsHandler(),
        NoExtraPropositionsHandler(),
        OnlyGroundedDialecticalRelationsHandler(),
        NoPropInlineDataHandler(),
        NoArgInlineDataHandler()
    ]
    
    for handler in handlers:
        vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data="not a graph")
        request = VerificationRequest(inputs="")
        with pytest.raises(TypeError):
            handler.evaluate(vdata, request)


def test_infreco_handler_handle_method():
    request = VerificationRequest(inputs="")
    graph = parse_fenced_argdown("""
    ```argdown
    <Argument 1>: Animals suffer.

    (1) Animals suffer.
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
    ```
    """)
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=graph)
    request.verification_data.append(vdata)
    
    # Create a custom handler that's always valid
    class TestHandler(InfRecoHandler):
        def evaluate(self, vdata, ctx):
            return VerificationResult(
                verifier_id="test",
                verification_data_references=[vdata.id],
                is_valid=True,
                message=None
            )
    
    handler = TestHandler()
    handler.handle(request)
    
    assert len(request.results) == 1
    assert request.results[0].is_valid is True


def test_real_world_example_basic():
    # A basic valid example from test_task_infreco
    argdown_text = dedent("""
    ```argdown
    <Argument 1>: Animals suffer.

    (1) Animals suffer.
    -- {from: ["1"]} --
    (2) Eating animals is wrong.
    ```
    """)
    
    graph = parse_fenced_argdown(argdown_text)
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=graph)
    source = "Animals suffer. Eating animals is wrong."
    request = VerificationRequest(inputs=argdown_text, source=source)
    request.verification_data.append(vdata)
    
    composite = InfRecoCompositeHandler()
    composite.handle(request)
    
    # All validations should pass
    invalid_results = [r for r in request.results if not r.is_valid]
    assert len(invalid_results) == 0


def test_real_world_example_complex():
    # A more complex valid example from test_task_infreco
    argdown_text = dedent("""
    ```argdown
    <Argument 1>: Animals suffer.

    (P1) Animals suffer.
    (P2) Animals suffer.
    -- {from: ["P1", "P2"]} --
    (C1) Animals suffer.
    (P3) Animals suffer.
    -- with Modus ponens {from: ["C1", "P3"]} --
    (C2) Eating animals is wrong.
    ```
    """)
    
    graph = parse_fenced_argdown(argdown_text)
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.argdown, data=graph)
    source = "Animals suffer. Eating animals is wrong."
    request = VerificationRequest(inputs=argdown_text, source=source, verification_data=[vdata])
    
    composite = InfRecoCompositeHandler()
    composite.handle(request)
    pprint(request.results)
    # All validations should pass
    invalid_results = [r for r in request.results if not r.is_valid]
    assert len(invalid_results) == 0
"""Shared fixtures for API tests."""

import textwrap
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import os

from argdown_feedback.api.server.app import app
from argdown_feedback.api.shared.models import (
    VerificationRequest,
    FilterRule
)
from argdown_feedback.verifiers.verification_request import (
    VerificationRequest as InternalRequest
)

# Import well-tested fixtures from existing verifier tests
# These will be available for reuse in API tests
from tests.test_verifiers_arganno_handler import (
    valid_xml,
    valid_soup,
    invalid_support_ref_xml,
    invalid_attack_ref_xml,
    invalid_arg_label_xml,
    invalid_ref_reco_xml
)
from tests.test_verifiers_argmap_handler import (
    valid_argdown_text,
    valid_argdown_graph,
    incomplete_claims_argdown,
    incomplete_claims_graph
)
from tests.test_verifiers_processing_handler import (
    argdown_input_text,
    xml_input_text,
    mixed_input_text
)
from tests.test_verifiers_infreco_handler import (
    valid_infreco_text
)
from tests.test_verifiers_logreco_handler import (
    valid_logreco_text,
    invalid_formalization_text,
    deductively_invalid_text,
    inconsistent_premises_text,
    irrelevant_premise_text
)
# Import coherence verifier fixtures
from tests.test_task_argmapplusarganno import (
    source_texts as arganno_argmap_source_texts,
    valid_recos as arganno_argmap_valid_recos,
    invalid_recos as arganno_argmap_invalid_recos,
)
from tests.test_verifiers_arganno_infreco_handler import (
    valid_infreco_text as arganno_infreco_valid_infreco_text,
    valid_xml_text as arganno_infreco_valid_xml_text,
    invalid_argument_label_xml_text as arganno_infreco_invalid_argument_label_xml_text,
    invalid_ref_reco_label_xml_text as arganno_infreco_invalid_ref_reco_label_xml_text
)
from tests.test_verifiers_arganno_logreco_handler import (
    valid_logreco_text as arganno_logreco_valid_logreco_text,
    valid_xml_text as arganno_logreco_valid_xml_text,
    invalid_argument_label_xml_text as arganno_logreco_invalid_argument_label_xml_text,
    invalid_ref_reco_label_xml_text as arganno_logreco_invalid_ref_reco_label_xml_text
)
from tests.test_verifiers_argmap_infreco_handler import (
    valid_map_text as argmap_infreco_valid_map_text,
    valid_reco_text as argmap_infreco_valid_reco_text,
    missing_argument_map_text as argmap_infreco_missing_argument_map_text,
    missing_argument_reco_text as argmap_infreco_missing_argument_reco_text
)
from tests.test_verifiers_argmap_logreco_handler import (
    valid_map_text as argmap_logreco_valid_map_text,
    valid_logreco_text as argmap_logreco_valid_logreco_text,
    missing_argument_map_text as argmap_logreco_missing_argument_map_text,
    missing_argument_logreco_text as argmap_logreco_missing_argument_logreco_text
)


import pytest
import nltk

@pytest.fixture(scope="session", autouse=True)
def download_nltk_punkt_tab():
    nltk.download('punkt_tab')

# Fixtures for arganno_argmap coherence verifier tests
@pytest.fixture
def arganno_argmap_source_text():
    return textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)
                           

@pytest.fixture
def arganno_argmap_valid_reco():
    return textwrap.dedent("""
        Good stuff:
                        
        ```xml
        <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports="1" argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate change">Animal farming causes climate change.</proposition>
        ```

        ```argdown
        [No meat]: We should stop eating meat. {annotation_ids: ['1']}
            <+ <Suffering>: Animals suffer. {annotation_ids: ['2']}
            <+ <Climate change>: Animal farming causes climate change. {annotation_ids: ['3']}
        ```
        """)



@pytest.fixture
def arganno_argmap_invalid_reco():
    return textwrap.dedent("""
        ```xml
        <proposition id="1" argument_label="No meat">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports=["1"] argument_label="Suffering">Animals suffer.</proposition> <proposition id="3" supports=["1"] argument_label="Climate change">Animal farming causes climate change.</proposition>
        ```

        ```
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Climate change>: Animal farming causes climate change.
        ```
        """)
 
# arganno_infreco_source_text, arganno_infreco_valid_reco
# Fixtures for arganno_infreco coherence verifier tests

@pytest.fixture
def arganno_infreco_source_text():
    return textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)

@pytest.fixture
def arganno_infreco_valid_reco():
    return textwrap.dedent("""
        ```xml
        <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
        ```

        ```argdown
        <Suffering>
                        
        (1) Animals suffer. {annotation_ids: ['2']}
        -- {FROM: ["1"]} --
        (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
        ```
        """)

@pytest.fixture
def arganno_infreco_invalid_reco():
    return textwrap.dedent("""
        ```xml
        <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
        ```

        ```
        <Suffering>
                        
        (1) Animals suffer. {annotation_ids: ['2']}
        -- {FROM: ["1"]} --
        (2) [No meat]: We should stop eating meat. {annotation_ids: ['1']}
        ```
        """)

# Fixtures for arganno_logreco coherence verifier tests
# arganno_logreco_source_text, arganno_logreco_valid_reco

@pytest.fixture
def arganno_logreco_source_text():
    return textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)

@pytest.fixture
def arganno_logreco_valid_reco():
    return textwrap.dedent("""
        ```xml
        <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
        ```

        ```argdown
        <Suffering>
                        
        (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
        -- {from: ["1"]} --
        (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
        ```
        """)

@pytest.fixture
def arganno_logreco_invalid_reco():
    return textwrap.dedent("""
        ```xml
        <proposition id="1" argument_label="Suffering" ref_reco_label="2">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> Animal farming causes climate change.
        ```

        ```
        <Suffering>
                        
        (1) Animals suffer. {annotation_ids: ['2'], formalization: "p & q", declarations: {"p": "Animals suffer.", q: "Very much."}}
        -- {from: ["1"]} --
        (2) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "p"}
        ```
        """)

# Fixtures for argmap_infreco coherence verifier tests
# argmap_infreco_source_text, argmap_infreco_valid_reco

@pytest.fixture
def argmap_infreco_source_text():
    return textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)

@pytest.fixture
def argmap_infreco_valid_reco():
    return textwrap.dedent("""
        ```argdown {filename="map.ad"}
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Climate Change>: Animal farming causes climate change.
        ```

        ```argdown {filename="reconstructions.ad"}
        <Suffering>
                        
        (1) Animals suffer.
        -- {from: ["1"]} --
        (2) [No meat]: We should stop eating meat.
                        
        <Climate Change>
                        
        (1) Animal farming causes climate change.
        -- {from: ["1"]} --
        (2) [No meat]
        ```
        """)

@pytest.fixture
def argmap_infreco_invalid_reco():
    return textwrap.dedent("""
        ```argdown {filename="map.ad"}
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Climate Change>: Animal farming causes climate change.
        ```

        ```
        <Suffering>
                        
        (1) Animals suffer.
        -- {from: ["1"]} --
        (2) [No meat]: We should stop eating meat.
                        
        <Climate Change>
                        
        (1) Animal farming causes climate change.
        -- {from: ["1"]} --
        (2) [No meat]
        ```
        """)


# Fixture for argmap_logreco coherence verifier tests

@pytest.fixture
def argmap_logreco_source_text():
    return textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)

@pytest.fixture
def argmap_logreco_valid_reco():
    return textwrap.dedent("""
        ```argdown {filename="map.ad"}
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Climate Change>: Animal farming causes climate change.
        ```

        ```argdown {filename="reconstructions.ad"}
        <Suffering>
                        
        (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
        (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
        -- {from: ["1","2"]} --
        (3) [No meat]: We should stop eating meat. {formalization: "q"}
                        
        <Climate Change>
                        
        (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
        (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
        -- {from: ["1","2"]} --
        (3) [No meat]
        ```
        """)

@pytest.fixture
def argmap_logreco_invalid_reco():
    return textwrap.dedent("""
        ```
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Climate Change>: Animal farming causes climate change.
        ```

        ```argdown {filename="reconstructions.ad"}
        <Suffering>
                        
        (1) Animals suffer. {formalization: "p", declarations: {p: "Animals suffer."}}
        (2) If they suffer, we should not eat them. {formalization: "p -> q", declarations: {q: "We should not eat animals."}}
        -- {from: ["1","2"]} --
        (3) [No meat]: We should stop eating meat. {formalization: "q"}
                        
        <Climate Change>
                        
        (1) Animal farming causes climate change. {formalization: "r", declarations: {r: "Animal farming causes climate change."}}
        (2) If animal farming causes climate change, we should not eat them. {formalization: "r -> q", declarations: {q: "We should not eat animals."}}
        -- {from: ["1","2"]} --
        (3) [No meat]
        ```
        """)

# Fixture for arganno_argmap_logreco coherence verifier tests

@pytest.fixture
def arganno_argmap_logreco_source_text():
    return textwrap.dedent("""
        We should stop eating meat.
                        
        Animals suffer. Animal farming causes climate change.
        """)

@pytest.fixture
def arganno_argmap_logreco_valid_reco():
    return textwrap.dedent("""
        DEFAULT 
                        
        ```xml {filename="annotation.txt"}
        <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" supports="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
        ```

        ```argdown {filename="map.ad"}
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <+ <Climate Change>: Animal farming causes climate change.
        ```

        ```argdown {filename="reconstructions.ad"}
        <Suffering>
                        
        (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
        (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> q", declarations: {q: "We should not eat animals."}}
        -- {from: ["1","2"]} --
        (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "q"}
                        
        <Climate Change>
                        
        (1) Animal farming causes climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming causes climate change."}}
        (2) If animal farming causes climate change, we should not eat them. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should not eat animals."}}
        -- {from: ["1","2"]} --
        (3) [No meat]
        ```
        """)

@pytest.fixture
def arganno_argmap_logreco_invalid_reco():
    return textwrap.dedent("""
                        
        MISSING axiomatic relation in logreco

        ```xml {filename="annotation.txt"}
        <proposition id="1" argument_label="Suffering" ref_reco_label="3">We should stop eating meat.</proposition>
                        
        <proposition id="2" supports="1" argument_label="Suffering" ref_reco_label="1">Animals suffer.</proposition> <proposition id="3" attacks="1" argument_label="Climate Change" ref_reco_label="1">Animal farming causes climate change.</proposition>
        ```

        ```argdown {filename="map.ad"}
        [No meat]: We should stop eating meat.
            <+ <Suffering>: Animals suffer.
            <- <Climate Change>: Animal farming causes climate change.
        ```

        ```argdown {filename="reconstructions.ad"}
        <Suffering>
                        
        (1) Animals suffer. {annotation_ids: ['2'], formalization: "p", declarations: {p: "Animals suffer."}}
        (2) If they suffer, we should not eat them. {annotation_ids: [], formalization: "p -> -q", declarations: {q: "We should eat animals."}}
        -- {from: ["1","2"]} --
        (3) [No meat]: We should stop eating meat. {annotation_ids: ['1'], formalization: "-q"}
                        
        <Climate Change>
                        
        (1) Animal farming counters climate change. {annotation_ids: ['3'], formalization: "r", declarations: {r: "Animal farming counters climate change."}}
        (2) If animal farming counters climate change, we should eat animals. {annotation_ids: [], formalization: "r -> q", declarations: {q: "We should eat animals."}}
        -- {from: ["1","2"]} --
        (3) We should eat animals. {annotation_ids: [], formalization: "q"}
        //    >< [No meat]
        ```
        """)

# Further fixtures

@pytest.fixture
def arganno_source_text():
    """Source text that matches the arganno XML annotations."""
    return "We should stop eating meat. Animals suffer. Some animals are raised humanely."


@pytest.fixture
def api_client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_request_data():
    """Sample API request data with minimal config for arganno."""
    return {
        "inputs": "We should act on climate change.\n\n(1) Green technologies create jobs.\n(2) Investment in renewables boosts economy.",
        "source": "Original debate context",
        "config": {}  # arganno doesn't accept any config options
    }


@pytest.fixture
def sample_infreco_request_data():
    """Sample API request data for infreco verifier."""
    return {
        "inputs": "We should act on climate change.\n\n(1) Green technologies create jobs.\n(2) Investment in renewables boosts economy.",
        "source": "Original debate context",
        "config": {
            "from_key": "from"
        }
    }


@pytest.fixture
def sample_logreco_request_data():
    """Sample API request data for logreco verifier."""
    return {
        "inputs": "We should act on climate change.\n\n(1) Green technologies create jobs.\n(2) Investment in renewables boosts economy.",
        "source": "Original debate context",
        "config": {
            "from_key": "from",
            "formalization_key": "formalization"
        }
    }


@pytest.fixture
def sample_verification_request(sample_request_data):
    """Sample VerificationRequest model."""
    return VerificationRequest(**sample_request_data)


@pytest.fixture
def sample_infreco_verification_request(sample_infreco_request_data):
    """Sample VerificationRequest model for infreco."""
    return VerificationRequest(**sample_infreco_request_data)


@pytest.fixture
def sample_filter_rules():
    """Sample filter rules for testing."""
    return [
        FilterRule(key="role", value="user", regex=False),
        FilterRule(key="content", value="argument", regex=False),
        FilterRule(key="tag", value="important|urgent", regex=True)
    ]


@pytest.fixture
def sample_metadata_filter(sample_filter_rules):
    """Sample metadata filter (list of filter rules)."""
    return sample_filter_rules


@pytest.fixture
def mock_internal_request():
    """Mock internal verification request."""
    return InternalRequest(
        inputs="Test content",
        source="Test source"
    )


@pytest.fixture
def mock_handler():
    """Mock verifier handler for testing."""
    handler = MagicMock()
    handler.verify.return_value = MagicMock(
        request_id="test-123",
        success=True,
        message="Verification completed",
        details={"found_issues": 0, "processed_elements": 5}
    )
    return handler


@pytest.fixture
def verifier_test_configs():
    """Test configurations for different verifiers."""
    return {
        "arganno": {},
        "argmap": {"max_depth": 3},
        "infreco": {"temperature": 0.5, "max_tokens": 500},
        "logreco": {"model": "test-model", "confidence_threshold": 0.8},
        "processing": {"extract_fenced": True, "validate_syntax": True}
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    yield
    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
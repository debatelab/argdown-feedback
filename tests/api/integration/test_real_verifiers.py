"""Integration tests with actual verifier handlers."""

from pprint import pprint
from argdown_feedback.api.shared.models import VerificationRequest


class TestRealArgannoVerifier:
    """Test integration with actual arganno handlers."""
    
    def test_arganno_with_valid_xml(self, api_client, valid_xml, arganno_source_text):
        """Test arganno verifier with valid XML annotation from fixture."""
        # Wrap XML in fenced code blocks for proper detection
        xml_input = f"```xml\n{valid_xml.strip()}\n```"
        request = VerificationRequest(
            inputs=xml_input,
            source=arganno_source_text,
            config={}
        )
        
        response = api_client.post("/api/v1/verify/arganno", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "is_valid" in data
        assert "results" in data
        assert "executed_handlers" in data
        
        # Should have some handlers executed
        assert len(data["executed_handlers"]) > 0
        
        # EXPECT: Valid input should pass verification
        assert data["is_valid"], f"Expected valid XML to pass verification, but got: {data['results']}"
        
    def test_arganno_with_invalid_support_ref(self, api_client, invalid_support_ref_xml, arganno_source_text):
        """Test arganno verifier with invalid support reference XML from fixture."""
        # Wrap XML in fenced code blocks for proper detection
        xml_input = f"```xml\n{invalid_support_ref_xml.strip()}\n```"
        request = VerificationRequest(
            inputs=xml_input,
            source=arganno_source_text,
            config={}
        )
        
        response = api_client.post("/api/v1/verify/arganno", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "results" in data
        assert "executed_handlers" in data
        
        # EXPECT: Invalid support reference should fail verification
        assert not data["is_valid"], f"Expected invalid support reference to fail verification, but got: {data['results']}"
        
    def test_arganno_with_invalid_attack_ref(self, api_client, invalid_attack_ref_xml, arganno_source_text):
        """Test arganno verifier with invalid attack reference XML from fixture."""
        # Wrap XML in fenced code blocks for proper detection
        xml_input = f"```xml\n{invalid_attack_ref_xml.strip()}\n```"
        request = VerificationRequest(
            inputs=xml_input,
            source=arganno_source_text,
            config={}
        )
        
        response = api_client.post("/api/v1/verify/arganno", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "results" in data
        
        # EXPECT: Invalid attack reference should fail verification
        assert not data["is_valid"], f"Expected invalid attack reference to fail verification, but got: {data['results']}"


class TestRealArgmapVerifier:
    """Test integration with actual argmap handlers."""
    
    def test_argmap_with_valid_argdown(self, api_client, valid_argdown_text):
        """Test argmap verifier with valid argdown structure from fixture."""
        request = VerificationRequest(
            inputs=valid_argdown_text,
            source="Real argmap test with valid fixture",
            config={}
        )
        
        response = api_client.post("/api/v1/verify/argmap", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "argmap"
        assert "results" in data
        assert "executed_handlers" in data
        
        # EXPECT: Valid argdown should pass verification
        assert data["is_valid"], f"Expected valid argdown to pass verification, but got: {data['results']}"
        
    def test_argmap_with_incomplete_claims(self, api_client, incomplete_claims_argdown):
        """Test argmap verifier with incomplete claims argdown from fixture."""
        request = VerificationRequest(
            inputs=incomplete_claims_argdown,
            source="Real argmap test with incomplete claims fixture",
            config={}
        )
        
        response = api_client.post("/api/v1/verify/argmap", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "argmap"
        assert "results" in data
        
        # EXPECT: Incomplete claims should fail verification
        assert not data["is_valid"], f"Expected incomplete claims to fail verification, but got: {data['results']}"


class TestRealInfrecoVerifier:
    """Test integration with actual infreco handlers."""
    
    def test_infreco_with_valid_reconstruction(self, api_client, valid_infreco_text):
        """Test infreco verifier with valid informal reconstruction from fixture."""
        request = VerificationRequest(
            inputs=valid_infreco_text,
            source="Real infreco test with valid fixture",
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/infreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "infreco"
        assert "results" in data

        # EXPECT: Valid argdown should pass verification
        assert data["is_valid"], f"Expected valid argdown to pass verification, but got: {data['results']}"



class TestRealProcessingHandler:
    """Test integration with actual processing handlers."""
    
    def test_processing_with_argdown_input(self, api_client, argdown_input_text):
        """Test with argdown input text from fixture."""
        # Use arganno since processing verifier doesn't exist
        request = VerificationRequest(
            inputs=argdown_input_text,
            source="Real processing test with argdown fixture",
            config={}
        )
        
        response = api_client.post("/api/v1/verify/arganno", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "results" in data
        
    def test_processing_with_xml_input(self, api_client, xml_input_text):
        """Test with XML input text from fixture."""
        request = VerificationRequest(
            inputs=xml_input_text,
            source="Real processing test with XML fixture",
            config={}
        )
        
        response = api_client.post("/api/v1/verify/arganno", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "results" in data
        
    def test_processing_with_mixed_input(self, api_client, mixed_input_text):
        """Test with mixed input text from fixture."""
        request = VerificationRequest(
            inputs=mixed_input_text,
            source="Real processing test with mixed fixture",
            config={}
        )
        
        response = api_client.post("/api/v1/verify/arganno", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno"
        assert "results" in data


class TestRealLogrecoVerifier:
    """Test integration with actual logreco handlers."""
    
    def test_logreco_with_valid_reconstruction(self, api_client, valid_logreco_text):
        """Test logreco verifier with valid logical reconstruction from fixture."""
        request = VerificationRequest(
            inputs=valid_logreco_text,
            source="Logical reconstruction test with valid fixture",
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/logreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "logreco"
        assert "results" in data

        # EXPECT: Valid argdown should pass verification
        assert data["is_valid"], f"Expected valid argdown to pass verification, but got: {data['results']}"

    def test_logreco_with_invalid_formalization(self, api_client, invalid_formalization_text):
        """Test logreco verifier with invalid formalization from fixture."""
        request = VerificationRequest(
            inputs=invalid_formalization_text,
            source="Logical reconstruction test with invalid formalization fixture",
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/logreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "logreco"
        assert "results" in data

        # EXPECT: Invalid formalization should fail verification
        assert not data["is_valid"], f"Expected invalid formalization to fail verification, but got: {data['results']}"
        
    def test_logreco_with_deductively_invalid(self, api_client, deductively_invalid_text):
        """Test logreco verifier with deductively invalid argument from fixture."""
        request = VerificationRequest(
            inputs=deductively_invalid_text,
            source="Logical reconstruction test with deductively invalid fixture",
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/logreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "logreco"
        assert "results" in data

        # EXPECT: Deductively invalid argument should fail verification
        assert not data["is_valid"], f"Expected deductively invalid argument to fail verification, but got: {data['results']}"


class TestRealArgannoArgmapCoherenceVerifier:
    """Test integration with actual arganno_argmap coherence handlers."""
    

#    source_texts as arganno_argmap_source_texts,
#    valid_recos as arganno_argmap_valid_recos,
#    invalid_recos as arganno_argmap_invalid_recos,

    def test_arganno_argmap_with_valid_data(self, api_client, arganno_argmap_source_text, arganno_argmap_valid_reco):
        """Test arganno_argmap verifier with valid coherent data from fixtures."""
        request = VerificationRequest(
            inputs=arganno_argmap_valid_reco,
            source=arganno_argmap_source_text,
            config={}
        )
        
        response = api_client.post("/api/v1/verify/arganno_argmap", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno_argmap"
        assert "results" in data
        
        # EXPECT: Valid coherent data should pass verification
        pprint(data['results'])
        assert data["is_valid"], f"Expected valid coherent arganno/argmap data to pass verification, but got: {data['results']}"
        
    def test_arganno_argmap_with_invalid_reco(self, api_client, arganno_argmap_source_text, arganno_argmap_invalid_reco):
        """Test arganno_argmap verifier with invalid label reference from fixture."""
        request = VerificationRequest(
            inputs=arganno_argmap_invalid_reco,
            source=arganno_argmap_source_text,
            config={}
        )
        
        response = api_client.post("/api/v1/verify/arganno_argmap", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno_argmap"
        assert "results" in data
        
        # EXPECT: Invalid label reference should fail verification
        assert not data["is_valid"], f"Expected invalid label reference to fail verification, but got: {data['results']}"
        

class TestRealArgannoInfrecoCoherenceVerifier:
    """Test integration with actual arganno_infreco coherence handlers."""
    
    def test_arganno_infreco_with_valid_data(self, api_client, arganno_infreco_source_text, arganno_infreco_valid_reco):
        """Test arganno_infreco verifier with valid coherent data from fixtures."""
        request = VerificationRequest(
            inputs=arganno_infreco_valid_reco,
            source=arganno_infreco_source_text,
            config={"from_key": "FROM"}
        )
        
        response = api_client.post("/api/v1/verify/arganno_infreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno_infreco"
        assert "results" in data

        # EXPECT: Valid argdown should pass verification
        pprint(data['results'])
        assert data["is_valid"], f"Expected valid argdown to pass verification, but got: {data['results']}"

        
    def test_arganno_infreco_with_invalid_reco(self, api_client, arganno_infreco_source_text, arganno_infreco_invalid_reco):
        """Test arganno_infreco verifier with invalid argument label from fixture."""
        request = VerificationRequest(
            inputs=arganno_infreco_invalid_reco,
            source=arganno_infreco_source_text,
            config={"from_key": "FROM"}
        )
        
        response = api_client.post("/api/v1/verify/arganno_infreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno_infreco"
        assert "results" in data

        # EXPECT: Invalid argument label should fail verification
        assert not data["is_valid"], f"Expected invalid argument label to fail verification, but got: {data['results']}"
        

class TestRealArgannoLogrecoCoherenceVerifier:
    """Test integration with actual arganno_logreco coherence handlers."""
    
    def test_arganno_logreco_with_valid_data(self, api_client, arganno_logreco_source_text, arganno_logreco_valid_reco):
        """Test arganno_logreco verifier with valid coherent data from fixtures."""
        request = VerificationRequest(
            inputs=arganno_logreco_valid_reco,
            source=arganno_logreco_source_text,
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/arganno_logreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno_logreco"
        assert "results" in data

        # EXPECT: Valid argdown should pass verification
        assert data["is_valid"], f"Expected valid argdown to pass verification, but got: {data['results']}"

        
    def test_arganno_logreco_with_invalid_reco(self, api_client, arganno_logreco_source_text, arganno_logreco_invalid_reco):
        """Test arganno_logreco verifier with invalid argument label from fixture."""
        request = VerificationRequest(
            inputs=arganno_logreco_invalid_reco,
            source=arganno_logreco_source_text,
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/arganno_logreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno_logreco"
        assert "results" in data

        # EXPECT: Invalid argument label should fail verification
        assert not data["is_valid"], f"Expected invalid argument label to fail verification, but got: {data['results']}"


class TestRealArgmapInfrecoCoherenceVerifier:
    """Test integration with actual argmap_infreco coherence handlers."""
    
    def test_argmap_infreco_with_valid_data(self, api_client, argmap_infreco_source_text, argmap_infreco_valid_reco):
        """Test argmap_infreco verifier with valid coherent data from fixtures."""
        request = VerificationRequest(
            inputs=argmap_infreco_valid_reco,
            source=argmap_infreco_source_text,
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/argmap_infreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "argmap_infreco"
        assert "results" in data

        # EXPECT: Valid argdown should pass verification
        assert data["is_valid"], f"Expected valid argdown to pass verification, but got: {data['results']}"


    def test_argmap_infreco_invalid_reco(self, api_client, argmap_infreco_source_text, argmap_infreco_invalid_reco):
        """Test argmap_infreco verifier with missing argument in map from fixture."""
        request = VerificationRequest(
            inputs=argmap_infreco_invalid_reco,
            source=argmap_infreco_source_text,
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/argmap_infreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "argmap_infreco"
        assert "results" in data

        # EXPECT: Missing argument in map should fail verification
        assert not data["is_valid"], f"Expected missing argument in map to fail verification, but got: {data['results']}"
        

class TestRealArgmapLogrecoCoherenceVerifier:
    """Test integration with actual argmap_logreco coherence handlers."""
    
    def test_argmap_logreco_with_valid_data(self, api_client, argmap_logreco_source_text, argmap_logreco_valid_reco):
        """Test argmap_logreco verifier with valid coherent data from fixtures."""
        request = VerificationRequest(
            inputs=argmap_logreco_valid_reco,
            source=argmap_logreco_valid_reco,
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/argmap_logreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "argmap_logreco"
        assert "results" in data

        # EXPECT: Valid argdown should pass verification
        assert data["is_valid"], f"Expected valid argdown to pass verification, but got: {data['results']}"


    def test_argmap_logreco_invalid_reco(self, api_client, argmap_logreco_source_text, argmap_logreco_invalid_reco):
        """Test argmap_logreco verifier with missing argument in map from fixture."""
        request = VerificationRequest(
            inputs=argmap_logreco_invalid_reco,
            source=argmap_logreco_source_text,
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/argmap_logreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "argmap_logreco"
        assert "results" in data

        # EXPECT: Missing argument in map should fail verification
        assert not data["is_valid"], f"Expected missing argument in map to fail verification, but got: {data['results']}"

        
class TestRealArgannoArgmapLogrecoCoherenceVerifier:
    """Test integration with actual arganno_argmap_logreco triple coherence handlers."""
    
    def test_arganno_argmap_logreco_valid_reco(self, api_client, arganno_argmap_logreco_source_text, arganno_argmap_logreco_valid_reco):
        """Test arganno_argmap_logreco verifier with combined coherent data from fixtures."""
        request = VerificationRequest(
            inputs=arganno_argmap_logreco_valid_reco,
            source=arganno_argmap_logreco_source_text,
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/arganno_argmap_logreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno_argmap_logreco"
        assert "results" in data

        # EXPECT: Valid argdown should pass verification
        assert data["is_valid"], f"Expected valid argdown to pass verification, but got: {data['results']}"


    def test_arganno_argmap_logreco_invalid_reco(self, api_client, arganno_argmap_logreco_source_text, arganno_argmap_logreco_invalid_reco):
        """Test arganno_argmap_logreco verifier with combined coherent data from fixtures."""
        request = VerificationRequest(
            inputs=arganno_argmap_logreco_invalid_reco,
            source=arganno_argmap_logreco_source_text,
            config={"from_key": "from"}
        )
        
        response = api_client.post("/api/v1/verify/arganno_argmap_logreco", json=request.model_dump())
        assert response.status_code == 200
        
        data = response.json()
        assert data["verifier"] == "arganno_argmap_logreco"
        assert "results" in data

        # EXPECT: Valid argdown should pass verification
        assert not data["is_valid"], f"Expected valid argdown to pass verification, but got: {data['results']}"

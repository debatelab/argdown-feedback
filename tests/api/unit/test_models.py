"""Unit tests for API shared models."""

from argdown_feedback.api.shared.models import (
    VerificationRequest,
    VerificationResponse,
    VerificationData,
    VerificationResult,
    VerifierInfo,
    VerifierConfigOption,
    FilterRule
)


class TestFilterRule:
    """Tests for FilterRule dataclass."""
    
    def test_create_simple_filter_rule(self):
        """Test creating a simple filter rule."""
        rule = FilterRule(key="role", value="user", regex=False)
        assert rule.key == "role"
        assert rule.value == "user"
        assert rule.regex is False
    
    def test_create_regex_filter_rule(self):
        """Test creating a regex filter rule."""
        rule = FilterRule(key="content", value="argument|premise", regex=True)
        assert rule.key == "content"
        assert rule.value == "argument|premise"
        assert rule.regex is True
    
    def test_default_regex_false(self):
        """Test that regex defaults to False."""
        rule = FilterRule(key="tag", value="important")
        assert rule.regex is False


class TestVerificationRequest:
    """Tests for VerificationRequest model."""
    
    def test_minimal_request(self):
        """Test creating a minimal verification request."""
        request = VerificationRequest(
            inputs="Test content",
            source=None,
            config=None
        )
        assert request.inputs == "Test content"
        assert request.source is None
        assert request.config is None
    
    def test_full_request(self):
        """Test creating a full verification request."""
        config = {"temperature": 0.7, "max_tokens": 1000}
        request = VerificationRequest(
            inputs="Test content",
            source="Source text",
            config=config
        )
        assert request.inputs == "Test content"
        assert request.source == "Source text"
        assert request.config == config
    
    def test_config_can_be_dict(self):
        """Test that config accepts arbitrary dictionary."""
        config = {
            "model": "gpt-4",
            "temperature": 0.5,
            "custom_param": "value"
        }
        request = VerificationRequest(
            inputs="Test",
            source=None,
            config=config
        )
        assert request.config == config


class TestVerificationResponse:
    """Tests for VerificationResponse model."""
    
    def test_minimal_response(self):
        """Test creating a minimal verification response."""
        response = VerificationResponse(
            verifier="test_verifier",
            is_valid=True,
            verification_data=[],
            results=[],
            executed_handlers=[],
            processing_time_ms=42.5
        )
        assert response.verifier == "test_verifier"
        assert response.is_valid is True
        assert response.verification_data == []
        assert response.results == []
        assert response.executed_handlers == []
        assert response.processing_time_ms == 42.5
    
    def test_full_response(self):
        """Test creating a full verification response."""
        verification_data = [
            VerificationData(
                id="test_1",
                dtype="argdown",
                code_snippet="```argdown\nTest\n```",
                metadata={"version": "v3"}
            )
        ]
        results = [
            VerificationResult(
                verifier_id="TestHandler",
                verification_data_references=["test_1"],
                is_valid=True,
                message="Test passed",
                details={"score": 0.95}
            )
        ]
        response = VerificationResponse(
            verifier="complex_verifier",
            is_valid=True,
            verification_data=verification_data,
            results=results,
            executed_handlers=["TestHandler"],
            processing_time_ms=123.45
        )
        assert response.verifier == "complex_verifier"
        assert response.is_valid is True
        assert len(response.verification_data) == 1
        assert len(response.results) == 1
        assert response.executed_handlers == ["TestHandler"]


class TestVerifierInfo:
    """Tests for VerifierInfo model."""
    
    def test_minimal_verifier_info(self):
        """Test creating minimal verifier info."""
        info = VerifierInfo(
            name="test_verifier",
            description="A test verifier",
            input_types=["argdown"],
            allowed_filter_roles=["test"],
            is_coherence_verifier=False
        )
        assert info.name == "test_verifier"
        assert info.description == "A test verifier"
        assert info.input_types == ["argdown"]
        assert info.allowed_filter_roles == ["test"]
        assert info.config_options == []
        assert info.is_coherence_verifier is False
    
    def test_full_verifier_info(self):
        """Test creating full verifier info."""
        config_options = [
            VerifierConfigOption(
                name="temperature",
                type="number",
                default=0.7,
                description="Temperature for generation",
                required=False
            )
        ]
        info = VerifierInfo(
            name="advanced_verifier",
            description="An advanced verifier",
            input_types=["argdown", "xml"],
            allowed_filter_roles=["user", "assistant"],
            config_options=config_options,
            is_coherence_verifier=True
        )
        assert info.name == "advanced_verifier"
        assert info.description == "An advanced verifier"
        assert info.input_types == ["argdown", "xml"]
        assert info.allowed_filter_roles == ["user", "assistant"]
        assert len(info.config_options) == 1
        assert info.is_coherence_verifier is True
    
    def test_json_serialization(self):
        """Test that VerifierInfo can be serialized to JSON."""
        info = VerifierInfo(
            name="test",
            description="Test verifier",
            input_types=["argdown"],
            allowed_filter_roles=["user"],
            is_coherence_verifier=False
        )
        json_data = info.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["name"] == "test"
        assert json_data["description"] == "Test verifier"


class TestModelIntegration:
    """Integration tests for model interactions."""
    
    def test_request_response_compatibility(self):
        """Test that request and response models work together."""
        # Create a request
        request = VerificationRequest(
            inputs="Test content",
            source=None,
            config={"temperature": 0.7}
        )
        
        # Simulate processing and create response
        response = VerificationResponse(
            verifier="test_verifier",
            is_valid=True,
            verification_data=[],
            results=[],
            executed_handlers=["TestHandler"],
            processing_time_ms=42.5
        )
        
        assert request.config is not None
        assert request.config["temperature"] == 0.7
        assert response.verifier == "test_verifier"
    
    def test_verifier_info_with_realistic_config(self):
        """Test VerifierInfo with realistic config options."""        
        config_options = [
            VerifierConfigOption(
                name="temperature",
                type="number",
                default=0.7,
                description="Generation temperature",
                required=False
            ),
            VerifierConfigOption(
                name="max_tokens",
                type="integer",
                default=1000,
                description="Maximum tokens to generate",
                required=False
            )
        ]
        
        info = VerifierInfo(
            name="llm_verifier",
            description="LLM-based verification",
            input_types=["argdown"],
            allowed_filter_roles=["infreco"],
            config_options=config_options,
            is_coherence_verifier=False
        )
        
        assert len(info.config_options) == 2
        assert info.config_options[0].name == "temperature"
        assert info.config_options[0].default == 0.7
        assert info.config_options[1].name == "max_tokens"
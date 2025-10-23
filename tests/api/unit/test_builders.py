"""Unit tests for API client builders."""

from argdown_feedback.api.client.builders import (
    create_arganno_request,
    create_argmap_request,
    create_infreco_request,
    create_logreco_request,
    create_argmap_infreco_request,
    create_arganno_argmap_request,
    create_arganno_infreco_request,
    create_arganno_logreco_request,
    create_argmap_logreco_request,
    create_arganno_argmap_logreco_request,
)
from argdown_feedback.api.shared.models import VerificationRequest


class TestArgannoRequestBuilder:
    """Tests for arganno request builder."""
    
    def test_minimal_arganno_request(self):
        """Test building minimal arganno request."""
        builder = create_arganno_request(
            inputs="Test XML content",
            source=None
        )
        request = builder.build()
        
        assert isinstance(request, VerificationRequest)
        assert request.inputs == "Test XML content"
        assert request.source is None
        assert request.config is None
    
    def test_full_arganno_request(self):
        """Test building full arganno request with config."""
        builder = create_arganno_request(
            inputs="Test XML content",
            source="Source text"
        )
        # Note: ArgannoRequestBuilder doesn't have config_option method,
        # so we test the basic functionality
        request = builder.build()
        
        assert request.inputs == "Test XML content"
        assert request.source == "Source text"
        assert request.config is None


class TestArgmapRequestBuilder:
    """Tests for argmap request builder."""
    
    def test_minimal_argmap_request(self):
        """Test building minimal argmap request."""
        builder = create_argmap_request(
            inputs="Test Argdown content",
            source=None
        )
        request = builder.build()
        
        assert isinstance(request, VerificationRequest)
        assert request.inputs == "Test Argdown content"
        assert request.source is None
        assert request.config is None
    
    def test_argmap_request_with_config(self):
        """Test building argmap request with config."""
        builder = create_argmap_request(
            inputs="Test content",
            source=None
        )
        # Note: ArgmapRequestBuilder doesn't have config_option method
        request = builder.build()
        
        assert request.inputs == "Test content"
        assert request.config is None


class TestInfrecoRequestBuilder:
    """Tests for infreco request builder."""
    
    def test_minimal_infreco_request(self):
        """Test building minimal infreco request."""
        builder = create_infreco_request(
            inputs="Test Argdown content",
            source=None
        )
        request = builder.build()
        
        assert isinstance(request, VerificationRequest)
        assert request.inputs == "Test Argdown content"
        assert request.source is None
        assert request.config is None
    
    def test_infreco_request_with_config(self):
        """Test building infreco request with config."""
        builder = create_infreco_request(
            inputs="Test content",
            source="Source text"
        )
        builder.config_option("from_key", "from")
        request = builder.build()
        
        assert request.config == {"from_key": "from"}
        assert request.source == "Source text"


class TestLogrecoRequestBuilder:
    """Tests for logreco request builder."""
    
    def test_minimal_logreco_request(self):
        """Test building minimal logreco request."""
        builder = create_logreco_request(
            inputs="Test content",
            source=None
        )
        request = builder.build()
        
        assert isinstance(request, VerificationRequest)
        assert request.inputs == "Test content"
        assert request.source is None
        assert request.config is None
    
    def test_logreco_request_with_config(self):
        """Test building logreco request with config."""
        builder = create_logreco_request(
            inputs="Test content",
            source=None
        )
        builder.config_option("from_key", "from")
        builder.config_option("formalization_key", "formalization")
        request = builder.build()
        
        assert request.config == {
            "from_key": "from",
            "formalization_key": "formalization"
        }


class TestCoherenceRequestBuilders:
    """Tests for coherence verification request builders."""
    
    def test_argmap_infreco_request(self):
        """Test building argmap+infreco coherence request."""
        builder = create_argmap_infreco_request(
            inputs="Test content",
            source="Source text"
        )
        request = builder.build()
        
        assert isinstance(request, VerificationRequest)
        assert request.inputs == "Test content"
        assert request.source == "Source text"
        assert request.config is None
    
    def test_arganno_argmap_request(self):
        """Test building arganno+argmap coherence request."""
        builder = create_arganno_argmap_request(
            inputs="Test content",
            source=None
        )
        request = builder.build()
        
        assert request.inputs == "Test content"
        assert request.source is None
        assert request.config is None
    
    def test_arganno_infreco_request(self):
        """Test building arganno+infreco coherence request."""
        builder = create_arganno_infreco_request(
            inputs="Test content",
            source="Source text"
        )
        request = builder.build()
        
        assert request.inputs == "Test content"
        assert request.source == "Source text"
    
    def test_arganno_logreco_request(self):
        """Test building arganno+logreco coherence request."""
        builder = create_arganno_logreco_request(
            inputs="Test content",
            source=None
        )
        request = builder.build()
        
        assert request.inputs == "Test content"
        assert request.source is None
    
    def test_argmap_logreco_request(self):
        """Test building argmap+logreco coherence request."""
        builder = create_argmap_logreco_request(
            inputs="Test content",
            source="Source text"
        )
        request = builder.build()
        
        assert request.inputs == "Test content"
        assert request.source == "Source text"
    
    def test_triple_coherence_request(self):
        """Test building arganno+argmap+logreco coherence request."""
        builder = create_arganno_argmap_logreco_request(
            inputs="Test content",
            source="Source text"
        )
        request = builder.build()
        
        assert request.inputs == "Test content"
        assert request.source == "Source text"
        assert request.config is None


class TestBuilderIntegration:
    """Integration tests for request builders."""
    
    def test_all_builders_return_verification_request(self):
        """Test that all builders return VerificationRequest instances."""
        builders_and_inputs = [
            (create_arganno_request, "XML content"),
            (create_argmap_request, "Argdown content"),
            (create_infreco_request, "Argdown content"),
            (create_logreco_request, "Content"),
            (create_argmap_infreco_request, "Content"),
            (create_arganno_argmap_request, "Content"),
            (create_arganno_infreco_request, "Content"),
            (create_arganno_logreco_request, "Content"),
            (create_argmap_logreco_request, "Content"),
            (create_arganno_argmap_logreco_request, "Content")
        ]
        
        for builder_func, test_input in builders_and_inputs:
            builder = builder_func(inputs=test_input, source=None)
            request = builder.build()
            assert isinstance(request, VerificationRequest)
            assert request.inputs == test_input
    
    def test_builder_config_handling(self):
        """Test that builders handle config consistently."""
        # Test infreco with config options
        infreco_builder = create_infreco_request(inputs="Test", source=None)
        infreco_builder.config_option("from_key", "from")
        request_with_config = infreco_builder.build()
        assert request_with_config.config == {"from_key": "from"}
        
        # Test logreco with config options
        logreco_builder = create_logreco_request(inputs="Test", source=None)
        logreco_builder.config_option("from_key", "from")
        logreco_builder.config_option("formalization_key", "formalization")
        request_with_config = logreco_builder.build()
        assert request_with_config.config == {
            "from_key": "from",
            "formalization_key": "formalization"
        }
        
        # Test builders without config methods
        builders_without_config = [
            create_arganno_request,
            create_argmap_request,
            create_argmap_infreco_request,
            create_arganno_argmap_request,
            create_arganno_infreco_request,
            create_arganno_logreco_request,
            create_argmap_logreco_request,
            create_arganno_argmap_logreco_request
        ]
        
        for builder_func in builders_without_config:
            builder = builder_func(inputs="Test", source=None)
            request = builder.build()
            assert request.config is None
    
    def test_builder_source_handling(self):
        """Test that all builders handle source parameter correctly."""
        test_source = "Test source content"
        
        builders = [
            create_arganno_request,
            create_argmap_request,
            create_infreco_request,
            create_logreco_request,
            create_argmap_infreco_request,
            create_arganno_argmap_request,
            create_arganno_infreco_request,
            create_arganno_logreco_request,
            create_argmap_logreco_request,
            create_arganno_argmap_logreco_request
        ]
        
        for builder_func in builders:
            # Test with source
            builder_with_source = builder_func(inputs="Test", source=test_source)
            request_with_source = builder_with_source.build()
            assert request_with_source.source == test_source
            
            # Test without source (None)
            builder_without_source = builder_func(inputs="Test", source=None)
            request_without_source = builder_without_source.build()
            assert request_without_source.source is None
    
    def test_request_serialization(self):
        """Test that built requests can be serialized to JSON."""
        builder = create_infreco_request(
            inputs="Test content",
            source="Source text"
        )
        builder.config_option("from_key", "from")
        request = builder.build()
        
        # Should be able to serialize to dict
        request_dict = request.model_dump()
        assert isinstance(request_dict, dict)
        assert request_dict["inputs"] == "Test content"
        assert request_dict["source"] == "Source text"
        assert request_dict["config"]["from_key"] == "from"
        
        # Should be able to recreate from dict
        recreated_request = VerificationRequest(**request_dict)
        assert recreated_request.inputs == request.inputs
        assert recreated_request.source == request.source
        assert recreated_request.config == request.config
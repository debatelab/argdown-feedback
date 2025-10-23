"""Unit tests for API filtering system."""

from argdown_feedback.api.shared.filtering import (
    FilterBuilder,
    ArgannoFilterBuilder,
    ArgmapFilterBuilder,
    InfrecoFilterBuilder,
    LogrecoFilterBuilder,
    FilterRule
)


class TestFilterRule:
    """Tests for FilterRule dataclass."""
    
    def test_create_filter_rule(self):
        """Test creating a filter rule."""
        rule = FilterRule(key="role", value="user", regex=False)
        assert rule.key == "role"
        assert rule.value == "user"
        assert rule.regex is False
    
    def test_filter_rule_defaults(self):
        """Test filter rule default values."""
        rule = FilterRule(key="tag", value="important")
        assert rule.key == "tag"
        assert rule.value == "important"
        assert rule.regex is False


class TestFilterBuilder:
    """Tests for base FilterBuilder class."""
    
    def test_create_empty_builder(self):
        """Test creating an empty filter builder."""
        builder = FilterBuilder()
        filters = builder.build()
        assert filters == {}
    
    def test_add_simple_rule(self):
        """Test adding a simple filter rule."""
        builder = FilterBuilder()
        builder.add("user", "role", "participant")
        
        filters = builder.build()
        assert "user" in filters
        assert filters["user"]["role"] == "participant"
    
    def test_add_regex_rule(self):
        """Test adding a regex filter rule."""
        builder = FilterBuilder()
        builder.add("content", "pattern", "important|urgent", regex=True)
        
        filters = builder.build()
        assert "content" in filters
        assert isinstance(filters["content"], list)
        assert filters["content"][0]["key"] == "pattern"
        assert filters["content"][0]["value"] == "important|urgent"
        assert filters["content"][0]["regex"] is True
    
    def test_multiple_rules_same_role(self):
        """Test adding multiple filter rules for same role."""
        builder = FilterBuilder()
        builder.add("user", "type", "message")
        builder.add("user", "priority", "high")
        
        filters = builder.build()
        assert "user" in filters
        assert isinstance(filters["user"], list)
        assert len(filters["user"]) == 2
    
    def test_fluent_interface(self):
        """Test that add returns self for chaining."""
        builder = FilterBuilder()
        result = builder.add("user", "role", "test").add("assistant", "type", "response")
        
        assert result is builder
        filters = builder.build()
        assert len(filters) == 2


class TestInfrecoFilterBuilder:
    """Tests for InfrecoFilterBuilder."""
    
    def test_create_infreco_builder(self):
        """Test creating an InfrecoFilterBuilder."""
        builder = InfrecoFilterBuilder()
        filters = builder.build()
        assert filters == {}
    
    def test_add_infreco_role(self):
        """Test adding infreco role filter."""
        builder = InfrecoFilterBuilder()
        builder.add("infreco", "version", "v3")
        
        filters = builder.build()
        assert "infreco" in filters
        assert filters["infreco"]["version"] == "v3"
    
    def test_infreco_fluent_interface(self):
        """Test fluent interface for InfrecoFilterBuilder."""
        builder = InfrecoFilterBuilder()
        result = builder.add("infreco", "version", "v3")
        
        assert result is builder
        assert isinstance(result, InfrecoFilterBuilder)


class TestArgannoFilterBuilder:
    """Tests for ArgannoFilterBuilder."""
    
    def test_create_arganno_builder(self):
        """Test creating an ArgannoFilterBuilder."""
        builder = ArgannoFilterBuilder()
        filters = builder.build()
        assert filters == {}
    
    def test_add_arganno_role(self):
        """Test adding arganno role filter."""
        builder = ArgannoFilterBuilder()
        builder.add("arganno", "format", "xml")
        
        filters = builder.build()
        assert "arganno" in filters
        assert filters["arganno"]["format"] == "xml"


class TestArgmapFilterBuilder:
    """Tests for ArgmapFilterBuilder."""
    
    def test_create_argmap_builder(self):
        """Test creating an ArgmapFilterBuilder."""
        builder = ArgmapFilterBuilder()
        filters = builder.build()
        assert filters == {}
    
    def test_add_argmap_role(self):
        """Test adding argmap role filter."""
        builder = ArgmapFilterBuilder()
        builder.add("argmap", "complexity", "simple")
        
        filters = builder.build()
        assert "argmap" in filters
        assert filters["argmap"]["complexity"] == "simple"


class TestLogRecoFilterBuilder:
    """Tests for LogRecoFilterBuilder."""
    
    def test_create_logreco_builder(self):
        """Test creating a LogRecoFilterBuilder."""
        builder = LogrecoFilterBuilder()
        filters = builder.build()
        assert filters == {}
    
    def test_add_logreco_role(self):
        """Test adding logreco role filter."""
        builder = LogrecoFilterBuilder()
        builder.add("logreco", "type", "formal")
        
        filters = builder.build()
        assert "logreco" in filters
        assert filters["logreco"]["type"] == "formal"


class TestFilterBuilderIntegration:
    """Integration tests for filter builders."""
    
    def test_different_builders_independence(self):
        """Test that different builders are independent."""
        infreco_builder = InfrecoFilterBuilder()
        argmap_builder = ArgmapFilterBuilder()
        
        infreco_builder.add("infreco", "version", "v3")
        argmap_builder.add("argmap", "version", "v2")
        
        infreco_filters = infreco_builder.build()
        argmap_filters = argmap_builder.build()
        
        assert "infreco" in infreco_filters
        assert "argmap" in argmap_filters
        assert infreco_filters["infreco"]["version"] == "v3"
        assert argmap_filters["argmap"]["version"] == "v2"
    
    def test_complex_filter_combinations(self):
        """Test complex filter rule combinations."""
        builder = FilterBuilder()
        builder.add("infreco", "version", "v3") \
               .add("infreco", "complexity", "high") \
               .add("argmap", "format", "standard")
        
        filters = builder.build()
        assert "infreco" in filters
        assert "argmap" in filters
        
        # infreco should have multiple rules (list format)
        assert isinstance(filters["infreco"], list)
        assert len(filters["infreco"]) == 2
        
        # argmap should have single rule (dict format)
        assert isinstance(filters["argmap"], dict)
        assert filters["argmap"]["format"] == "standard"
    
    def test_simple_vs_advanced_format(self):
        """Test that single exact matches use simple format."""
        builder = FilterBuilder()
        
        # Single exact match -> simple format
        builder.add("user", "role", "participant")
        
        # Multiple rules -> advanced format
        builder.add("admin", "level", "high")
        builder.add("admin", "access", "full")
        
        filters = builder.build()
        
        # Single rule: simple dict format
        assert isinstance(filters["user"], dict)
        assert filters["user"]["role"] == "participant"
        
        # Multiple rules: list format
        assert isinstance(filters["admin"], list)
        assert len(filters["admin"]) == 2
"""
Updated unit tests for verifier registry to remove VerifierSpec-related assumptions.
"""

import pytest

from argdown_feedback.api.server.services import (
    verifier_registry,
)
from argdown_feedback.api.shared.models import VerifierInfo
from argdown_feedback.api.shared.exceptions import VerifierNotFoundError


class TestVerifierRegistry:
    """Tests for verifier registry functionality."""

    def test_registry_contains_expected_verifiers(self):
        """Test that registry contains expected verifier types."""
        verifier_names = verifier_registry.list_verifiers()
        expected_verifiers = [
            "arganno",
            "argmap",
            "infreco",
            "logreco",
        ]

        for verifier in expected_verifiers:
            assert verifier in verifier_names, f"Expected verifier {verifier} not found in registry"

    def test_registry_values_are_verifier_specs(self):
        """Test that registry contains VerifierInfo instances."""
        verifier_names = verifier_registry.list_verifiers()

        for name in verifier_names:
            info = verifier_registry.get_verifier_info(name)
            assert isinstance(info, VerifierInfo)
            assert info.name == name
            assert hasattr(info, 'input_types')
            assert hasattr(info, 'allowed_filter_roles')
            assert hasattr(info, 'config_options')
            assert hasattr(info, 'is_coherence_verifier')


class TestCreateVerifierHandler:
    """Tests for verifier handler creation."""

    def test_create_arganno_handler(self):
        """Test creating an arganno handler."""
        handler = verifier_registry.create_handler("arganno")
        assert handler is not None
        assert hasattr(handler, 'process')

    def test_create_argmap_handler(self):
        """Test creating an argmap handler."""
        handler = verifier_registry.create_handler("argmap")
        assert handler is not None
        assert hasattr(handler, 'process')

    def test_create_infreco_handler(self):
        """Test creating an infreco handler."""
        handler = verifier_registry.create_handler("infreco")
        assert handler is not None
        assert hasattr(handler, 'process')

    def test_create_logreco_handler(self):
        """Test creating a logreco handler."""
        handler = verifier_registry.create_handler("logreco")
        assert handler is not None
        assert hasattr(handler, 'process')

    def test_create_handler_with_config(self):
        """Test creating handler with configuration."""
        config = {"from_key": "FROM"}
        # Note: Most handlers don't accept config parameters directly
        handler = verifier_registry.create_handler("infreco", **config)
        assert handler is not None
        assert hasattr(handler, 'process')
        # Config is used internally but may not be stored as attributes

    def test_create_handler_invalid_name(self):
        """Test creating handler with invalid name raises error."""
        with pytest.raises(VerifierNotFoundError):
            verifier_registry.create_handler("invalid_verifier")


class TestGetVerifierInfo:
    """Tests for getting verifier information."""

    def test_get_arganno_info(self):
        """Test getting arganno verifier info."""
        info = verifier_registry.get_verifier_info("arganno")
        assert isinstance(info, VerifierInfo)
        assert info.name == "arganno"
        assert "xml" in info.input_types
        assert "arganno" in info.allowed_filter_roles

    def test_get_argmap_info(self):
        """Test getting argmap verifier info."""
        info = verifier_registry.get_verifier_info("argmap")
        assert isinstance(info, VerifierInfo)
        assert info.name == "argmap"
        assert "argdown" in info.input_types
        assert "argmap" in info.allowed_filter_roles

    def test_get_infreco_info(self):
        """Test getting infreco verifier info."""
        info = verifier_registry.get_verifier_info("infreco")
        assert isinstance(info, VerifierInfo)
        assert info.name == "infreco"
        assert "argdown" in info.input_types
        assert "infreco" in info.allowed_filter_roles

    def test_get_logreco_info(self):
        """Test getting logreco verifier info."""
        info = verifier_registry.get_verifier_info("logreco")
        assert isinstance(info, VerifierInfo)
        assert info.name == "logreco"
        assert len(info.input_types) > 0
        assert "logreco" in info.allowed_filter_roles

    def test_get_info_invalid_name(self):
        """Test getting info for invalid verifier name."""
        with pytest.raises(VerifierNotFoundError):
            verifier_registry.get_verifier_info("invalid_verifier")


class TestListVerifiers:
    """Tests for listing verifiers."""

    def test_list_all_verifiers(self):
        """Test listing all verifiers."""
        verifiers = verifier_registry.get_all_verifiers_info()
        assert len(verifiers.core) > 0

        # Check that we have the expected core verifiers
        core_names = [v.name for v in verifiers.core]
        expected_core = ["arganno", "argmap", "infreco", "logreco"]

        for expected in expected_core:
            assert expected in core_names

    def test_verifier_info_structure(self):
        """Test that verifier info has proper structure."""
        verifiers = verifier_registry.get_all_verifiers_info()

        for verifier in verifiers.core + verifiers.coherence:
            assert isinstance(verifier, VerifierInfo)
            assert verifier.name
            assert verifier.description
            assert isinstance(verifier.input_types, list)
            assert isinstance(verifier.allowed_filter_roles, list)
            assert isinstance(verifier.config_options, list)
            assert isinstance(verifier.is_coherence_verifier, bool)

    def test_core_vs_coherence_classification(self):
        """Test that verifiers are properly classified as core vs coherence."""
        verifiers = verifier_registry.get_all_verifiers_info()

        # Core verifiers should not be marked as coherence verifiers
        for verifier in verifiers.core:
            assert not verifier.is_coherence_verifier

        # Coherence verifiers should be marked as such
        for verifier in verifiers.coherence:
            assert verifier.is_coherence_verifier


class TestRegistryIntegration:
    """Integration tests for registry functionality."""

    def test_create_and_verify_all_handlers(self):
        """Test that all registered handlers can be created and have process method."""
        verifier_names = verifier_registry.list_verifiers()

        for name in verifier_names:
            handler = verifier_registry.create_handler(name)
            assert handler is not None
            assert hasattr(handler, 'process')
            assert callable(getattr(handler, 'process'))

    def test_info_matches_registry(self):
        """Test that get_verifier_info matches registry data."""
        verifier_names = verifier_registry.list_verifiers()

        for name in verifier_names:
            info = verifier_registry.get_verifier_info(name)

            assert info.name == name
            assert len(info.input_types) > 0  # Ensure there is at least one input type
            assert len(info.allowed_filter_roles) > 0  # Ensure there is at least one allowed filter role

    def test_list_verifiers_completeness(self):
        """Test that get_all_verifiers_info includes all registry entries."""
        verifier_names = set(verifier_registry.list_verifiers())
        verifiers_list = verifier_registry.get_all_verifiers_info()

        all_listed_verifiers = (
            verifiers_list.core +
            verifiers_list.coherence +
            verifiers_list.content_check
        )
        listed_names = {v.name for v in all_listed_verifiers}

        assert listed_names == verifier_names, f"Missing verifiers: {verifier_names - listed_names}"
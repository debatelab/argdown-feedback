import copy
from pprint import pprint
import pytest
from bs4 import BeautifulSoup
import textwrap


from argdown_feedback.verifiers.core.arganno_handler import (
    ArgannoHandler, 
    SourceTextIntegrityHandler,
    NestedPropositionHandler,
    PropositionIdPresenceHandler,
    PropositionIdUniquenessHandler,
    SupportReferenceValidityHandler,
    AttackReferenceValidityHandler,
    AttributeValidityHandler,
    ElementValidityHandler,
    ArgumentLabelValidityHandler,
    RefRecoLabelValidityHandler,
    ArgannoCompositeHandler
)
from argdown_feedback.verifiers.verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
    VerificationResult
)


@pytest.fixture
def valid_xml():
    return """
    <proposition id="1">We should stop eating meat.</proposition>
    <proposition id="2" supports="1">Animals suffer.</proposition>
    <proposition id="3" attacks="2">Some animals are raised humanely.</proposition>
    """


@pytest.fixture
def valid_soup(valid_xml):
    return BeautifulSoup(valid_xml, 'html.parser')


@pytest.fixture
def nested_props_xml():
    return """
    <proposition id="1">
        We should <proposition id="2">stop eating meat</proposition>.
    </proposition>
    """


@pytest.fixture
def missing_id_xml():
    return """
    <proposition id="1">We should stop eating meat.</proposition>
    <proposition>Animals suffer.</proposition>
    """


@pytest.fixture
def duplicate_id_xml():
    return """
    <proposition id="1">We should stop eating meat.</proposition>
    <proposition id="1">Animals suffer.</proposition>
    """


@pytest.fixture
def invalid_support_ref_xml():
    return """
    <proposition id="1">We should stop eating meat.</proposition>
    <proposition id="2" supports="3">Animals suffer.</proposition>
    """


@pytest.fixture
def invalid_attack_ref_xml():
    return """
    <proposition id="1">We should stop eating meat.</proposition>
    <proposition id="2" attacks="3">Animals suffer.</proposition>
    """


@pytest.fixture
def unknown_attr_xml():
    return """
    <proposition id="1" from="somewhere">We should stop eating meat.</proposition>
    """


@pytest.fixture
def unknown_element_xml():
    return """
    <proposition id="1">We should stop eating meat.</proposition>
    <claim id="2">Animals suffer.</claim>
    """


@pytest.fixture
def invalid_arg_label_xml():
    return """
    <proposition id="1" argument_label="invalid">We should stop eating meat.</proposition>
    """


@pytest.fixture
def invalid_ref_reco_xml():
    return """
    <proposition id="1" ref_reco_label="invalid">We should stop eating meat.</proposition>
    """


@pytest.fixture
def verification_request(valid_soup):
    source = "We should stop eating meat. Animals suffer. Some animals are raised humanely."
    verification_data = [
        PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    ]
    request = VerificationRequest(inputs="", source=source, verification_data=verification_data)
    return request


def test_arganno_handler_is_applicable():
    handler = SourceTextIntegrityHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=None)
    request = VerificationRequest(inputs="", source="")
    
    assert handler.is_applicable(vdata, request) is True
    
    vdata.dtype = VerificationDType.argdown
    assert handler.is_applicable(vdata, request) is False


def test_source_text_integrity_handler_valid(valid_soup):
    handler = SourceTextIntegrityHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="We should stop eating meat. Animals suffer. Some animals are raised humanely.")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_source_text_integrity_handler_invalid(valid_soup):
    handler = SourceTextIntegrityHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="Different source text that doesn't match.")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "was altered" in result.message



def test_source_text_integrity_handler_roughly_equal():
    handler = SourceTextIntegrityHandler()
    str1 = "We should stop eating meat."
    str2 = "We should stop eating  meat."
    str3 = "We should stop\n\n eating meat.  "
    str4 = "We      should stop eating meat. \n"
    str5 = "We should stop  \teating meat."

    str01 = "We should stoop eating meat."

    assert handler._are_roughly_equal(str1, str2) is True
    assert handler._are_roughly_equal(str1, str3) is True
    assert handler._are_roughly_equal(str1, str4) is True
    assert handler._are_roughly_equal(str1, str5) is True
    assert handler._are_roughly_equal(str1, str01) is False

    str10 = (
        "We should stop eating meat. Animals suffer. Some animals are raised humanely. "
        "We should stop eating meat. Animals suffer. Some animals are raised humanely. "
        "We should stop eating meat. Animals suffer. Some animals are raised humanely. "
        "We should stop eating meat. Animals suffer. Some animals are raised humanely. "
        "We should stop eating meat. Animals suffer. Some animals are raised humanely. "
    )
    str11 = (
        "I should stop eating meat. Animals suffer. Some animals are raised humanely. \n"
        "We should stop eating meat. Animals suffer. Some animals are raised humanely.\n"
        "We should stop eating meat. Animals suffer. Some animals are raised humanely.\n"
        "We should stop eating meat. Animals suffer. Some animals are raised humanely.\n"
        "We should stop eating meat. Animals suffer. Some animals are raised humanely.\n"
    )
    str010 = (
        "We should stop eating meat. Rabbits suffer. Some animals are raised humanely. "
        "We should stop eating meat. Animals suffer. Some animals are raised humanely. "
        "We should stop eating meat. Animals suffer. Some animals are raised humanely. "
        "We should stop eating meat. Animals suffer. Some animals are raised humanely."
    )
    assert handler._are_roughly_equal(str10, str11) is True
    assert handler._are_roughly_equal(str10, str010) is False


def test_nested_proposition_handler_valid(valid_soup):
    handler = NestedPropositionHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_nested_proposition_handler_invalid(nested_props_xml):
    handler = NestedPropositionHandler()
    soup = BeautifulSoup(nested_props_xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Nested annotations" in result.message


def test_proposition_id_presence_handler_valid(valid_soup):
    handler = PropositionIdPresenceHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_proposition_id_presence_handler_invalid(missing_id_xml):
    handler = PropositionIdPresenceHandler()
    soup = BeautifulSoup(missing_id_xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Missing id" in result.message


def test_proposition_id_uniqueness_handler_valid(valid_soup):
    handler = PropositionIdUniquenessHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_proposition_id_uniqueness_handler_invalid(duplicate_id_xml):
    handler = PropositionIdUniquenessHandler()
    soup = BeautifulSoup(duplicate_id_xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Duplicate ids" in result.message


def test_support_reference_validity_handler_valid(valid_soup):
    handler = SupportReferenceValidityHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_support_reference_validity_handler_invalid(invalid_support_ref_xml):
    handler = SupportReferenceValidityHandler()
    soup = BeautifulSoup(invalid_support_ref_xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "does not exist" in result.message


def test_attack_reference_validity_handler_valid(valid_soup):
    handler = AttackReferenceValidityHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_attack_reference_validity_handler_invalid(invalid_attack_ref_xml):
    handler = AttackReferenceValidityHandler()
    soup = BeautifulSoup(invalid_attack_ref_xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "does not exist" in result.message


def test_attribute_validity_handler_valid(valid_soup):
    handler = AttributeValidityHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_attribute_validity_handler_invalid(unknown_attr_xml):
    handler = AttributeValidityHandler()
    soup = BeautifulSoup(unknown_attr_xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Unknown attribute" in result.message


def test_element_validity_handler_valid(valid_soup):
    handler = ElementValidityHandler()
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_element_validity_handler_invalid(unknown_element_xml):
    handler = ElementValidityHandler()
    soup = BeautifulSoup(unknown_element_xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Unknown element" in result.message


def test_argument_label_validity_handler_valid(valid_soup):
    handler = ArgumentLabelValidityHandler(legal_labels=["A", "B", "C"])
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_argument_label_validity_handler_invalid(invalid_arg_label_xml):
    handler = ArgumentLabelValidityHandler(legal_labels=["A", "B", "C"])
    soup = BeautifulSoup(invalid_arg_label_xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Illegal argument label" in result.message


def test_ref_reco_label_validity_handler_valid(valid_soup):
    handler = RefRecoLabelValidityHandler(legal_labels=["A", "B", "C"])
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=valid_soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is True
    assert result.message is None


def test_ref_reco_label_validity_handler_invalid(invalid_ref_reco_xml):
    handler = RefRecoLabelValidityHandler(legal_labels=["A", "B", "C"])
    soup = BeautifulSoup(invalid_ref_reco_xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is not None
    assert result.is_valid is False
    assert "Illegal ref_reco label" in result.message


def test_composite_handler(verification_request):
    # First create a handler with an invalid component
    soup_nested = BeautifulSoup("<proposition id='1'><proposition id='2'>Nested</proposition></proposition>", 'html.parser')
    vdata_nested = PrimaryVerificationData(id="nested", dtype=VerificationDType.xml, data=soup_nested)
    request = copy.deepcopy(verification_request)
    request.verification_data.append(vdata_nested)
    
    composite = ArgannoCompositeHandler()
    composite.process(request)
    
    # Should find issues in the nested proposition data
    assert len(request.results) > 0
    print(request.results)
    # Check that the nested proposition handler found the issue
    nested_results = [r for r in request.results if r.message is not None and "Nested" in r.message]
    assert len(nested_results) > 0


def test_handle_none_data():
    handlers = [
        SourceTextIntegrityHandler(),
        NestedPropositionHandler(),
        PropositionIdPresenceHandler(),
        PropositionIdUniquenessHandler(),
        SupportReferenceValidityHandler(),
        AttackReferenceValidityHandler(),
        AttributeValidityHandler(),
        ElementValidityHandler(),
        ArgumentLabelValidityHandler(legal_labels=["A"]),
        RefRecoLabelValidityHandler(legal_labels=["A"])
    ]
    
    for handler in handlers:
        vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=None)
        request = VerificationRequest(inputs="", source="")
        result = handler.evaluate(vdata, request)
        assert result is None


def test_handle_invalid_data_type():
    handlers = [
        SourceTextIntegrityHandler(),
        NestedPropositionHandler(),
        PropositionIdPresenceHandler(),
        PropositionIdUniquenessHandler(),
        SupportReferenceValidityHandler(),
        AttackReferenceValidityHandler(),
        AttributeValidityHandler(),
        ElementValidityHandler(),
        ArgumentLabelValidityHandler(legal_labels=["A"]),
        RefRecoLabelValidityHandler(legal_labels=["A"])
    ]
    
    for handler in handlers:
        vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data="not a soup")
        request = VerificationRequest(inputs="", source="")
        with pytest.raises(ValueError):
            handler.evaluate(vdata, request)


def test_no_legal_labels_skips_validation():
    # For argument label handler
    handler = ArgumentLabelValidityHandler()
    soup = BeautifulSoup("<proposition id='1' argument_label='X'>Test</proposition>", 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is None
    
    # For ref_reco label handler
    handler = RefRecoLabelValidityHandler()
    soup = BeautifulSoup("<proposition id='1' ref_reco_label='Y'>Test</proposition>", 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request = VerificationRequest(inputs="", source="")
    
    result = handler.evaluate(vdata, request)
    assert result is None


def test_arganno_handler_handle_method():
    request = VerificationRequest(inputs="", source="")
    soup = BeautifulSoup("<proposition id='1'>Test</proposition>", 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    request.verification_data.append(vdata)
    
    # Create a custom handler that's always valid
    class TestHandler(ArgannoHandler):
        def evaluate(self, vdata, ctx):
            return VerificationResult(
                verifier_id="test",
                verification_data_references=[vdata.id],
                is_valid=True,
                message=None
            )
    
    handler = TestHandler()
    handler.process(request)
    
    assert len(request.results) == 1
    assert request.results[0].is_valid is True


def test_real_world_example_from_test_task_arganno():
    xml = textwrap.dedent("""
    <proposition id="1">We should stop eating meat.</proposition>
                    
    <proposition id="2" supports="1">Animals suffer.</proposition> 
    <proposition id="3" supports="2">Animal farming causes climate change.</proposition>
    """)
    
    soup = BeautifulSoup(xml, 'html.parser')
    vdata = PrimaryVerificationData(id="test", dtype=VerificationDType.xml, data=soup)
    source = (
        "We should stop eating meat. Animals suffer. Animal farming causes climate change."
    )
    request = VerificationRequest(inputs="", source=source, verification_data=[vdata])    
    composite = ArgannoCompositeHandler()
    composite.process(request)
    pprint(request.results)    
    # All validations should pass
    invalid_results = [r for r in request.results if not r.is_valid]
    assert len(invalid_results) == 0
"""
Registry for mapping verifier names to their handler classes and specifications.
"""

from typing import Dict, List, Type, Any, Tuple
from dataclasses import dataclass

from argdown_feedback.verifiers.processing_handler import ArgdownParser, DefaultProcessingHandler, FencedCodeBlockExtractor, XMLParser
from argdown_feedback.verifiers.verification_request import PrimaryVerificationData, VDFilter, VerificationDType

from ....verifiers.base import BaseHandler, CompositeHandler
from ....verifiers.core.arganno_handler import ArgannoCompositeHandler
from ....verifiers.core.argmap_handler import ArgMapCompositeHandler
from ....verifiers.core.infreco_handler import InfRecoCompositeHandler
from ....verifiers.core.logreco_handler import LogRecoCompositeHandler
from ....verifiers.core.content_check_handler import HasAnnotationsHandler, HasArgdownHandler
from ....verifiers.coherence.arganno_argmap_handler import ArgannoArgmapCoherenceHandler
from ....verifiers.coherence.arganno_infreco_handler import ArgannoInfrecoCoherenceHandler
from ....verifiers.coherence.arganno_logreco_handler import ArgannoLogrecoCoherenceHandler
from ....verifiers.coherence.argmap_infreco_handler import ArgmapInfrecoCoherenceHandler
from ....verifiers.coherence.argmap_logreco_handler import ArgmapLogrecoCoherenceHandler
from ....verifiers.coherence.arganno_argmap_logreco_handler import ArgannoArgmapLogrecoCoherenceHandler

from ...shared.models import VerifierInfo, VerifierConfigOption, VerifiersList
from ...shared.exceptions import VerifierNotFoundError


@dataclass
class VerifierSpec:
    """Specification for a verifier including its handler class and metadata."""
    name: str
    handler_class: Type[BaseHandler]
    description: str
    input_types: List[str]
    allowed_filter_roles: List[str]
    config_options: List[VerifierConfigOption]
    preprocessing_handler_classes: List[Tuple[Type[BaseHandler], Dict[str, Any]]]
    default_filters: Dict[str, List[Dict[str, Any]]] = {}
    is_coherence_verifier: bool = False


class VerifierRegistry:
    """Registry for managing available verifiers and their specifications."""
    
    def __init__(self):
        self._verifiers: Dict[str, VerifierSpec] = {}
        self._initialize_verifiers()
    
    def _initialize_verifiers(self):
        """Initialize the registry with all available verifiers."""
        
        # Core verifiers
        self.register(VerifierSpec(
            name="arganno",
            handler_class=ArgannoCompositeHandler,
            description="Validates argumentative annotations in XML format",
            input_types=["xml"],
            allowed_filter_roles=["arganno"],
            preprocessing_handler_classes=[
                (FencedCodeBlockExtractor, {"name": "FencedCodeBlockExtractor"}),
                (XMLParser, {"name": "XMLAnnotationParser"}),
                (HasAnnotationsHandler, {})
            ],
            config_options=[]  # ArgannoCompositeHandler doesn't accept any config options
        ))
        
        self.register(VerifierSpec(
            name="argmap",
            handler_class=ArgMapCompositeHandler,
            description="Validates argument maps in Argdown format",
            input_types=["argdown"],
            allowed_filter_roles=["argmap"],
            preprocessing_handler_classes=[
                (FencedCodeBlockExtractor, {"name": "FencedCodeBlockExtractor"}),
                (ArgdownParser, {"name": "ArgdownParser"}),
                (HasArgdownHandler, {})
            ],
            config_options=[]
        ))
        
        self.register(VerifierSpec(
            name="infreco",
            handler_class=InfRecoCompositeHandler,
            description="Validates informal argument reconstruction in Argdown format",
            input_types=["argdown"],
            allowed_filter_roles=["infreco"],
            preprocessing_handler_classes=[
                (FencedCodeBlockExtractor, {"name": "FencedCodeBlockExtractor"}),
                (ArgdownParser, {"name": "ArgdownParser"}),
                (HasArgdownHandler, {})
            ],
            config_options=[
                VerifierConfigOption(
                    name="from_key",
                    type="string",
                    default="from",
                    description="Key used for inference information in arguments",
                    required=False
                )
            ]
        ))

        self.register(VerifierSpec(
            name="logreco",
            handler_class=LogRecoCompositeHandler,
            description="Validates logical argument reconstruction in Argdown format",
            input_types=["argdown"],
            allowed_filter_roles=["logreco"],
            preprocessing_handler_classes=[
                (FencedCodeBlockExtractor, {"name": "FencedCodeBlockExtractor"}),
                (ArgdownParser, {"name": "ArgdownParser"}),
                (InfRecoCompositeHandler, {}),
                (HasArgdownHandler, {})
            ],
            config_options=[
                VerifierConfigOption(
                    name="from_key",
                    type="string",
                    default="from",
                    description="Key used for inference information in arguments",
                    required=False
                ),
                VerifierConfigOption(
                    name="formalization_key",
                    type="string",
                    default="formalization",
                    description="Key used for formalization information",
                    required=False
                ),
                VerifierConfigOption(
                    name="declarations_key",
                    type="string",
                    default="declarations",
                    description="Key used for declarations information",
                    required=False
                )
            ]
        ))
                
        # Coherence verifiers
        self.register(VerifierSpec(
            name="arganno_argmap",
            handler_class=ArgannoArgmapCoherenceHandler,
            description="Checks coherence between argumentative annotations and argument maps",
            input_types=["xml", "argdown"],
            allowed_filter_roles=["arganno", "argmap"],
            preprocessing_handler_classes=[
                (DefaultProcessingHandler, {}),
                (HasAnnotationsHandler, {}),
                (HasArgdownHandler, {}),
                (ArgannoCompositeHandler, {}),
                (ArgMapCompositeHandler, {})
            ],
            config_options=[],
            is_coherence_verifier=True
        ))
        
        self.register(VerifierSpec(
            name="arganno_infreco",
            handler_class=ArgannoInfrecoCoherenceHandler,
            description="Checks coherence between argumentative annotations and inference reconstruction",
            input_types=["xml", "argdown"],
            allowed_filter_roles=["arganno", "infreco"],
            preprocessing_handler_classes=[
                (DefaultProcessingHandler, {}),
                (HasAnnotationsHandler, {}),
                (HasArgdownHandler, {}),
            ],
            config_options=[
                VerifierConfigOption(
                    name="from_key",
                    type="string",
                    default="from",
                    description="Key used for inference information in arguments",
                    required=False
                )
            ],
            is_coherence_verifier=True
        ))
        
        self.register(VerifierSpec(
            name="arganno_logreco",
            handler_class=ArgannoLogrecoCoherenceHandler,
            description="Checks coherence between argumentative annotations and logical reconstruction",
            input_types=["xml", "argdown"],
            allowed_filter_roles=["arganno", "logreco"],
            preprocessing_handler_classes=[
                (DefaultProcessingHandler, {}),
                (HasAnnotationsHandler, {}),
                (HasArgdownHandler, {}),
            ],
            config_options=[
                VerifierConfigOption(
                    name="from_key",
                    type="string",
                    default="from",
                    description="Key used for inference information in arguments",
                    required=False
                )
            ],
            is_coherence_verifier=True
        ))
        
        
        self.register(VerifierSpec(
            name="argmap_infreco",
            handler_class=ArgmapInfrecoCoherenceHandler,
            description="Checks coherence between argument maps and inference reconstruction",
            input_types=["argdown"],
            allowed_filter_roles=["argmap", "infreco"],
            default_filters={
                "argmap": [{"key": "filename", "value": "map.*", "regex": True}],
                "infreco": [{"key": "filename", "value": "reconstructions.*", "regex": True}]
            },
            preprocessing_handler_classes=[
                (FencedCodeBlockExtractor, {"name": "FencedCodeBlockExtractor"}),
                (ArgdownParser, {"name": "ArgdownParser"}),
                (HasArgdownHandler, {"name": "HasArgdownHandler.map", "filter_roles": ["argmap"]}),
                (HasArgdownHandler, {"name": "HasArgdownHandler.reco", "filter_roles": ["infreco"]}),
            ],
            config_options=[
                VerifierConfigOption(
                    name="from_key",
                    type="string",
                    default="from",
                    description="Key used for inference information in arguments",
                    required=False
                ),
            ],
            is_coherence_verifier=True
        ))

        self.register(VerifierSpec(
            name="argmap_logreco",
            handler_class=ArgmapLogrecoCoherenceHandler,
            description="Checks coherence between argument maps and logical reconstruction",
            input_types=["argdown"],
            allowed_filter_roles=["argmap", "logreco"],
            default_filters={
                "argmap": [{"key": "filename", "value": "map.*", "regex": True}],
                "logreco": [{"key": "filename", "value": "reconstructions.*", "regex": True}]
            },
            preprocessing_handler_classes=[
                (FencedCodeBlockExtractor, {"name": "FencedCodeBlockExtractor"}),
                (ArgdownParser, {"name": "ArgdownParser"}),
                (HasArgdownHandler, {"name": "HasArgdownHandler.map", "filter_roles": ["argmap"]}),
                (HasArgdownHandler, {"name": "HasArgdownHandler.reco", "filter_roles": ["logreco"]}),
            ],
            config_options=[
                VerifierConfigOption(
                    name="from_key",
                    type="string",
                    default="from",
                    description="Key used for inference information in arguments",
                    required=False
                )
            ],
            is_coherence_verifier=True
        ))
        

        self.register(VerifierSpec(
            name="arganno_argmap_logreco",
            handler_class=ArgannoArgmapLogrecoCoherenceHandler,
            description="Checks coherence between annotations, argument maps, and logical reconstruction",
            input_types=["xml", "argdown"],
            allowed_filter_roles=["arganno", "argmap", "logreco"],
            default_filters={
                "argmap": [{"key": "filename", "value": "map.*", "regex": True}],
                "logreco": [{"key": "filename", "value": "reconstructions.*", "regex": True}]
            },
            preprocessing_handler_classes=[
                (DefaultProcessingHandler, {}),
                (HasAnnotationsHandler, {"filter_roles": ["arganno"]}),
                (HasArgdownHandler, {"name": "HasArgdownHandler.map", "filter_roles": ["argmap"]}),
                (HasArgdownHandler, {"name": "HasArgdownHandler.reco", "filter_roles": ["logreco"]}),
            ],
            config_options=[
                VerifierConfigOption(
                    name="from_key",
                    type="string",
                    default="from",
                    description="Key used for inference information in arguments",
                    required=False
                )
            ],
            is_coherence_verifier=True
        ))
    
    def register(self, spec: VerifierSpec) -> None:
        """Register a verifier specification."""
        self._verifiers[spec.name] = spec
    
    def get_spec(self, name: str) -> VerifierSpec:
        """Get verifier specification by name."""
        if name not in self._verifiers:
            raise VerifierNotFoundError(name, list(self._verifiers.keys()))
        return self._verifiers[name]
    
    def get_handler_class(self, name: str) -> Type[BaseHandler]:
        """Get handler class for a verifier."""
        return self.get_spec(name).handler_class
    
    def create_handler(self, name: str, filters: dict[str, list[dict[str,str]]], **kwargs) -> BaseHandler:
        """Create a handler instance for a verifier with full preprocessing pipeline."""
        spec = self.get_spec(name)

        # create VDFilters
        if any(key not in spec.allowed_filter_roles for key in filters.keys()):
            invalid_roles = [key for key in filters.keys() if key not in spec.allowed_filter_roles]
            raise ValueError(f"Invalid filter roles for verifier '{name}': {invalid_roles}. Allowed roles: {spec.allowed_filter_roles}")

        vd_filters = self.create_vd_filters({**spec.default_filters, **filters})
        
        # Create preprocessing handlers with their specific configurations
        preprocessing_handlers = self.get_preprocessing_handlers(name, vd_filters=vd_filters, **kwargs)
        
        # Create main handler
        kwargs = self.assign_filters(kwargs, vd_filters)
        main_handler = spec.handler_class(**kwargs)
        
        # Return composite handler with full pipeline
        return CompositeHandler(
            name=f"{name}_full_pipeline",
            handlers=preprocessing_handlers + [main_handler]
        )
    
    def create_vd_filters(self, filters: dict[str, list[dict[str,str]]]) -> dict[str, VDFilter]:
        """Create VDFilter functions for given filter roles and criteria."""

        vd_filters: dict[str, VDFilter] = {}

        for role, criteria_list in filters.items():            
            def make_filter(criteria_list: list[dict[str,str]]) -> VDFilter:
                def vd_filter(vdata: PrimaryVerificationData) -> bool:
                    if vdata.dtype == VerificationDType.xml and role != "arganno":
                        return False
                    elif vdata.dtype != VerificationDType.xml and role == "arganno":
                        return False
                    for criteria in criteria_list:
                        if vdata.metadata is None:
                            return False
                        key = criteria.get("key")
                        value = criteria.get("value")
                        if key not in vdata.metadata or value is None:
                            return False
                        if not criteria.get("regex"):
                            if not vdata.metadata.get(key) == value:
                                return False
                        else:
                            regex_pattern = value
                            import re
                            if not re.match(regex_pattern, str(vdata.metadata.get(key))):
                                return False
                    return True
                return vd_filter

            vd_filters[role] = make_filter(criteria_list)
        
        return vd_filters


    def get_preprocessing_handlers(self, name: str, vd_filters: dict[str,VDFilter], **kwargs) -> List[BaseHandler]:
        """Get preprocessing handler instances for a verifier."""
        spec = self.get_spec(name)
        handlers = []
        for handler_class, init_kwargs in spec.preprocessing_handler_classes:
            merged_kwargs = {**init_kwargs, **kwargs}
            merged_kwargs = self.assign_filters(merged_kwargs, vd_filters)
            handlers.append(handler_class(**merged_kwargs))
        return handlers


    def assign_filters(self, handler_kwargs: Dict[str, Any], vd_filters: dict[str, VDFilter]) -> Dict[str, Any]:
        """Assign VDFilter functions to handler initialization arguments based on filter roles."""
        if "filter_roles" in handler_kwargs:
            filter_roles = handler_kwargs.pop("filter_roles")
            if any(role not in vd_filters for role in filter_roles):
                invalid_roles = [role for role in filter_roles if role not in vd_filters]
                raise ValueError(f"Invalid filter roles: {invalid_roles}. Available roles: {list(vd_filters.keys())}")
            if len(filter_roles) == 1:
                handler_kwargs["filter"] = vd_filters.get(filter_roles[0])
            elif len(filter_roles) > 1:
                handler_kwargs["filters"] = [vd_filters.get(role) for role in filter_roles]
        return handler_kwargs


    def list_verifiers(self) -> List[str]:
        """List all available verifier names."""
        return list(self._verifiers.keys())
    
    def get_verifier_info(self, name: str) -> VerifierInfo:
        """Get verifier information for API responses."""
        spec = self.get_spec(name)
        return VerifierInfo(
            name=spec.name,
            description=spec.description,
            input_types=spec.input_types,
            allowed_filter_roles=spec.allowed_filter_roles,
            config_options=spec.config_options,
            is_coherence_verifier=spec.is_coherence_verifier
        )
    
    def get_all_verifiers_info(self) -> VerifiersList:
        """Get information about all verifiers grouped by category."""
        core = []
        coherence = []
        content_check = []
        
        for spec in self._verifiers.values():
            info = self.get_verifier_info(spec.name)
            
            if spec.name.startswith("has_"):
                content_check.append(info)
            elif spec.is_coherence_verifier:
                coherence.append(info)
            else:
                core.append(info)
        
        return VerifiersList(
            core=core,
            coherence=coherence,
            content_check=content_check
        )
    
    def validate_config_options(self, verifier_name: str, config: Dict[str, Any]) -> List[str]:
        """Validate configuration options for a verifier. Returns list of invalid options."""
        spec = self.get_spec(verifier_name)
        valid_options = {opt.name for opt in spec.config_options}
        # Always allow 'filters' in config
        valid_options.add("filters")
        
        invalid_options = []
        for key in config.keys():
            if key not in valid_options:
                invalid_options.append(key)
        
        return invalid_options
    
    def validate_filter_roles(self, verifier_name: str, filter_roles: List[str]) -> List[str]:
        """Validate filter roles for a verifier. Returns list of invalid roles."""
        spec = self.get_spec(verifier_name)
        valid_roles = set(spec.allowed_filter_roles)
        
        invalid_roles = []
        for role in filter_roles:
            if role not in valid_roles:
                invalid_roles.append(role)
        
        return invalid_roles


# Global registry instance
verifier_registry = VerifierRegistry()
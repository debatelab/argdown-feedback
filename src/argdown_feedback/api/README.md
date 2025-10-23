

### Core Structure

```python
# Individual verifier endpoints - the main pattern
POST /verify/{verifier_name}
{
    "inputs": "string",           # Required: the code snippet  
    "source": "string",           # Optional: source text for some verifiers
    "config": {                   # Verifier-specific configuration
        "levenshtein_tolerance": 0.01,
        "from_key": "from",
        "N": 1,
        // ... other verifier-specific options
    }
}

# Discovery endpoints
GET /verifiers                    # List all available verifiers
GET /verifiers/{verifier_name}    # Get verifier details & config schema
```

### Available Verifier Endpoints

Core Verifiers:

```python
POST /verify/arganno              # ArgannoCompositeHandler
POST /verify/argmap               # ArgMapCompositeHandler  
POST /verify/infreco              # InfRecoCompositeHandler
POST /verify/logreco              # LogRecoCompositeHandler
POST /verify/has_annotations      # HasAnnotationsHandler
POST /verify/has_argdown          # HasArgdownHandler
```

Coherence Verifiers:

```python
POST /verify/arganno_argmap       # ArgAnnotArgmapCoherenceHandler
POST /verify/arganno_infreco      # ArgannoInfrecoCoherenceHandler
POST /verify/arganno_logreco      # ArgannoLogrecoCoherenceHandler
POST /verify/argmap_infreco       # ArgmapInfrecoCoherenceHandler
POST /verify/argmap_logreco       # ArgmapLogrecoCoherenceHandler
POST /verify/arganno_argmap_logreco # ArgannoArgmapLogrecoCoherenceHandler
```


Custom metadata filtering:

```python
@dataclass
class MetadataFilterRule:
    """Single metadata filter rule."""
    key: str
    value: Any
    regex: bool = False
    
    def matches(self, metadata_value: Any) -> bool:
        """Check if metadata value matches this rule."""
        if metadata_value is None:
            return False
            
        if self.regex:
            import re
            # Convert both to strings for regex matching
            pattern = str(self.value)
            text = str(metadata_value)
            return bool(re.search(pattern, text))
        else:
            # Exact match
            return metadata_value == self.value

@dataclass
class MetadataFilter:
    """Collection of metadata filter rules."""
    rules: List[MetadataFilterRule]
    
    def matches(self, vdata: PrimaryVerificationData) -> bool:
        """Check if verification data matches ALL filter rules (AND logic)."""
        if not vdata.metadata:
            return len(self.rules) == 0
        
        for rule in self.rules:
            metadata_value = vdata.metadata.get(rule.key)
            if not rule.matches(metadata_value):
                return False
        return True
```

```python
# Simple exact matching (core)
{
    "config": {
        filters:{
            "arganno": {
                "version": "v3",
                "task": "annotation"
            }
        }
    }
}

# Simple exact matching (coherence)
{
    "config": {
        filters:{
            "arganno": {
                "version": "v3",
                "task": "annotation"
            },
            "argmap": {
                "version": "v3",
                "task": "reconstruction"
            }
        }
    }
}


# Advanced with regex (core)
{
    "config": {
        "filters": {
            "arganno": [
                {
                    "key": "version", 
                    "value": "v[3-4]",
                    "regex": true
                },
                {
                    "key": "task",
                    "value": "reconstruction|analysis", 
                    "regex": true
                },
                {
                    "key": "author",
                    "value": "student123",
                    "regex": false  # exact match
                }
            ]
        }
    }
}
```

BUT note: identifiers should respect current default filters, e.g. arganno only applies if vd.dtype == VerificationDType.xml, and logreco, argmap, infreco apply if vd.dtype == VerificationDType.argdown.

```python
            # Default filters for argmap and infreco data
            def filter_fn1(vd: PrimaryVerificationData) -> bool:
                metadata: dict = vd.metadata if vd.metadata is not None else {}
                return vd.dtype == VerificationDType.argdown and metadata.get("filename", "").startswith("map")
            def filter_fn2(vd: PrimaryVerificationData) -> bool:
                metadata: dict = vd.metadata if vd.metadata is not None else {}
                return vd.dtype == VerificationDType.argdown and metadata.get("filename", "").startswith("reconstructions")

```

The current filters are:

```python
{
    "config": {
        "filters": {
            "argmap": {
                "filename": "map.ad",
            },
            "infreco": {
                "filename": "reconstructions.ad", 
            }
        }
    }
}
```

## Implementation Implications

* Filter Mapping: The filter identifiers ("argmap", "infreco", "arganno", "logreco") represent semantic roles, not just data types 
* Multiple Argdown Blocks: We can have multiple argdown blocks in one request, differentiated by metadata
* Last Block Selection: Code and Coherence verifiers use the most recent block matching each filter
* Backwards Compatibility: The default filters (like filename.startswith("map")) should remain as fallbacks

This is a much more sophisticated and flexible design than I initially understood. Your configuration approach is exactly right for handling multiple argdown blocks with different semantic purposes!


## Threading / Async 

```python 
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import time

class VerificationService:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.verifier_registry = self._build_verifier_registry()
    
    async def verify_async(self, verifier_name: str, request: VerificationRequest) -> Dict[str, Any]:
        start_time = time.time()
        
        # Run verification in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._verify_sync,
            verifier_name,
            request
        )
        
        # Add processing time to response
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        return result
    
    def _verify_sync(self, verifier_name: str, request: VerificationRequest) -> Dict[str, Any]:
        # Your existing synchronous verification logic
        handler = self.verifier_registry[verifier_name]
        verification_request = self._build_verification_request(request)
        result = handler.process(verification_request)
        
        return {
            "verifier": verifier_name,
            "is_valid": result.is_valid(),
            "verification_data": [vd.__dict__ for vd in result.verification_data],
            "results": [r.__dict__ for r in result.results],
            "executed_handlers": result.executed_handlers
        }

# Usage in FastAPI
verification_service = VerificationService(max_workers=8)

@app.post("/verify/{verifier_name}")
async def verify_code(verifier_name: str, request: VerificationRequest):
    return await verification_service.verify_async(verifier_name, request)
```

* Use multiple Uvicorn workers: `uvicorn app:app --workers 4`

E.g.:

```yml
# docker-compose.yml
services:
  api:
    build: .
    command: uvicorn app:app --workers 4 --host 0.0.0.0
    ports:
      - "8000:8000"
  
  nginx:
    image: nginx
    # Load balance across multiple API instances
```

## Client

```python
# client.py
import asyncio
from typing import Dict, Any, Optional, List, Union
import httpx
from dataclasses import dataclass

from .models import VerificationRequest, VerificationResponse, VerifierInfo

class VerifiersClient:
    def __init__(self, base_url: str, async_client: bool = True):
        self.base_url = base_url
        if async_client:
            self.client = httpx.AsyncClient(timeout=30.0)
        else:
            self.client = httpx.Client(timeout=30.0)
        self.is_async = async_client
    
    async def verify_async(self, verifier_name: str, request: VerificationRequest) -> VerificationResponse:
        response = await self.client.post(
            f"{self.base_url}/verify/{verifier_name}",
            json=request.__dict__
        )
        response.raise_for_status()
        return VerificationResponse(**response.json())
    
    def verify_sync(self, verifier_name: str, request: VerificationRequest) -> VerificationResponse:
        response = self.client.post(
            f"{self.base_url}/verify/{verifier_name}",
            json=request.__dict__
        )
        response.raise_for_status()
        return VerificationResponse(**response.json())
    
    def verify(self, verifier_name: str, request: VerificationRequest) -> Union[VerificationResponse, asyncio.Task]:
        if self.is_async:
            return self.verify_async(verifier_name, request)
        else:
            return self.verify_sync(verifier_name, request)


# Sync usage
client = VerifiersClient("https://api.example.com")
result = client.verify(
    "infreco",
    inputs="""```argdown
    <Arg>: Test argument.
    (1) Premise
    -- {from: ["1"]} --
    (2) Conclusion
    ```""",
    from_key="from"
)

# Async usage
async_client = VerifiersClient("https://api.example.com", async_client=True)
result = await async_client.verify("infreco", request)


```


Type checking:

```python
from typing import Literal, Union, get_args

# Define allowed filter roles for each verifier using Literal types
InfrecoRoles = Literal["infreco"]
ArgannoRoles = Literal["arganno"] 
ArgmapRoles = Literal["argmap"]
ArgmapInfrecoRoles = Literal["argmap", "infreco"]
ArgannoArgmapRoles = Literal["arganno", "argmap"]


@dataclass
class FilterRule:
    key: str
    value: Any
    regex: bool = False

class FilterBuilder:
    def __init__(self):
        self._filters: Dict[str, List[FilterRule]] = {}
    
    def add(self, role: str, key: str, value: Any, regex: bool = False) -> 'FilterBuilder':
        """Add a filter rule for a role."""
        if role not in self._filters:
            self._filters[role] = []
        self._filters[role].append(FilterRule(key=key, value=value, regex=regex))
        return self
    
    def build(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build the final filters dictionary."""
        return {
            role: [{"key": rule.key, "value": rule.value, "regex": rule.regex} for rule in rules]
            for role, rules in self._filters.items()
        }


# Verifier-specific request builders
class InfrecoRequestBuilder:
    def __init__(self, inputs: str, source: str = None):
        self.inputs = inputs
        self.source = source
        self.config = {}
        self.filter_builder = TypedFilterBuilder()
    
    def config_option(self, key: Literal["from_key", "N"], value: Any) -> 'InfrecoRequestBuilder':
        """Only allow valid config options for infreco."""
        self.config[key] = value
        return self
    
    def add_filter(self, role: InfrecoRoles, key: str, value: Any, regex: bool = False) -> 'InfrecoRequestBuilder':
        """Only allow 'infreco' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        config_dict = self.config.copy()
        if self.filter_builder._filters:
            config_dict["filters"] = self.filter_builder.build()
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )

class ArgmapInfrecoRequestBuilder:
    def __init__(self, inputs: str, source: str = None):
        self.inputs = inputs
        self.source = source
        self.config = {}
        self.filter_builder = TypedFilterBuilder()
    
    def config_option(self, key: Literal["from_key"], value: Any) -> 'ArgmapInfrecoRequestBuilder':
        """Only allow valid config options for argmap_infreco."""
        self.config[key] = value
        return self
    
    def add_filter(self, role: ArgmapInfrecoRoles, key: str, value: Any, regex: bool = False) -> 'ArgmapInfrecoRequestBuilder':
        """Allow 'argmap' and 'infreco' role filters."""
        self.filter_builder.add(role, key, value, regex)
        return self
    
    def build(self) -> VerificationRequest:
        config_dict = self.config.copy()
        if self.filter_builder._filters:
            config_dict["filters"] = self.filter_builder.build()
        
        return VerificationRequest(
            inputs=self.inputs,
            source=self.source,
            config=config_dict if config_dict else None
        )

# Factory functions
def create_infreco_request(inputs: str, source: str = None) -> InfrecoRequestBuilder:
    return InfrecoRequestBuilder(inputs, source)

def create_argmap_infreco_request(inputs: str, source: str = None) -> ArgmapInfrecoRequestBuilder:
    return ArgmapInfrecoRequestBuilder(inputs, source)

# Usage with IDE type checking:
request = (create_infreco_request(inputs=text)
    .config_option("from_key", "premises")     # ✅ Valid
    .config_option("invalid_key", "value")     # ❌ Type error
    .add_filter("infreco", "version", "v3")    # ✅ Valid
    .add_filter("argmap", "version", "v3")     # ❌ Type error
    .build())
```
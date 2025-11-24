# Argdown Feedback API

## Overview

The Argdown Feedback API provides a FastAPI-based REST service for verifying argdown documents, XML annotations, and checking coherence between different types of argument analysis artifacts. The API uses a handler-based architecture with configurable filtering and virtue scoring capabilities.

## Running the Server

### Quick Start

The FastAPI server automatically downloads required NLTK resources (`punkt`) at startup via the lifespan handler. No manual setup is required.

To start the server:

```bash
uvicorn argdown_feedback.api.server.app:app --reload
```

The API will be available at `http://localhost:8000` with:
- Interactive docs at `http://localhost:8000/docs`
- ReDoc documentation at `http://localhost:8000/redoc`
- Health check at `http://localhost:8000/health`

### Production Deployment

For production use with multiple workers:

```bash
uvicorn argdown_feedback.api.server.app:app --workers 4 --host 0.0.0.0 --port 8000
```

**Note for Testing/Direct Usage**: If you're running tests or using the verifiers directly (not through the FastAPI server), you'll need to download NLTK resources manually:

```python
import nltk
nltk.download('punkt')
```

## API Endpoints

All endpoints use the `/api/v1` prefix.

### Core Structure

```
POST /api/v1/verify/{verifier_name}
GET  /api/v1/verifiers
GET  /api/v1/verifiers/{verifier_name}
GET  /api/v1/verify/{verifier_name}/info
GET  /health
```

### Verification Request Format

```json
{
    "inputs": "string",           // Required: text containing code blocks  
    "source": "string",           // Optional: source text for some verifiers
    "config": {                   // Optional: verifier-specific configuration
        "from_key": "from",
        "filters": {
            "role_name": {
                "key": "value"    // Simple exact match
            },
            "another_role": [     // Advanced with multiple rules/regex
                {
                    "key": "version",
                    "value": "v[3-4]",
                    "regex": true
                }
            ]
        },
        "enable_scorer_name": true   // Enable specific virtue scorers
    }
}
```

### Verification Response Format

```json
{
    "verifier": "string",
    "is_valid": true,
    "verification_data": [
        {
            "id": "string",
            "dtype": "argdown|xml",
            "code_snippet": "string",
            "metadata": {}
        }
    ],
    "results": [
        {
            "verifier_id": "string",
            "verification_data_references": ["string"],
            "is_valid": true,
            "message": "string",
            "details": {}
        }
    ],
    "scores": [
        {
            "scorer_id": "string",
            "scorer_description": "string",
            "scoring_data_references": ["string"],
            "score": 0.85,
            "message": "string",
            "details": {}
        }
    ],
    "executed_handlers": ["string"],
    "processing_time_ms": 42.5
}
```

## Available Verifiers

### Core Verifiers

Core verifiers validate individual types of argument analysis artifacts:

| Endpoint | Description | Input Types | Filter Roles | Config Options |
|----------|-------------|-------------|--------------|----------------|
| `POST /api/v1/verify/arganno` | Validates XML annotations | xml | arganno | - |
| `POST /api/v1/verify/argmap` | Validates argument maps | argdown | argmap | - |
| `POST /api/v1/verify/infreco` | Validates informal reconstructions | argdown | infreco | from_key |
| `POST /api/v1/verify/logreco` | Validates logical reconstructions | argdown | logreco | from_key, formalization_key, declarations_key |
| `POST /api/v1/verify/has_annotations` | Checks for XML annotations | xml | - | - |
| `POST /api/v1/verify/has_argdown` | Checks for argdown content | argdown | - | - |

### Coherence Verifiers

Coherence verifiers check consistency between different types of artifacts:

| Endpoint | Description | Filter Roles | Config Options |
|----------|-------------|--------------|----------------|
| `POST /api/v1/verify/arganno_argmap` | Checks arganno ↔ argmap coherence | arganno, argmap | - |
| `POST /api/v1/verify/arganno_infreco` | Checks arganno ↔ infreco coherence | arganno, infreco | from_key |
| `POST /api/v1/verify/arganno_logreco` | Checks arganno ↔ logreco coherence | arganno, logreco | from_key |
| `POST /api/v1/verify/argmap_infreco` | Checks argmap ↔ infreco coherence | argmap, infreco | from_key |
| `POST /api/v1/verify/argmap_logreco` | Checks argmap ↔ logreco coherence | argmap, logreco | from_key |
| `POST /api/v1/verify/arganno_argmap_logreco` | Checks arganno ↔ argmap ↔ logreco coherence | arganno, argmap, logreco | from_key |

### Discovery Endpoints

```bash
# List all available verifiers with details
GET /api/v1/verifiers

# Get detailed information about a specific verifier
GET /api/v1/verifiers/{verifier_name}
```

## Metadata Filtering System

### Overview

The filtering system allows you to work with multiple code blocks in a single request by using metadata to distinguish between different semantic roles (e.g., argmap vs. infreco).

### Filter Roles

Filter roles represent semantic purposes, not just data types:
- `arganno`: XML annotation blocks
- `argmap`: Argument map argdown blocks
- `infreco`: Informal reconstruction argdown blocks  
- `logreco`: Logical reconstruction argdown blocks

### Filter Format

The API supports two filter formats, automatically chosen based on complexity:

#### Simple Format (Single Exact Match)

```json
{
    "config": {
        "filters": {
            "infreco": {
                "filename": "reconstruction.ad"
            }
        }
    }
}
```

#### Advanced Format (Multiple Rules or Regex)

```json
{
    "config": {
        "filters": {
            "infreco": [
                {
                    "key": "version",
                    "value": "v[3-4]",
                    "regex": true
                },
                {
                    "key": "filename",
                    "value": "reconstruction.*",
                    "regex": true
                }
            ]
        }
    }
}
```

### Filter Semantics

- **Filter matching**: ALL rules within a role must match (AND logic)
- **Data type constraints**: Filters automatically respect data types:
  - `arganno` only applies if `dtype == VerificationDType.xml`
  - `argmap`, `infreco`, `logreco` only apply if `dtype == VerificationDType.argdown`
- **Block selection**: Verifiers use the most recent block matching each filter

### Default Filters

Coherence verifiers apply default filters if none are specified:

```python
# Default filter for argmap
{"argmap": [{"key": "filename", "value": "map.*", "regex": true}]}

# Default filter for infreco
{"infreco": [{"key": "filename", "value": "reconstruction.*", "regex": true}]}
```

### Multiple Argdown Blocks Example

```json
{
    "inputs": "```argdown
    <!-- filename: map.ad -->
    <Argument Map>: The overall argument structure
    ...
    ```
    
    ```argdown
    <!-- filename: reconstruction.ad -->
    <Arg>: Detailed reconstruction
    (1) Premise
    -- {from: [\"1\"]} --
    (2) Conclusion
    ```",
    "config": {
        "filters": {
            "argmap": {
                "filename": "map.ad"
            },
            "infreco": {
                "filename": "reconstruction.ad"
            }
        }
    }
}
```

## Virtue Scoring System

### Overview

Verifiers can optionally score argument analysis artifacts along various normative dimensions (virtues). Each verifier has associated scorer classes that can be enabled via configuration.

### Enabling Scorers

Scorers are disabled by default to reduce processing time. Enable them using config options:

```json
{
    "config": {
        "enable_infreco_subarguments_scorer": true,
        "enable_infreco_premises_scorer": true,
        "enable_argmap_size_scorer": true
    }
}
```

### Available Scorers

#### Informal Reconstruction (infreco)

- `infreco_subarguments_scorer`: Evaluates number of sub-arguments (intermediate conclusions)
- `infreco_premises_scorer`: Evaluates number of premises

#### Argument Maps (argmap, argmap_infreco)

- `argmap_size_scorer` / `argmap_infreco_size_scorer`: Evaluates argument map size (nodes + edges)
- `argmap_density_scorer` / `argmap_infreco_density_scorer`: Evaluates map density (edges/nodes ratio)
- `argmap_faithfulness_scorer` / `argmap_infreco_faithfulness_scorer`: Evaluates faithfulness to source text using Levenshtein distance

### Score Response Format

```json
{
    "scores": [
        {
            "scorer_id": "infreco.infreco_premises_scorer",
            "scorer_description": "Scores the number of premises in the informal reconstruction.",
            "scoring_data_references": [],
            "score": 0.75,
            "message": "Number of premises found: 3.",
            "details": {
                "premises_count": 2
            }
        }
    ]
}
```

## Async Processing Architecture

### Thread Pool Executor

The verification service uses `ThreadPoolExecutor` to run CPU-intensive verification tasks in background threads while maintaining FastAPI's async capabilities:

```python
class VerificationService:
    def __init__(self, max_workers: int = 8):  # Default: 8 workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def verify_async(self, verifier_name: str, request: VerificationRequest):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._verify_sync,
            verifier_name,
            request
        )
        return result
```

### Benefits

- Non-blocking verification for concurrent requests
- Efficient resource utilization
- Automatic processing time tracking
- Clean separation between async I/O and CPU-bound work

## Python Client

### Installation

The client is included in the `argdown_feedback.api.client` module.

### Basic Usage

#### Synchronous Client

```python
from argdown_feedback.api.client import VerifiersClient
from argdown_feedback.api.shared.models import VerificationRequest

# Initialize client
client = VerifiersClient("http://localhost:8000", async_client=False)

# Create request
request = VerificationRequest(
    inputs="""```argdown
    <Arg>: Test argument.
    (1) Premise
    -- {from: ["1"]} --
    (2) Conclusion
    ```""",
    config={"from_key": "from"}
)

# Verify
response = client.verify_sync("infreco", request)
print(f"Valid: {response.is_valid}")
print(f"Processing time: {response.processing_time_ms}ms")

# Cleanup
client.close()
```

#### Asynchronous Client

```python
import asyncio
from argdown_feedback.api.client import VerifiersClient
from argdown_feedback.api.shared.models import VerificationRequest

async def main():
    # Initialize async client
    async with VerifiersClient("http://localhost:8000", async_client=True) as client:
        # Create request
        request = VerificationRequest(
            inputs="...",
            config={"from_key": "from"}
        )
        
        # Verify
        response = await client.verify_async("infreco", request)
        print(f"Valid: {response.is_valid}")
        
        # List available verifiers
        verifiers = await client.list_verifiers_async()
        print(f"Available verifiers: {len(verifiers.core)} core, {len(verifiers.coherence)} coherence")

asyncio.run(main())
```

### Type-Safe Request Builders

The client provides type-safe request builders with compile-time validation:

```python
from argdown_feedback.api.client.builders import (
    create_infreco_request,
    create_argmap_infreco_request
)

# Type-safe builder for infreco verifier
request = (create_infreco_request(inputs=text)
    .config_option("from_key", "premises")     # ✅ Valid
    # .config_option("invalid_key", "value")   # ❌ Type error in IDE
    .add_filter("infreco", "version", "v3")    # ✅ Valid  
    # .add_filter("argmap", "version", "v3")   # ❌ Type error in IDE
    .build())

# Type-safe builder for coherence verifier
request = (create_argmap_infreco_request(inputs=text)
    .config_option("from_key", "from")
    .add_filter("argmap", "filename", "map.ad")
    .add_filter("infreco", "filename", "reconstruction.ad")
    .build())
```

### Available Builder Functions

**Core Verifiers:**
- `create_arganno_request()`
- `create_argmap_request()`
- `create_infreco_request()`
- `create_logreco_request()`
- `create_has_annotations_request()`
- `create_has_argdown_request()`

**Coherence Verifiers:**
- `create_arganno_argmap_request()`
- `create_arganno_infreco_request()`
- `create_arganno_logreco_request()`
- `create_argmap_infreco_request()`
- `create_argmap_logreco_request()`
- `create_arganno_argmap_logreco_request()`

## Error Handling

### Error Response Format

```json
{
    "detail": {
        "error": "error_type",
        "message": "Human-readable error message",
        "additional_field": "..."
    }
}
```

### HTTP Status Codes

| Code | Error Type | Description |
|------|------------|-------------|
| 400 | `verification_failed` | Verification processing failed |
| 404 | `verifier_not_found` | Invalid verifier name |
| 422 | `invalid_config` | Invalid configuration options |
| 422 | `invalid_filter` | Invalid filter roles |
| 500 | `internal_server_error` | Unexpected server error |

### Example Error Responses

**Verifier Not Found (404)**
```json
{
    "detail": {
        "error": "verifier_not_found",
        "message": "Verifier 'invalid_name' not found",
        "verifier_name": "invalid_name",
        "available_verifiers": ["arganno", "argmap", "infreco", ...]
    }
}
```

**Invalid Configuration (422)**
```json
{
    "detail": {
        "error": "invalid_config",
        "message": "Invalid configuration options: ['bad_option']",
        "invalid_options": ["bad_option"]
    }
}
```

**Invalid Filters (422)**
```json
{
    "detail": {
        "error": "invalid_filter",
        "message": "Invalid filter roles for infreco: ['argmap']",
        "invalid_roles": ["argmap"]
    }
}
```

## Architecture Overview

### Component Structure

```
api/
├── client/                 # HTTP client library
│   ├── client.py          # Sync/async client implementation
│   └── builders.py        # Type-safe request builders
├── server/                 # FastAPI server
│   ├── app.py             # Main application & middleware
│   ├── routes/            # API route handlers
│   │   ├── verification.py
│   │   └── discovery.py
│   └── services/          # Business logic
│       ├── verification_service.py   # Async processing
│       ├── verifier_registry.py      # Verifier management
│       └── builders/                 # Verifier builders
│           ├── base.py
│           ├── core/                 # Core verifier builders
│           └── coherence/            # Coherence verifier builders
└── shared/                 # Shared models & utilities
    ├── models.py          # Pydantic models
    ├── filtering.py       # Filter builders
    └── exceptions.py      # Custom exceptions
```

### Key Design Patterns

1. **Builder Pattern**: Verifiers are constructed via builder classes that encapsulate configuration and pipeline construction
2. **Registry Pattern**: `VerifierRegistry` manages available verifiers and validates configurations
3. **Filter System**: Flexible metadata-based filtering for multi-block code processing
4. **Scorer Pattern**: Pluggable virtue scorers with enable/disable configuration
5. **Async Wrapper**: ThreadPoolExecutor wraps synchronous verification for async FastAPI endpoints

### Extending the API

#### Adding a New Verifier

1. Create a builder class in `services/builders/`:

```python
from .base import VerifierBuilder

class MyVerifierBuilder(VerifierBuilder):
    name = "my_verifier"
    description = "Description of what it does"
    input_types = ["argdown"]
    allowed_filter_roles = ["my_role"]
    config_options = [...]
    
    def build_handlers_pipeline(self, filters_spec, **kwargs):
        # Build handler pipeline
        return [...]
```

2. Import the builder in `services/builders/__init__.py` or appropriate category module

3. The verifier will be automatically registered and available at `/api/v1/verify/my_verifier`

#### Adding a New Scorer

1. Create a scorer class:

```python
from argdown_feedback.api.server.services.verifier_registry import BaseScorer

class MyScorer(BaseScorer):
    scorer_id = "my_scorer"
    scorer_description = "Description of what this scores"
    
    def score(self, result: VerificationRequest) -> ScoringResult:
        # Implement scoring logic
        return ScoringResult(...)
```

2. Add to the verifier builder's `scorer_classes` list

3. Enable via config: `{"enable_my_scorer": true}`

## Development & Testing

### Running Tests

```bash
# All API tests
pytest tests/api/

# Integration tests only
pytest tests/api/integration/

# Unit tests only  
pytest tests/api/unit/

# Performance tests
pytest tests/api/performance/
```

### Interactive API Testing

Use the built-in Swagger UI:

```bash
uvicorn argdown_feedback.api.server.app:app --reload
# Visit http://localhost:8000/docs
```

## Configuration Reference

### Common Config Options

| Option | Type | Default | Description | Verifiers |
|--------|------|---------|-------------|-----------|
| `from_key` | string | "from" | Key for inference info | infreco, logreco, coherence |
| `formalization_key` | string | - | Key for formalization | logreco |
| `declarations_key` | string | - | Key for declarations | logreco |
| `enable_<scorer_id>` | bool | false | Enable specific scorer | All with scorers |

### Verifier-Specific Configurations

Each verifier's available options can be discovered via:

```bash
GET /api/v1/verifiers/{verifier_name}
```

This returns a `VerifierInfo` object with `config_options` listing all available configuration parameters.

## Performance Considerations

- **Worker Threads**: Default 8 workers handle concurrent CPU-bound verification tasks
- **Async I/O**: FastAPI handles I/O-bound operations (HTTP requests) asynchronously
- **Scorer Performance**: Disable scorers if not needed to reduce processing time
- **Filter Complexity**: Simple filters are faster than regex-based filters
- **Code Block Extraction**: Large inputs with many code blocks may increase processing time

## License

[Include your license information]

## Contributing

[Include contribution guidelines]

```
src/argdown_feedback/
├── api/                           # New: All API-related code
│   ├── __init__.py               # API package exports
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── routes/
│   │   ├── services/
│   │   └── middleware/
│   ├── client/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── builders/
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── models.py             # Shared Pydantic models
│   │   ├── filtering.py
│   │   └── exceptions.py
│   └── types.py                  # API-specific type definitions
├── verifiers/                    # Existing: Core verification logic
│   ├── __init__.py
│   ├── base.py
│   ├── verification_request.py
│   ├── processing_handler.py
│   ├── core/
│   └── coherence/
├── tasks/                        # Existing: Task definitions
├── logic/                        # Existing: Logic utilities
└── __init__.py
```

Phase 1: Foundation & Models (Week 1)
Create shared models (server/models.py)

VerificationRequest with config and filters fields
VerificationResponse with all result fields
VerifierInfo for discovery API
FilterRule and related filter classes
Set up basic FastAPI app (server/app.py)

Basic FastAPI application setup
CORS configuration
Health check endpoint
Basic error handling
Create verifier registry (server/services/verifier_registry.py)

Registry mapping verifier names to handler classes
Verifier specifications (allowed roles, config schemas)
Factory functions for creating handlers


Phase 2: Core Verification Service (Week 2)
Implement verification service (server/services/verification_service.py)

Threading setup with ThreadPoolExecutor
Core verification logic that bridges API models to verifier handlers
Request/response transformation
Error handling and logging
Enhanced filtering system (server/services/filter_service.py)

Configurable metadata filter creation
Regex support implementation
Integration with existing FencedCodeBlockExtractor
Role-based filter application
Basic verification endpoints (server/routes/verification.py)

/verify/{verifier_name} POST endpoint
Request validation
Integration with verification service


Phase 3: Discovery & Advanced Features (Week 3)
Discovery API (server/routes/discovery.py)

GET /verifiers - list all verifiers
GET /verifiers/{verifier_name} - verifier details and config schema
Dynamic schema generation from verifier specs
Configuration system (server/config.py)

Server configuration classes
Environment-based configuration
Deployment settings (workers, timeouts, etc.)
Error handling & middleware (server/middleware/)

Global exception handlers
Request/response logging
Rate limiting (optional)


Phase 4: Type-Safe Client Library (Week 4)
Base client implementation (client/client.py)

VerifiersClient with sync/async support
HTTP client setup with proper error handling
Basic verification methods
Filter builder system (base.py)

FilterBuilder with .add() method
Conversion to API format
Validation logic
Verifier-specific builders (client/builders/)

Type definitions with Literal types for roles and config options
Individual builder classes for each verifier
Factory functions with type hints


Phase 5: Integration & Testing (Week 5)
Integration testing

End-to-end tests with real verifier handlers
Client-server integration tests
Performance testing with threading
Type checking validation

mypy validation for client type safety
IDE integration testing
Documentation for type-safe usage
Deployment preparation

Docker configuration for multi-worker deployment
nginx configuration examples
Production deployment guidelines


Phase 6: Documentation & Polish (Week 6)
API documentation

OpenAPI/Swagger documentation
Client library documentation
Usage examples and tutorials
Performance optimization

Thread pool tuning
Memory usage optimization
Caching strategies (if needed)
Final testing & release preparation

Load testing
Security review
Package building and distribution


Key Implementation Decisions
Model Sharing Strategy
Use shared Pydantic models between client and server
Import server models directly in client (Option 1 from our discussion)
Ensures perfect type compatibility
Threading Architecture
ThreadPoolExecutor for CPU-intensive verification tasks
Multiple Uvicorn workers for horizontal scaling
Async FastAPI endpoints with sync verification in threads
Type Safety Approach
Literal types for compile-time validation
Verifier-specific builder classes
Factory functions with proper type hints
Filter System Integration
Extend existing FencedCodeBlockExtractor with configurable filters
Maintain backward compatibility with current default filters
Support both simple dict and advanced regex filters
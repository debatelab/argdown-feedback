#!/usr/bin/env python3
"""
Demo script showing the type-safe client builders in action.

This demonstrates the exact usage patterns described in the README,
including the example: create_infreco_request(inputs=text).config_option("from_key", "premises").add_filter("infreco", "version", "v3").build()
"""

import sys
sys.path.insert(0, 'src')

from argdown_feedback.api.client.builders import (
    create_infreco_request,
    create_arganno_request, 
    create_argmap_request,
    create_logreco_request,
    create_argmap_infreco_request,
    create_arganno_argmap_request,
    create_arganno_infreco_request,
    create_argmap_logreco_request,
    create_arganno_logreco_request,
    create_arganno_argmap_logreco_request
)


def demo_readme_example():
    """Demo the exact example from the README."""
    print("=== README Example ===")
    
    text = """
```argdown
<P1>: Some premise
<P2>: Another premise  
----
<C1>: Conclusion
```
"""
    
    # Exact README example
    request = (create_infreco_request(inputs=text)
        .config_option("from_key", "premises")  
        .add_filter("infreco", "version", "v3")
        .build())
    
    print("✅ README example works!")
    print(f"   Inputs: {request.inputs[:50]}...")
    print(f"   Config: {request.config}")
    print()


def demo_advanced_filtering():
    """Demo advanced filtering capabilities."""
    print("=== Advanced Filtering Demo ===")
    
    # Complex filter with regex
    request = (create_infreco_request("Test content")
        .config_option("from_key", "premises")
        .add_filter("infreco", "version", "v3", regex=False)
        .add_filter("infreco", "task", "reconstruction|analysis", regex=True)
        .add_filter("infreco", "quality", "high", regex=False)
        .build())
    
    print("✅ Complex inference reconstruction with multiple filters")
    print(f"   Config: {request.config}")
    print()


def demo_coherence_checks():
    """Demo coherence verification builders."""
    print("=== Coherence Verification Demo ===")
    
    # Argmap + Infreco coherence
    request = (create_argmap_infreco_request("Test coherence content")
        .config_option("from_key", "from")
        .add_filter("argmap", "version", "v3")
        .add_filter("infreco", "version", "v3") 
        .add_filter("infreco", "task", "analysis")
        .build())
    
    print("✅ Argmap + Infreco coherence check")
    print(f"   Config: {request.config}")
    
    # Triple coherence check  
    triple_request = (create_arganno_argmap_logreco_request("Complex content")
        .config_option("from_key", "mapping")
        .add_filter("arganno", "annotator", "expert")
        .add_filter("argmap", "version", "v3")
        .add_filter("logreco", "level", "detailed")
        .build())
    
    print("✅ Triple coherence check (Arganno + Argmap + Logreco)")
    print(f"   Config: {triple_request.config}")
    print()


def demo_type_safety():
    """Demo compile-time type safety features."""
    print("=== Type Safety Demo ===")
    
    # This works - correct verifier role
    try:
        (create_infreco_request("content")
            .add_filter("infreco", "version", "v3")  # ✅ Valid role
            .build())
        print("✅ Valid filter role accepted")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Demonstrate type safety (this shows a type error at development time)
    print("⚠️  Type checker prevents invalid filter roles at compile time")
    print("   Example: create_infreco_request().add_filter('argmap', ...) would fail type check")
    print()


def demo_all_verifiers():
    """Demo builders for available verifier types."""
    print("=== Available Verifier Types Demo ===")
    
    builders = [
        ("Inference Reconstruction", create_infreco_request),
        ("Argument Annotation", create_arganno_request),
        ("Argument Mapping", create_argmap_request), 
        ("Logical Reconstruction", create_logreco_request),
        ("Argmap + Infreco", create_argmap_infreco_request),
        ("Arganno + Argmap", create_arganno_argmap_request),
        ("Arganno + Infreco", create_arganno_infreco_request),
        ("Argmap + Logreco", create_argmap_logreco_request),
        ("Arganno + Logreco", create_arganno_logreco_request),
        ("Arganno + Argmap + Logreco", create_arganno_argmap_logreco_request),
    ]
    
    for name, builder_func in builders:
        request = builder_func("Test content").build()
        print(f"✅ {name}: {type(request).__name__}")
    
    print()


def demo_filter_composition():
    """Demo how filters compose for complex verifiers."""
    print("=== Filter Composition Demo ===")
    
    # Show how filters work for multi-verifier scenarios
    request = (create_arganno_argmap_logreco_request("Complex analysis")
        .config_option("from_key", "comprehensive")
        .add_filter("arganno", "version", "v3")
        .add_filter("arganno", "annotator", "expert") 
        .add_filter("argmap", "version", "v3")
        .add_filter("argmap", "layout", "hierarchical")
        .add_filter("logreco", "level", "detailed")
        .add_filter("logreco", "format", "natural", regex=False)
        .build())
    
    print("✅ Complex 3-verifier composition with role-specific filters")
    filters = request.config.get('filters', {}) if request.config else {}
    print(f"   Config filters: {filters}")
    print()


if __name__ == "__main__":
    print("🚀 Type-Safe Client Builders Demo")
    print("=" * 50)
    print()
    
    demo_readme_example()
    demo_advanced_filtering() 
    demo_coherence_checks()
    demo_type_safety()
    demo_all_verifiers()
    demo_filter_composition()
    
    print("✅ All demos completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- ✅ README example works exactly as specified")
    print("- ✅ Type-safe verifier role validation")
    print("- ✅ Fluent builder API with method chaining")
    print("- ✅ Custom metadata filtering with regex support")
    print("- ✅ Available verifier combinations supported")
    print("- ✅ Compile-time safety with Literal types")
    print("- ✅ Complex filter composition for multi-verifier scenarios")
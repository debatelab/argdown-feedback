"checks dialectical relations between propositions in Argdown"

from pyargdown import (
    Argdown,
    DialecticalType,
    Proposition,
    Valence,
)

NEGATION_SCHEMES = [
    "NOT: {text}",
    "Not: {text}",
    "NOT {text}",
    "Not {text}",   
]


def are_identical(prop1: Proposition | None, prop2: Proposition | None) -> bool:
    """Check if two propositions are identical."""
    if prop1 is None or prop2 is None:
        return False
    return (
        prop1.label == prop2.label
        or any(text in prop2.texts for text in prop1.texts)
    )

def are_contradictory(prop1: Proposition | None, prop2: Proposition | None, argdown: Argdown | None = None) -> bool:
    """Check if two propositions are identical."""
    if prop1 is None or prop2 is None:
        return False
    if prop1.label == prop2.label:
        return False
    if argdown is not None:
        if any(
            drel.source in [prop1.label, prop2.label]
            and drel.target in [prop1.label, prop2.label]
            and drel.source != drel.target
            and drel.valence in [Valence.ATTACK, Valence.CONTRADICT]
            for drel in argdown.dialectical_relations
        ):
            return True
    negations_prop1 = [
        scheme.format(text=text) for scheme in NEGATION_SCHEMES for text in prop1.texts
    ]
    negations_prop2 = [
        scheme.format(text=text) for scheme in NEGATION_SCHEMES for text in prop2.texts
    ]
    return (
        any(text in negations_prop2 for text in prop1.texts)
        or any(text in negations_prop1 for text in prop2.texts)
    )

def indirectly_supports(from_label: str, to_label: str, argdown_map: Argdown) -> bool:
    """Check if one node directly or indirectly (via intermediate prop) supports another one in argument map."""
    if from_label == to_label:
        return True

    rels_direct = argdown_map.get_dialectical_relation(from_label, to_label)
    rels_direct = [] if rels_direct is None else rels_direct
    if any(rd.valence == Valence.SUPPORT for rd in rels_direct):
        return True

    for prop in argdown_map.propositions:
        if prop.label is None or prop.label == from_label or prop.label == to_label:
            continue
        rels1 = argdown_map.get_dialectical_relation(from_label, prop.label)
        rels2 = argdown_map.get_dialectical_relation(prop.label, to_label)
        rels1 = [] if rels1 is None else rels1
        rels2 = [] if rels2 is None else rels2
        for rel1 in rels1:
            for rel2 in rels2:
                if (
                    (rel1.valence == Valence.SUPPORT and rel2.valence == Valence.SUPPORT)
                    or (rel1.valence in [Valence.ATTACK, Valence.CONTRADICT] and rel2.valence in [Valence.ATTACK, Valence.CONTRADICT]) 
                ):
                    return True

    return False


def indirectly_attacks(from_label: str, to_label: str, argdown_map: Argdown) -> bool:
    """Check if one node directly or indirectly (via intermediate prop) attacks another one in argument map."""
    if from_label == to_label:
        return False

    rels_direct = argdown_map.get_dialectical_relation(from_label, to_label)
    rels_direct = [] if rels_direct is None else rels_direct
    if any(rd.valence == Valence.ATTACK for rd in rels_direct):
        return True

    for prop in argdown_map.propositions:
        if prop.label is None or prop.label == from_label or prop.label == to_label:
            continue
        rels1 = argdown_map.get_dialectical_relation(from_label, prop.label)
        rels2 = argdown_map.get_dialectical_relation(prop.label, to_label)
        rels1 = [] if rels1 is None else rels1
        rels2 = [] if rels2 is None else rels2
        for rel1 in rels1:
            for rel2 in rels2:
                if (
                    (rel1.valence == Valence.SUPPORT and rel2.valence in [Valence.ATTACK, Valence.CONTRADICT])
                    or (rel1.valence in [Valence.ATTACK, Valence.CONTRADICT] and rel2.valence == Valence.SUPPORT) 
                ):
                    return True

    return False



def are_strictly_identical(prop1: Proposition | None, prop2: Proposition | None, argdown: Argdown | None = None) -> bool:
    """Check if two propositions are identical."""
    if (
        prop1 is None or prop1.label is None
        or prop2 is None or prop2.label is None
    ):
        return False

    # equivalence via dialectical relations        
    if argdown is not None:
        rels1 = argdown.get_dialectical_relation(prop1.label, prop2.label)
        rels2 = argdown.get_dialectical_relation(prop2.label, prop1.label)
        if rels1 and rels2 :
            for rel1 in rels1:
                for rel2 in rels2:
                    if (
                        rel1 is not None and rel2 is not None
                        and rel1.valence == Valence.SUPPORT
                        and DialecticalType.AXIOMATIC in rel1.dialectics
                        and rel2.valence == Valence.SUPPORT
                        and DialecticalType.AXIOMATIC in rel2.dialectics
                    ):
                        return True

    return prop1.label == prop2.label


def are_strictly_contradictory(prop1: Proposition | None, prop2: Proposition | None, argdown: Argdown | None = None) -> bool:
    """Check if two propositions are identical."""
    if prop1 is None or prop2 is None:
        return False
    if prop1.label == prop2.label:
        return False
    if argdown is not None:
        if any(
            drel.source in [prop1, prop2]
            and drel.target in [prop1, prop2]
            and drel.source != drel.target
            and drel.valence in [Valence.ATTACK, Valence.CONTRADICT]
            and DialecticalType.AXIOMATIC in drel.dialectics
            for drel in argdown.dialectical_relations
        ):
            return True
    return False

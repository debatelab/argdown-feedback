"""smtlib.py



Z3:

(define-fun premise1 () Bool (=> p q))
(define-fun premise2 () Bool (=> q r))
(define-fun premise3 () Bool (=> r s))
;; Define the premise:
(define-fun conclusion () Bool (=> p s))
(define-fun argument () Bool
        (=> (and premise1 premise2 premise3)
                conclusion)
)
(assert (not argument))
(check-sat)


(echo "Type here and press Run to run z3 on your SMTLIB formulas")
(declare-sort Univ)
(declare-fun R (Univ Univ) Bool)
(assert (forall ((x Univ) (y Univ)) (R x y)))
(check-sat)



"""

from nltk.sem.logic import Expression  # type: ignore
from z3 import parse_smt2_string, SimpleSolver, unsat  # type: ignore

from .logic_renderer import render_expression, UNIVERSAL_TYPE
from .logic import Syntax, get_arities, get_propositional_variables


def _SMT_preamble(
    plchd_substitutions: list[list[str]],
    propositional_variables: list[str],
    predicate_arities: dict[str, int],
) -> str:
    """SMT preamble to define propositional variables, constants and predicates."""
    smtlib_lines = []
    # define unversal type
    smtlib_lines.append(f"(declare-sort {UNIVERSAL_TYPE})")

    # define propositional variables
    for k, v in dict(plchd_substitutions).items():
        if k in propositional_variables:
            smtlib_lines.append(f"(declare-fun {k} () Bool) ;; {v}")

    # define constants
    for k, v in dict(plchd_substitutions).items():
        if k not in propositional_variables and k.islower():
            smtlib_lines.append(f"(declare-const {k} {UNIVERSAL_TYPE}) ;; {v}")

    # define predicates
    for k, v in dict(plchd_substitutions).items():
        if predicate_arities.get(k,0) > 0:
            arity = " ".join([UNIVERSAL_TYPE] * predicate_arities[k])
            smtlib_lines.append(f"(declare-fun {k} ({arity}) Bool) ;; {v}")


    return "\n".join(smtlib_lines)


def _SMT_program(
    premises_formalized_nltk: dict[int, Expression],
    conclusion_formalized_nltk: dict[int, Expression],
    intermediary_conclusions_formalized_nltk: dict[int, Expression] | None = None,
    inference_tree: dict[int, list[int]] | None = None,
) -> str:
    """SMT program to test validity of inferences.
    uses propositional_variables to decide whether lower-char var is
    a propositional variable.
    """

    if (
        intermediary_conclusions_formalized_nltk is None
        or inference_tree is None
    ):
        assert (
            intermediary_conclusions_formalized_nltk is None
            and inference_tree is None
        )

    formulae_z3 = {}
    for i, e in premises_formalized_nltk.items():
        assert i not in formulae_z3
        formulae_z3[i] = render_expression(e, Syntax.Z3)
    for i, e in conclusion_formalized_nltk.items():
        assert i not in formulae_z3
        formulae_z3[i] = render_expression(e, Syntax.Z3)
    if intermediary_conclusions_formalized_nltk:
        for i, e in intermediary_conclusions_formalized_nltk.items():
            assert i not in formulae_z3
            formulae_z3[i] = render_expression(e, Syntax.Z3)

    smtlib_lines = []
    for i, _ in premises_formalized_nltk.items():
        smtlib_lines.append(
            f"(define-fun premise{i} () Bool {formulae_z3[i]})"
        )
    if intermediary_conclusions_formalized_nltk:
        for i, _ in intermediary_conclusions_formalized_nltk.items():
            smtlib_lines.append(
                f"(define-fun conclusion{i} () Bool {formulae_z3[i]})"
            )
    for i, _ in conclusion_formalized_nltk.items():
        smtlib_lines.append(
            f"(define-fun conclusion{i} () Bool {formulae_z3[i]})"
        )

    if inference_tree is None:
        smtlib_lines.append(
            f"(define-fun argument () Bool (=> (and {' '.join([f'premise{i}' for i in premises_formalized_nltk.keys()])}) conclusion{list(conclusion_formalized_nltk.keys())[0]}))"
        )
        smtlib_lines.append("(assert (not argument))")
        smtlib_lines.append("(check-sat)")
    else:
        for conclusion_ref, premises_refs in inference_tree.items():
            local_premises: list[str] = []
            for ref in premises_refs:
                if ref in [i for i in premises_formalized_nltk.keys()]:
                    local_premises.append(f"premise{ref}")
                else:
                    local_premises.append(f"conclusion{ref}")
            smtlib_lines.append("(push)")
            smtlib_lines.append(
                f"(define-fun subargument{conclusion_ref} () Bool (=> (and {' '.join(local_premises)}) conclusion{conclusion_ref}))"
            )
            smtlib_lines.append(f"(assert (not subargument{conclusion_ref}))")
            smtlib_lines.append(f'(echo "Check validity of inference to conclusion ({conclusion_ref}):")')
            smtlib_lines.append("(check-sat)")
            smtlib_lines.append("(pop)")

    return "\n".join(smtlib_lines)


def SMT_program_local(
    premises_formalized_nltk: dict[int, Expression],
    intermediary_conclusions_formalized_nltk: dict[int, Expression],
    conclusion_formalized_nltk: dict[int, Expression],
    inference_tree: dict[int, list[int]],
    plchd_substitutions: list[list[str]],
) -> str:
    """SMT program to test validity of local inferences."""
    predicate_arities = {}
    propositional_variables = []
    for expr in premises_formalized_nltk:
        predicate_arities.update(get_arities(expr))
        propositional_variables.extend(get_propositional_variables(expr))
    propositional_variables = list(set(propositional_variables))

    preamble = _SMT_preamble(
        plchd_substitutions=plchd_substitutions,
        propositional_variables=propositional_variables,
        predicate_arities=predicate_arities,
    )
    snippet = _SMT_program(
        premises_formalized_nltk=premises_formalized_nltk,
        intermediary_conclusions_formalized_nltk=intermediary_conclusions_formalized_nltk,
        conclusion_formalized_nltk=conclusion_formalized_nltk,
        inference_tree=inference_tree,
    )
    return "\n".join([preamble, snippet])


def SMT_program_global(
    premises_formalized_nltk: dict[int, Expression],
    conclusion_formalized_nltk: dict[int, Expression],
    plchd_substitutions: list[list[str]],
) -> str:
    """SMT program for to test validity of global inference.
    fromm all premises to the conclusion."""
    predicate_arities = {}
    propositional_variables = []
    for expr in premises_formalized_nltk:
        predicate_arities.update(get_arities(expr))
        propositional_variables.extend(get_propositional_variables(expr))
    propositional_variables = list(set(propositional_variables))
    preamble = _SMT_preamble(
        plchd_substitutions=plchd_substitutions,
        propositional_variables=propositional_variables,
        predicate_arities=predicate_arities,
    )
    snippet = _SMT_program(
        premises_formalized_nltk=premises_formalized_nltk,
        conclusion_formalized_nltk=conclusion_formalized_nltk,
    )
    return "\n".join([preamble, snippet])


def check_validity_z3(
    premises_formalized_nltk: dict[int, Expression],
    conclusion_formalized_nltk: dict[int, Expression],
    plchd_substitutions: list[list[str]],
):
    """Executes the SMT-LIB code using Z3 solver."""
    smtlib_code = SMT_program_global(
        premises_formalized_nltk=premises_formalized_nltk,
        conclusion_formalized_nltk=conclusion_formalized_nltk,
        plchd_substitutions=plchd_substitutions,
    )
    solver = SimpleSolver()
    ast = parse_smt2_string(smtlib_code)
    solver.add(ast)
    valid = (solver.check() == unsat)
    return valid
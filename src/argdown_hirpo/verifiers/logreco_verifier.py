from nltk.sem.logic import Expression, NegatedExpression  # type: ignore
from pyargdown import (
    Argdown,
    Conclusion,
)

from .infreco_verifier import InfRecoVerifier
from argdown_hirpo.logic.fol_parser import FOLParser
from argdown_hirpo.logic.smtlib import check_validity_z3


DEFAULT_EVAL_DIMENSIONS_MAP = {
    "illformed_argument": [
        "has_pcs",
        "starts_with_premise",
        "ends_with_conclusion",
        "has_no_duplicate_pcs_labels",
    ],
    "missing_label_gist": [
        "has_label",
        "has_gist",
    ],
    "missing_inference_info": [
        "has_inference_data"
    ],
    "unknown_proposition_references": [
        "prop_refs_exist",
    ],  # in inference info
    "unused_propositions": [
        "uses_all_props",
    ],
    "disallowed_material": [
        "no_extra_propositions",
    ],  # more propositions
    "flawed_formalizations": [
        "has_flawless_formalizations"
    ],
    "invalid_inference": [
        "is_globally_deductively_valid",
        "is_locally_deductively_valid",
    ],
    "redundant_premises": [
        "all_premises_relevant"
    ],
    "inconsistent_premises": [
        "premises_consistent"
    ],
}

class LogRecoVerifier(InfRecoVerifier):
    """
    LogRecoVerifier is a specialized verifier for logical reconstructions
    with methods for checking formalizations and deductive validity
    """

    default_eval_dimensions_map = DEFAULT_EVAL_DIMENSIONS_MAP

    def __init__(
        self,
        argdown: Argdown,
        from_key: str = "from",
        formalization_key: str = "formalization",
        declarations_key: str = "declarations",
        argument_idx: int = 0,
    ):
        super().__init__(argdown, from_key=from_key, argument_idx=argument_idx)

        self.formalization_key = formalization_key
        self.declarations_key = declarations_key

        all_expressions, all_declarations, flawed_formalizations_error = (
            self._parse_formalizations()
        )
        self.all_expressions = all_expressions
        self.all_declarations = all_declarations
        self.flawed_formalizations_error = flawed_formalizations_error

    def _parse_formalizations(
        self,
    ) -> tuple[dict[str, Expression], dict[str, str], str | None]:
        """
        Parse the formalizations in the argdown snippet and check for errors.
        """
        all_expressions: dict[str, Expression] = {}
        all_declarations: dict[str, str] = {}

        if self.argument is None or not self.argument.pcs:
            return all_expressions, all_declarations, None

        msgs = []
        for pr in self.argument.pcs:
            prop = next(
                p for p in self.argdown.propositions if p.label == pr.proposition_label
            )
            if not prop.data:
                msgs.append(
                    f"Proposition ({pr.label}) lacks inline yaml data with formalization info."
                )
                continue

            formalization = prop.data.get(self.formalization_key)
            declarations = prop.data.get(self.declarations_key)

            if formalization is None:
                msgs.append(
                    f"Inline yaml of proposition ({pr.label}) lacks {self.formalization_key} key."
                )

            if declarations:
                if isinstance(declarations, dict):
                    for k, v in declarations.items():
                        if k in all_declarations:
                            msgs.append(
                                f"Duplicate declaration: Variable '{k}' in the inline yaml of proposition "
                                f"({pr.label}) has been declared before."
                            )
                        else:
                            all_declarations[k] = v
                else:
                    msgs.append(
                        f"'{self.declarations_key}' of proposition ({pr.label}) is not a dict."
                    )

            if formalization:
                try:
                    expr = FOLParser.parse(formalization)
                except ValueError as e:
                    expr = None
                    msgs.append(
                        f"Formalization {formalization} of proposition ({pr.label}) is not a well-formed "
                        f"first-order logic formula. NLTK parser error: {str(e)}"
                    )

                if expr:
                    all_expressions[pr.label] = expr

                    if declarations and isinstance(declarations, dict):
                        for k, _ in declarations.items():
                            if k not in [str(v) for v in expr.variables()]:
                                msgs.append(
                                    f"Variable '{k}' declared with proposition ({pr.label}) is not used "
                                    f"in the corresponding formalization '{formalization}'."
                                )

        for k, expr in all_expressions.items():
            for v in expr.variables():
                if str(v) not in all_declarations:
                    msgs.append(
                        f"Variable '{v}' in formalization of proposition ({k}) is not declared anywhere."
                    )

        flawed_formalizations_error = " ".join(msgs) if msgs else None

        return all_expressions, all_declarations, flawed_formalizations_error

    def has_flawless_formalizations(
        self,
    ) -> tuple[bool | None, str | None]:
        """
        Check if all formalizations are flawless.
        """
        if self.argument is None or not self.argument.pcs:
            return None, None

        if self.flawed_formalizations_error:
            return False, self.flawed_formalizations_error

        return True, None
    
    def is_globally_deductively_valid(self) -> tuple[bool | None, str | None]:
        """
        Check if the argument is deductively valid globally.
        """
        if self.argument is None or not self.argument.pcs:
            return None, None

        expr_premises: dict[str, Expression] = {}
        for pr in self.argument.pcs:
            if not isinstance(pr, Conclusion):
                expr = self.all_expressions.get(pr.label)
                if expr:
                    expr_premises[pr.label] = expr
        expr_conclusion: dict[str, Expression] = {}
        pr_last = self.argument.pcs[-1]
        if isinstance(pr_last, Conclusion):
            expr = self.all_expressions.get(pr_last.label)
            if expr:
                expr_conclusion[pr_last.label] = expr

        if not expr_premises or not expr_conclusion:
            return False, "Failed to evaluate global deductive validity due to missing or flawed formalizations."

        try:
            deductively_valid, smtcode = check_validity_z3(
                premises_formalized_nltk=expr_premises,
                conclusion_formalized_nltk=expr_conclusion,
                plchd_substitutions=[[k,v] for k,v in self.all_declarations.items()],
            )
            if not deductively_valid:
                return False, (
                    "According to the provided formalizations, the argument is not deductively valid. "
                    f"SMT2LIB program used to check validity:\n {smtcode}\n"
                )
        except Exception as e:
            return False, f"Failed to evaluate global deductive validity with SMT2LIB/z3: {e}."

        return True, None

    
    def is_locally_deductively_valid(self) -> tuple[bool | None, str | None]:
        """
        Check if the argument is deductively valid locally, i.e. for each subargument
        """
        if self.argument is None or not self.argument.pcs:
            return None, None

        msgs = []
        for c in self.argument.pcs:
            if isinstance(c, Conclusion):
                expr_premises = {}
                for label in c.inference_data.get(self.from_key, []):
                    expr = self.all_expressions.get(label)
                    if expr:
                        expr_premises[label] = expr
                expr_conclusion = {}
                expr = self.all_expressions.get(c.label)
                if expr:
                    expr_conclusion[c.label] = expr

                if not expr_premises or not expr_conclusion:
                    msgs.append(
                        f"Failed to evaluate deductive validity of sub-inference to ({c.label}) "
                        "due to missing or flawed formalizations / inference info."
                    )
                else:
                    try:
                        deductively_valid, smtcode = check_validity_z3(
                            premises_formalized_nltk=expr_premises,
                            conclusion_formalized_nltk=expr_conclusion,
                            plchd_substitutions=[[k,v] for k,v in self.all_declarations.items()],
                        )
                        if not deductively_valid:
                            msgs.append(
                                "According to the provided formalizations and inference info, the sub-inference "
                                f"to conclusion ({c.label}) is not deductively valid. "
                                f"SMT2LIB program used to check validity of this subargument:\n {smtcode}\n"
                            )
                    except Exception as e:
                        msgs.append(
                            f"Failed to evaluate deductive validity of sub-inference to ({c.label}) "
                            f"with SMT2LIB/z3: {e}."
                        )

        if msgs:
            return False, "\n".join(msgs)

        return True, None

    def all_premises_relevant(self) -> tuple[bool | None, str | None]:
        """
        Check if the final conclusion follows from a real subset of the premises.
        """
        print("Checking relevance of premises...")
        if self.argument is None or not self.argument.pcs:
            return None, None

        msgs = []
        expr_premises: dict[str, Expression] = {}
        for pr in self.argument.pcs:
            if not isinstance(pr, Conclusion):
                expr = self.all_expressions.get(pr.label)
                if expr:
                    expr_premises[pr.label] = expr
        expr_conclusion: dict[str, Expression] = {}
        pr_last = self.argument.pcs[-1]
        if isinstance(pr_last, Conclusion):
            expr = self.all_expressions.get(pr_last.label)
            if expr:
                expr_conclusion[pr_last.label] = expr

        if not expr_premises or not expr_conclusion:
            msgs.append("Failed to evaluate logical relevance of premises due to missing or flawed formalizations.")

        if len(expr_premises) == 1:
            return True, None  # implicitly asuming the conclusion is not a tautology

        for k in expr_premises.keys():
            subset = expr_premises.copy()
            subset.pop(k)
            try:
                deductively_valid, smtcode = check_validity_z3(
                    premises_formalized_nltk=subset,
                    conclusion_formalized_nltk=expr_conclusion,
                    plchd_substitutions=[[k,v] for k,v in self.all_declarations.items()],
                )
                
                if deductively_valid:
                    msgs.append(
                        f"According to the provided formalizations, premise ({k}) is not required to logically "
                        f"infer the final conclusion. SMT2LIB program used to check validity:\n {smtcode}\n"
                    )
            except Exception as e:
                msgs.append(f"Failed to evaluate relevance of premise ({k}) with SMT2LIB/z3: {e}.")

        if msgs:
            return False, "\n".join(msgs)

        return True, None


    def premises_consistent(self) -> tuple[bool | None, str | None]:
        """
        Check if the argument's premises are consistent.

        We are effectively testing if the premises {p1...pn} entail ¬p1, which is equivalent to
        testing if the premises are consistent.
        """
        if self.argument is None or not self.argument.pcs:
            return None, None

        expr_premises: dict[str, Expression] = {}
        for pr in self.argument.pcs:
            if not isinstance(pr, Conclusion):
                expr = self.all_expressions.get(pr.label)
                if expr:
                    expr_premises[pr.label] = expr

        if not expr_premises:
            return True, None

        _key = next(iter(expr_premises))
        _concl = NegatedExpression(expr_premises[_key])
        expr_conclusion: dict[str, Expression] = {f"{_key}_neg": _concl}

        try:
            deductively_valid, smtcode = check_validity_z3(
                premises_formalized_nltk=expr_premises,
                conclusion_formalized_nltk=expr_conclusion,
                plchd_substitutions=[[k,v] for k,v in self.all_declarations.items()],
            )
            if deductively_valid:
                return False, (
                    "According to the provided formalizations, the argument's premises are NOT logically consistent."
                )
        except Exception as e:
            return False, f"Failed to evaluate premises' concistency with SMT2LIB/z3: {e}."

        return True, None


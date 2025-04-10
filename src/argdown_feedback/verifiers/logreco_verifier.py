from typing import Dict
from nltk.sem.logic import Expression, NegatedExpression  # type: ignore
from pyargdown import (
    Argdown,
    Conclusion,
    DialecticalType,
    Valence,
)

from .infreco_verifier import InfRecoVerifier
from argdown_feedback.logic.fol_parser import FOLParser
from argdown_feedback.logic.smtlib import check_validity_z3


# Custom Type, maps argument.label to pcs.label to formalization
ExpressionsStoreT = Dict[str, Dict[str, Expression]]


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

    @classmethod
    def run_battery(  # type: ignore[override]
        cls, argdown: Argdown, eval_dimensions_map: dict[str, list[str]] | None = None
    ) -> tuple[dict[str, str], ExpressionsStoreT, dict[str, str]]:
        if eval_dimensions_map is None:
            eval_dimensions_map = cls.default_eval_dimensions_map
        eval_results: dict[str, str] = {k: "" for k in eval_dimensions_map}
        all_expressions: ExpressionsStoreT = {}
        all_declarations: dict[str, str] = {}
        for argument_idx, argument in enumerate(argdown.arguments):
            verifier = cls(argdown, argument_idx=argument_idx)
            all_expressions[argument.label if argument.label else "None"] = verifier.all_expressions.copy()
            all_declarations.update(verifier.all_declarations)
            for eval_dim, veri_fn_names in eval_dimensions_map.items():
                for veri_fn_name in veri_fn_names:
                    veri_fn = getattr(verifier, veri_fn_name)
                    check, msg = veri_fn()
                    if check is False:
                        msg = msg if msg else veri_fn_name
                        eval_results[eval_dim] = (
                            eval_results[eval_dim]
                            + f"Error in argument <{argument.label}>: {msg} "
                        )
        eval_results = {k: v.strip() for k, v in eval_results.items()}
        return eval_results, all_expressions, all_declarations

    @staticmethod
    def _has_formally_grounded_relations(
        argdown_reco: Argdown,
        all_expressions: ExpressionsStoreT,
        all_declarations: dict[str, str],
    ) -> tuple[bool | None, str | None]:
        
        # convenience dict
        all_expressions_proplabels: dict[str,Expression] = {}
        for argument in argdown_reco.arguments:
            if argument.label in all_expressions and argument.pcs:
                for pr in argument.pcs:
                    if pr.label in all_expressions[argument.label]:
                        all_expressions_proplabels[pr.proposition_label] = all_expressions[argument.label][pr.label]

        msgs = []
        for drel in argdown_reco.dialectical_relations:
            if (
                drel.source not in all_expressions_proplabels
                or drel.target not in all_expressions_proplabels
            ):
                continue  # we're so far ignoring props that don't figure in any argument
            if DialecticalType.AXIOMATIC in drel.dialectics:
                if drel.valence == Valence.SUPPORT:
                    deductively_valid, smtcode = check_validity_z3(
                        premises_formalized_nltk={"1": all_expressions_proplabels[drel.source]},
                        conclusion_formalized_nltk={"2": all_expressions_proplabels[drel.target]},
                        plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                    )
                    if not deductively_valid:
                        msgs.append(
                            f"According to the provided formalizations, proposition '{drel.source}' does "
                            f"not entail the supported proposition '{drel.target}'. (SMTLIB program used to check "
                            f"entailment:\n {smtcode})"
                        )
                elif drel.valence == Valence.ATTACK:
                    deductively_valid, smtcode = check_validity_z3(
                        premises_formalized_nltk={"1": all_expressions_proplabels[drel.source]},
                        conclusion_formalized_nltk={"2": NegatedExpression(all_expressions_proplabels[drel.target])},
                        plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                    )
                    if not deductively_valid:
                        msgs.append(
                            f"According to the provided formalizations, proposition '{drel.source}' does not "
                            f"entail the negation of the attacked proposition '{drel.target}'. (SMTLIB program used to check "
                            f"contradiction:\n {smtcode})"
                        )
                elif drel.valence == Valence.CONTRADICT:
                    deductively_valid_1, smtcode_1 = check_validity_z3(
                        premises_formalized_nltk={"1": all_expressions_proplabels[drel.source]},
                        conclusion_formalized_nltk={"2": NegatedExpression(all_expressions_proplabels[drel.target])},
                        plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                    )
                    deductively_valid_2, smtcode_2 = check_validity_z3(
                        premises_formalized_nltk={"1": all_expressions_proplabels[drel.target]},
                        conclusion_formalized_nltk={"2": NegatedExpression(all_expressions_proplabels[drel.source])},
                        plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                    )
                    deductively_valid = deductively_valid_1 and deductively_valid_2
                    if not deductively_valid:
                        msgs.append(
                            f"According to the provided formalizations, proposition '{drel.source}' is not the "
                            f"the negation of the proposition '{drel.target}', despite both being declared as "
                            f"contradictory. (SMTLIB programs used to check contradiction:\n{smtcode_1}\n-----\n{smtcode_2})"
                        )
        if msgs:
            return False, "\n".join(msgs)
        return True, None


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
                        if k in all_declarations and all_declarations[k] != v:
                            msgs.append(
                                f"Duplicate declaration: Variable '{k}' in the inline yaml of proposition "
                                f"({pr.label}) has been declared before and is inconsistent with the previous "
                                f"declaration '{all_declarations[k]}'."
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
                            if k not in [str(v) for v in expr.variables() | expr.predicates() | expr.constants()]:
                                msgs.append(
                                    f"Variable '{k}' declared with proposition ({pr.label}) is not used "
                                    f"in the corresponding formalization '{formalization}'."
                                )

        for k, expr in all_expressions.items():
            for v in expr.variables() | expr.predicates() | expr.constants():
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


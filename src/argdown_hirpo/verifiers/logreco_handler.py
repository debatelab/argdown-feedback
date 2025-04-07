from abc import abstractmethod
from typing import Dict, List, Optional, Tuple
import logging

from nltk.sem.logic import Expression, NegatedExpression  # type: ignore
from pyargdown import (
    ArgdownMultiDiGraph,
    Conclusion,
    DialecticalType,
    Valence,
)

from .verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
    VerificationResult,
)
from .base import BaseHandler, CompositeHandler
from .infreco_handler import InfRecoHandler
from argdown_hirpo.logic.fol_parser import FOLParser
from argdown_hirpo.logic.smtlib import check_validity_z3

# Custom Type, maps argument.label to pcs.label to formalization
ExpressionsStoreT = Dict[str, Dict[str, Expression]]


class LogRecoHandler(InfRecoHandler):
    """Base handler interface for evaluating logical argument reconstructions."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        from_key: str = "from",
        formalization_key: str = "formalization",
        declarations_key: str = "declarations",
    ):
        super().__init__(name, logger, from_key)
        self.from_key = from_key
        self.formalization_key = formalization_key
        self.declarations_key = declarations_key
        
    def cached_formalizations(
        self, vdata_id: str, request: VerificationRequest
    ) -> tuple[dict[str, Expression] | None, dict[str, str] | None]:
        """Return cached formalizations, if any, from the request."""
        vdata = next((d for d in request.verification_data if d.id == vdata_id), None)
        if vdata is None:
            self.logger.warning(f"Verification data with ID {vdata_id} not found in request.")
            return None, None
        if vdata.dtype != VerificationDType.argdown:
            self.logger.debug(f"Verification data with ID {vdata_id} is not of type argdown.")
            return None, None
        vr_details = next((r.details for r in request.results if vdata_id in r.verification_data_references), None)
        if vr_details is None:
            self.logger.debug(f"No verification result found for verification data with ID {vdata_id}.")
            return None, None
        return vr_details.get("all_expressions"), vr_details.get("all_declarations")


class WellFormedFormulasHandler(LogRecoHandler):
    """Parses and checks first-order logic formulas in argdown code snippets.
    Stores the artifacts in the verification result object (details)."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        """Parse first-order logic formulas in argdown code snippets."""
        if vdata.data is None:
            return None
        if not isinstance(vdata.data, ArgdownMultiDiGraph):
            raise TypeError("Internal error: vdata.data is not a ArgdownMultiDiGraph")

        argdown = vdata.data
        all_expressions: Dict[str, Expression] = {}  # proposition_label to Expression
        all_declarations: Dict[str, str] = {}
        msgs: list[str] = []

        for argument in argdown.arguments:
            for pr in argument.pcs:
                prop = next(
                    p for p in argdown.propositions if p.label == pr.proposition_label
                )
                if not prop.data:
                    msgs.append(
                        f"Proposition ({pr.label}) in argument <{argument.label}> lacks inline yaml data with formalization info."
                    )
                    continue
                formalization = prop.data.get(self.formalization_key)
                declarations = prop.data.get(self.declarations_key)
                if formalization is None:
                    msgs.append(
                        f"Inline yaml of proposition ({pr.label}) in argument <{argument.label}> lacks {self.formalization_key} key."
                    )
                if declarations:
                    if isinstance(declarations, dict):
                        for k, v in declarations.items():
                            if k in all_declarations and all_declarations[k] != v:
                                msgs.append(
                                    f"Duplicate declaration: Variable '{k}' in the inline yaml of proposition "
                                    f"({pr.label}) in argument <{argument.label}> has been declared before and is inconsistent with the previous "
                                    f"declaration '{all_declarations[k]}'."
                                )
                            else:
                                all_declarations[k] = v
                    else:
                        msgs.append(
                            f"'{self.declarations_key}' of proposition ({pr.label}) in argument <{argument.label}> is not a dict."
                        )
                if formalization:
                    try:
                        expr = FOLParser.parse(formalization)
                    except ValueError as e:
                        expr = None
                        msgs.append(
                            f"Formalization {formalization} of proposition ({pr.label}) in argument <{argument.label}> is not a well-formed "
                            f"first-order logic formula. NLTK parser error: {str(e)}"
                        )
                    if expr:
                        all_expressions[pr.proposition_label] = expr

                        if declarations and isinstance(declarations, dict):
                            for k, _ in declarations.items():
                                if k not in [str(v) for v in expr.variables()]:  # very hacky check here
                                    msgs.append(
                                        f"Variable '{k}' declared with proposition ({pr.label}) in argument <{argument.label}> is not used "
                                        f"in the corresponding formalization '{formalization}'."
                                    )
        # Check if all free variables have been declared
        for prop_label, expr in all_expressions.items():
            for v in expr.variables():
                if str(v) not in all_declarations:
                    msgs.append(
                        f"Variable '{v}' in formalization '{str(expr)}' of proposition [{prop_label}] is not declared anywhere."
                    )
        # Check if all declared variables are used in the formalization
        for k, v in all_declarations.items():
            if k not in [str(v) for v in all_expressions.keys()]:
                msgs.append(
                    f"Variable '{k}' declared is not used in any formalization."
                )

        vresult = VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=len(msgs) == 0,
            message=" - ".join(msgs) if msgs else None,
            details={
                "all_expressions": all_expressions,
                "all_declarations": all_declarations,
            },
        )

        return vresult





class GlobalDeductiveValidityHandler(LogRecoHandler):
    """Handler that checks if all arguments are globally deductively valid."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise TypeError("Internal error: Argdown is not a MultiDiGraph")
        
        all_expressions, all_declarations = self.cached_formalizations(vdata.id, ctx)        

        # Skip if there are formalization errors
        if not all_expressions or not all_declarations:
            return None

        msgs = []        
        for argument in argdown.arguments:
            if not argument.pcs:
                continue
                
            arg_label = f"<{argument.label}>" if argument.label else "<unlabeled argument>"
            
            expr_premises: Dict[str, Expression] = {}
            for pr in argument.pcs:
                if not isinstance(pr, Conclusion):
                    expr = all_expressions.get(pr.proposition_label)
                    if expr:
                        expr_premises[pr.label] = expr
                        
            expr_conclusion: Dict[str, Expression] = {}
            pr_last = argument.pcs[-1]
            if isinstance(pr_last, Conclusion):
                expr = all_expressions.get(pr_last.proposition_label)
                if expr:
                    expr_conclusion[pr_last.label] = expr

            if not expr_premises or not expr_conclusion:
                msgs.append(
                    f"In {arg_label}: Failed to evaluate global deductive validity due to missing or flawed formalizations."
                )
                continue

            try:
                deductively_valid, smtcode = check_validity_z3(
                    premises_formalized_nltk=expr_premises,
                    conclusion_formalized_nltk=expr_conclusion,
                    plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                )
                if not deductively_valid:
                    msgs.append(
                        f"In {arg_label}: According to the provided formalizations, the argument is not deductively valid. "
                        f"SMT2LIB program used to check validity:\n {smtcode}\n"
                    )
            except Exception as e:
                msgs.append(
                    f"In {arg_label}: Failed to evaluate global deductive validity with SMT2LIB/z3: {str(e)}."
                )
        
        if msgs:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message=" ".join(msgs),
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class LocalDeductiveValidityHandler(LogRecoHandler):
    """Handler that checks if all sub-arguments are locally deductively valid."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise TypeError("Internal error: Argdown is not a MultiDiGraph")
        
        all_expressions, all_declarations = self.cached_formalizations(vdata.id, ctx)
        
        # Skip if there are formalization errors
        if not all_expressions or not all_declarations:
            return None

        msgs = []        
        for argument in argdown.arguments:
            if not argument.pcs:
                continue
                
            arg_label = f"<{argument.label}>" if argument.label else "<unlabeled argument>"
            
            for c in argument.pcs:
                if isinstance(c, Conclusion):
                    expr_premises = {}
                    for label in c.inference_data.get(self.from_key, []):
                        proposition_label = next(
                            (pr.proposition_label for pr in argument.pcs if pr.label == label),
                            None,
                        )
                        expr = all_expressions.get(proposition_label) if proposition_label is not None else None
                        if expr:
                            expr_premises[label] = expr
                    
                    expr_conclusion = {}
                    expr = all_expressions.get(c.proposition_label)
                    if expr:
                        expr_conclusion[c.label] = expr

                    if not expr_premises or not expr_conclusion:
                        msgs.append(
                            f"In {arg_label}: Failed to evaluate deductive validity of sub-inference to ({c.label}) "
                            "due to missing or flawed formalizations / inference info."
                        )
                    else:
                        try:
                            deductively_valid, smtcode = check_validity_z3(
                                premises_formalized_nltk=expr_premises,
                                conclusion_formalized_nltk=expr_conclusion,
                                plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                            )
                            if not deductively_valid:
                                msgs.append(
                                    f"In {arg_label}: According to the provided formalizations and inference info, "
                                    f"the sub-inference to conclusion ({c.label}) is not deductively valid. "
                                    f"SMT2LIB program used to check validity of this subargument:\n {smtcode}\n"
                                )
                        except Exception as e:
                            msgs.append(
                                f"In {arg_label}: Failed to evaluate deductive validity of sub-inference to ({c.label}) "
                                f"with SMT2LIB/z3: {str(e)}."
                            )
        
        if msgs:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message=" ".join(msgs),
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class AllPremisesRelevantHandler(LogRecoHandler):
    """Handler that checks if all premises are relevant to the argument."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise TypeError("Internal error: Argdown is not a MultiDiGraph")
        
        all_expressions, all_declarations = self.cached_formalizations(vdata.id, ctx)
        # Skip if there are formalization errors
        if not all_expressions or not all_declarations:
            return None

        msgs = []        
        for argument in argdown.arguments:
            if not argument.pcs:
                continue
                
            arg_label = f"<{argument.label}>" if argument.label else "<unlabeled argument>"
            
            # Skip if there are formalization errors
            if not all_expressions:
                continue
            
            expr_premises: Dict[str, Expression] = {}
            for pr in argument.pcs:
                if not isinstance(pr, Conclusion):
                    expr = all_expressions.get(pr.proposition_label)
                    if expr:
                        expr_premises[pr.label] = expr
                        
            expr_conclusion: Dict[str, Expression] = {}
            pr_last = argument.pcs[-1]
            if isinstance(pr_last, Conclusion):
                expr = all_expressions.get(pr_last.proposition_label)
                if expr:
                    expr_conclusion[pr_last.label] = expr

            if not expr_premises or not expr_conclusion:
                msgs.append(
                    f"In {arg_label}: Failed to evaluate logical relevance of premises due to missing or flawed formalizations."
                )
                continue

            if len(expr_premises) == 1:
                continue  # implicitly assuming the conclusion is not a tautology

            for k in expr_premises.keys():
                subset = expr_premises.copy()
                subset.pop(k)
                try:
                    deductively_valid, smtcode = check_validity_z3(
                        premises_formalized_nltk=subset,
                        conclusion_formalized_nltk=expr_conclusion,
                        plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                    )
                    
                    if deductively_valid:
                        msgs.append(
                            f"In {arg_label}: According to the provided formalizations, premise ({k}) is not required "
                            f"to logically infer the final conclusion. SMT2LIB program used to check validity:\n {smtcode}\n"
                        )
                except Exception as e:
                    msgs.append(
                        f"In {arg_label}: Failed to evaluate relevance of premise ({k}) with SMT2LIB/z3: {str(e)}."
                    )
        
        if msgs:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message=" ".join(msgs),
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class PremisesConsistentHandler(LogRecoHandler):
    """Handler that checks if the premises are consistent."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise TypeError("Internal error: Argdown is not a MultiDiGraph")
        
        msgs = []

        all_expressions, all_declarations = self.cached_formalizations(vdata.id, ctx)
        # Skip if there are formalization errors
        if not all_expressions or not all_declarations:
            return None
        
        for argument in argdown.arguments:
            if not argument.pcs:
                continue
                
            arg_label = f"<{argument.label}>" if argument.label else "<unlabeled argument>"
            
            expr_premises: Dict[str, Expression] = {}
            for pr in argument.pcs:
                if not isinstance(pr, Conclusion):
                    expr = all_expressions.get(pr.proposition_label)
                    if expr:
                        expr_premises[pr.label] = expr

            if not expr_premises:
                continue

            try:
                _key = next(iter(expr_premises))
                _concl = NegatedExpression(expr_premises[_key])
                expr_conclusion: Dict[str, Expression] = {f"{_key}_neg": _concl}
                
                deductively_valid, smtcode = check_validity_z3(
                    premises_formalized_nltk=expr_premises,
                    conclusion_formalized_nltk=expr_conclusion,
                    plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                )
                if deductively_valid:
                    msgs.append(
                        f"In {arg_label}: According to the provided formalizations, the argument's premises are NOT logically consistent."
                    )
            except Exception as e:
                msgs.append(
                    f"In {arg_label}: Failed to evaluate premises' consistency with SMT2LIB/z3: {str(e)}."
                )
        
        if msgs:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message=" ".join(msgs),
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class FormallyGroundedRelationsHandler(LogRecoHandler):
    """Handler that checks if dialectical relations are formally grounded."""

    def evaluate(self, vdata: PrimaryVerificationData, ctx: VerificationRequest) -> VerificationResult | None:
        argdown = vdata.data
        if argdown is None:
            return None
        if not isinstance(argdown, ArgdownMultiDiGraph):
            raise TypeError("Internal error: Argdown is not a MultiDiGraph")
        
        # Skip if no dialectical relations
        if not argdown.dialectical_relations:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=True,
                message=None,
            )
        
        all_expressions, all_declarations = self.cached_formalizations(vdata.id, ctx)
        # Skip if there are formalization errors
        if not all_expressions or not all_declarations:
            return None
        
        # Check each dialectical relation
        msgs = []
        for drel in argdown.dialectical_relations:
            if (
                drel.source not in all_expressions
                or drel.target not in all_expressions
            ):
                continue  # Ignore props that don't figure in any argument
                
            if DialecticalType.AXIOMATIC in drel.dialectics:
                if drel.valence == Valence.SUPPORT:
                    try:
                        deductively_valid, smtcode = check_validity_z3(
                            premises_formalized_nltk={"1": all_expressions[drel.source]},
                            conclusion_formalized_nltk={"2": all_expressions[drel.target]},
                            plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                        )
                        if not deductively_valid:
                            msgs.append(
                                f"According to the provided formalizations, proposition '{drel.source}' does "
                                f"not entail the supported proposition '{drel.target}'. (SMTLIB program used to check "
                                f"entailment:\n {smtcode})"
                            )
                    except Exception as e:
                        msgs.append(f"Failed to check support relation {drel.source} -> {drel.target}: {str(e)}")
                        
                elif drel.valence == Valence.ATTACK:
                    try:
                        deductively_valid, smtcode = check_validity_z3(
                            premises_formalized_nltk={"1": all_expressions[drel.source]},
                            conclusion_formalized_nltk={"2": NegatedExpression(all_expressions[drel.target])},
                            plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                        )
                        if not deductively_valid:
                            msgs.append(
                                f"According to the provided formalizations, proposition '{drel.source}' does not "
                                f"entail the negation of the attacked proposition '{drel.target}'. (SMTLIB program used to check "
                                f"contradiction:\n {smtcode})"
                            )
                    except Exception as e:
                        msgs.append(f"Failed to check attack relation {drel.source} -> {drel.target}: {str(e)}")
                        
                elif drel.valence == Valence.CONTRADICT:
                    try:
                        deductively_valid_1, smtcode_1 = check_validity_z3(
                            premises_formalized_nltk={"1": all_expressions[drel.source]},
                            conclusion_formalized_nltk={"2": NegatedExpression(all_expressions[drel.target])},
                            plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                        )
                        deductively_valid_2, smtcode_2 = check_validity_z3(
                            premises_formalized_nltk={"1": all_expressions[drel.target]},
                            conclusion_formalized_nltk={"2": NegatedExpression(all_expressions[drel.source])},
                            plchd_substitutions=[[k,v] for k,v in all_declarations.items()],
                        )
                        deductively_valid = deductively_valid_1 and deductively_valid_2
                        if not deductively_valid:
                            msgs.append(
                                f"According to the provided formalizations, proposition '{drel.source}' is not "
                                f"the negation of the proposition '{drel.target}', despite both being declared as "
                                f"contradictory. (SMTLIB programs used to check contradiction:\n{smtcode_1}\n-----\n{smtcode_2})"
                            )
                    except Exception as e:
                        msgs.append(f"Failed to check contradiction relation {drel.source} <-> {drel.target}: {str(e)}")
        
        if msgs:
            return VerificationResult(
                verifier_id=self.name,
                verification_data_references=[vdata.id],
                is_valid=False,
                message=" ".join(msgs),
            )
            
        return VerificationResult(
            verifier_id=self.name,
            verification_data_references=[vdata.id],
            is_valid=True,
            message=None,
        )


class LogRecoCompositeHandler(CompositeHandler[LogRecoHandler]):
    """A composite handler that groups all logical reconstruction verification handlers together."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        handlers: List[LogRecoHandler] | None = None,
        from_key: str = "from",
        formalization_key: str = "formalization",
        declarations_key: str = "declarations",
    ):
        super().__init__(name, logger, handlers)
        
        # Initialize with default handlers if none provided
        if not handlers:
            self.handlers = [
                # Formalization handlers
                WellFormedFormulasHandler(
                    from_key=from_key,
                    formalization_key=formalization_key,
                    declarations_key=declarations_key
                ),
                
                # Deductive validity handlers
                GlobalDeductiveValidityHandler(
                    from_key=from_key,
                    formalization_key=formalization_key,
                    declarations_key=declarations_key
                ),
                LocalDeductiveValidityHandler(
                    from_key=from_key,
                    formalization_key=formalization_key,
                    declarations_key=declarations_key
                ),
                
                # Logical analysis handlers
                AllPremisesRelevantHandler(
                    from_key=from_key,
                    formalization_key=formalization_key,
                    declarations_key=declarations_key
                ),
                PremisesConsistentHandler(
                    from_key=from_key,
                    formalization_key=formalization_key,
                    declarations_key=declarations_key
                ),
                
                # Dialectical relation handlers
                FormallyGroundedRelationsHandler(
                    from_key=from_key,
                    formalization_key=formalization_key,
                    declarations_key=declarations_key
                ),
            ]

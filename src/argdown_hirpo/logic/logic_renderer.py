from abc import ABC, abstractmethod
import logging

from nltk.sem.logic import (  # type: ignore
    Expression,
    Variable,
    IndividualVariableExpression,
    FunctionVariableExpression,
    EqualityExpression,
    ApplicationExpression,
    ConstantExpression,
    AllExpression,
    ExistsExpression,
    NegatedExpression,
    AndExpression,
    OrExpression,
    ImpExpression,
    IffExpression,
)

from .logic import Syntax

UNIVERSAL_TYPE = "Universal"


class LogicRenderingMachine(ABC):

    @abstractmethod
    def equality(self, first: str, second: str) -> str:
        pass

    @abstractmethod
    def application(self, function: str, arguments: list[str]) -> str:
        pass

    @abstractmethod
    def exists(self, variables: list[Variable], term: str) -> str:
        pass

    @abstractmethod
    def all(self, variables: list[Variable], term: str) -> str:
        pass

    @abstractmethod
    def negated(self, term: str) -> str:
        pass

    @abstractmethod
    def conjunction(self, conjuncts: list[str]) -> str:
        pass

    @abstractmethod
    def disjunction(self, disjuncts: list[str]) -> str:
        pass

    @abstractmethod
    def implication(self, first: str, second: str) -> str:
        pass

    @abstractmethod
    def iff(self, first: str, second: str) -> str:
        pass

    def postprocess(self, rendered: str) -> str:
        # remove unnecessary parentheses
        while (
            rendered[0] == "(" and rendered[-1] == ")" and
            all(rendered[1:i].count("(") >= rendered[1:i].count(")") for i in range(1, len(rendered)-1))
        ):
            rendered = rendered[1:-1]

        return rendered


class LatexLogicRenderingMachine(LogicRenderingMachine):

    def equality(self, first: str, second: str) -> str:
        return f"{first} = {second}"

    def application(self, function: str, arguments: list[str]) -> str:
        return f"{function}({','.join(arguments)})"

    def exists(self, variables: list[Variable], term: str) -> str:
        quantifiers = " ".join([f"\\exists {v}" for v in variables])
        return f"{quantifiers}: {term}"

    def all(self, variables: list[Variable], term: str) -> str:
        quantifiers = " ".join([f"\\forall {v}" for v in variables])
        return f"{quantifiers}: {term}"

    def negated(self, term: str) -> str:
        return f"\\lnot {term}"

    def conjunction(self, conjuncts: list[str]) -> str:
        return "(" + " \\land ".join(conjuncts) + ")"

    def disjunction(self, disjuncts: list[str]) -> str:
        return "(" + " \\lor ".join(disjuncts) + ")"

    def implication(self, first: str, second: str) -> str:
        return f"({first} \\rightarrow {second})"

    def iff(self, first: str, second: str) -> str:
        return f"({first} \\leftrightarrow {second})"
    
    def postprocess(self, rendered: str) -> str:
        rendered = super().postprocess(rendered)
        rendered = rendered.replace(": \\exists", " \\exists")
        rendered = rendered.replace(": \\forall", " \\forall")
        return rendered


class Z3LogicRenderingMachine(LogicRenderingMachine):

    def __init__(self, universal_type: str = UNIVERSAL_TYPE) -> None:
        self.universal_type = universal_type

    def equality(self, first: str, second: str) -> str:
        return f"(= {first} {second})"

    def application(self, function: str, arguments: list[str]) -> str:
        return f"({function} {' '.join(arguments)})"

    def exists(self, variables: list[Variable], term: str) -> str:
        var_list = " ".join([f"({v} {self.universal_type})" for v in variables])
        return f"(exists ({var_list}) {term})"

    def all(self, variables: list[Variable], term: str) -> str:
        var_list = " ".join([f"({v} {self.universal_type})" for v in variables])
        return f"(forall ({var_list}) {term})"

    def negated(self, term: str) -> str:
        return f"(not {term})"

    def conjunction(self, conjuncts: list[str]) -> str:
        return f'(and {" ".join(conjuncts)})'

    def disjunction(self, disjuncts: list[str]) -> str:
        return f'(or {" ".join(disjuncts)})'

    def implication(self, first: str, second: str) -> str:
        return f"(=> {first} {second})"

    def iff(self, first: str, second: str) -> str:
        return f"(= {first} {second})"
    
    def postprocess(self, rendered: str) -> str:
        return rendered






def render_expression(expression: Expression, syntax: Syntax) -> str:
    """Render expression in given syntax."""

    def resolve_conjunction(expression: AndExpression) -> list[Expression]:
        """resolves conjunctions in the expression tree"""
        if isinstance(expression, AndExpression):
            return resolve_conjunction(expression.first) + resolve_conjunction(expression.second)
        return [expression]

    def resolve_disjunction(expression: OrExpression) -> list[Expression]:
        """resolves disjunctions in the expression tree"""
        if isinstance(expression, OrExpression):
            return resolve_disjunction(expression.first) + resolve_disjunction(expression.second)
        return [expression]
    
    def resolve_application(expression: ApplicationExpression) -> tuple[FunctionVariableExpression,list[IndividualVariableExpression]]:
        """resolves nested applications in the expression tree"""
        if isinstance(expression.function, ApplicationExpression):
            function, arguments = resolve_application(expression.function)
            return function, arguments + [expression.argument]
        return expression.function, [expression.argument]

    def resolve_all_expression(expression: AllExpression) -> tuple[list[Variable],Expression]:
        """resolves nested all-expressions in the expression tree"""
        if isinstance(expression.term, AllExpression):
            variables, term = resolve_all_expression(expression.term)
            return [expression.variable] + variables, term
        return [expression.variable], expression.term

    def resolve_exists_expression(expression: ExistsExpression) -> tuple[list[Variable],Expression]:
        """resolves nested all-expressions in the expression tree"""
        if isinstance(expression.term, ExistsExpression):
            variables, term = resolve_exists_expression(expression.term)
            return [expression.variable] + variables, term
        return [expression.variable], expression.term

    def render(expression: Expression, lrm: LogicRenderingMachine) -> str:
        """recursively renders the expression tree"""
        if isinstance(expression, (IndividualVariableExpression, FunctionVariableExpression)):
            return expression.variable.name
        if isinstance(expression, EqualityExpression):
            return lrm.equality(str(expression.first), str(expression.second))
        if isinstance(expression, ConstantExpression):
            return str(expression)
        if isinstance(expression, ApplicationExpression):
            function, arguments = resolve_application(expression)
            return lrm.application(render(function, lrm), [render(a, lrm) for a in arguments])
        if isinstance(expression, ExistsExpression):
            variables, term = resolve_exists_expression(expression)
            return lrm.exists([render(v, lrm) for v in variables], render(term, lrm))
        if isinstance(expression, AllExpression):
            variables, term = resolve_all_expression(expression)
            return lrm.all([render(v, lrm) for v in variables], render(term, lrm))
        if isinstance(expression, NegatedExpression):
            return lrm.negated(render(expression.term, lrm))
        if isinstance(expression, AndExpression):
            conjuncts = resolve_conjunction(expression)
            return lrm.conjunction([render(c, lrm) for c in conjuncts])
        if isinstance(expression, OrExpression):
            disjuncts = resolve_disjunction(expression)
            return lrm.disjunction([render(d, lrm) for d in disjuncts])
        if isinstance(expression, ImpExpression):
            return lrm.implication(render(expression.first, lrm), render(expression.second, lrm))
        if isinstance(expression, IffExpression):
            return lrm.iff(render(expression.first, lrm), render(expression.second, lrm))
        return str(expression)

    lrm: LogicRenderingMachine
    if syntax == Syntax.NLTK:
        return str(expression)
    elif syntax == Syntax.LATEX:
        lrm = LatexLogicRenderingMachine()
    elif syntax == Syntax.Z3:
        lrm = Z3LogicRenderingMachine()
    else:
        logging.getLogger().warning(f"Unsupported syntax: {syntax}. Defaulting to NLTK.")
        return str(expression)

    rendered = render(expression, lrm)
    rendered = lrm.postprocess(rendered)

    return rendered


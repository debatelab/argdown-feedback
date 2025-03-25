"""logic.py"""

import enum


from nltk.sem.logic import (  # type: ignore
    Expression,
    IndividualVariableExpression,
    EqualityExpression,
    ApplicationExpression,
    ConstantExpression,
)


class Syntax(enum.Enum):
    NLTK = "nltk"
    LATEX = "latex"
    Z3 = "z3"
    DEEPA2 = "deepa2"


def key_present_in_form(form: str, key: str) -> bool:
    """Check if the key is present in the form
    without making assumptions about the syntax format.
    """
    adj_chars = [" ", "(", ")", "[", "]", "{", "}", ":", ".", ">", "-", "&", "!"]
    present = any(
        (adj_char + key) in form or (key + adj_char) in form for adj_char in adj_chars
    )
    return present

def get_propositional_variables(expression: Expression) -> list[str]:
    """returns a list of propositional variables in the expression"""
    variables = []

    def visit(subexpression: Expression):
        """recursively visits the expression tree and analyse IndividualVariableExpressions"""
        if isinstance(subexpression, IndividualVariableExpression):
            variables.extend(
                [str(v) for v in subexpression.variables()]
            )
        elif not isinstance(
            subexpression,
            (ApplicationExpression, ConstantExpression, IndividualVariableExpression),
        ):
            subexpression.visit(visit, lambda x: None)

    visit(expression)
    return list(set(variables))


def get_arities(expression: Expression) -> dict[str, int]:
    """returns a dictionary of variables and their arity"""
    variables = {k: 0 for k in expression.variables()}

    def visit(subexpression: Expression):
        """recursively visits the expression tree and analyse ApplicationExpressions"""
        if isinstance(subexpression, ApplicationExpression):
            arity = 1
            while isinstance(subexpression.function, ApplicationExpression):
                arity += 1
                subexpression = subexpression.function
            if (
                variables[subexpression.function.variable] > 0
                and variables[subexpression.function.variable] != arity
            ):
                raise ValueError(
                    f"Inconsistent arity of variables in expression {expression}"
                )
            variables[subexpression.function.variable] = arity
        elif not isinstance(
            subexpression,
            (ConstantExpression, IndividualVariableExpression, EqualityExpression),
        ):
            subexpression.visit(visit, lambda x: None)

    visit(expression)
    return {str(k): v for k, v in variables.items()}

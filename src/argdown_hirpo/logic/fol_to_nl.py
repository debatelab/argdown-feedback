

from nltk.sem.logic import (  # type: ignore
    Expression,
    NegatedExpression,
    AndExpression,
    OrExpression,
    ApplicationExpression,
    EqualityExpression,
    FunctionVariableExpression,
    ConstantExpression,
    ExistsExpression,
    AllExpression,
    ImpExpression,
    IffExpression,
)

from string import Formatter


class FOL2NLTranslator:

    @classmethod
    def _get_placeholders(cls, scheme: str) -> list[str]:
      return list(set(key for _, key, _, _ in Formatter().parse(scheme) if key))


    @classmethod
    def translate_to_nl_sentence(cls, expression: Expression, declarations: dict[str, str]) -> str:
        """translate a logic expression to natural language sentence"""
        scheme = cls.translate_to_nl_scheme(expression)
        substitutions = declarations.copy()
        for placeholder in cls._get_placeholders(scheme):
            if placeholder not in substitutions:
                substitutions[placeholder] = placeholder
        print(substitutions)
        return scheme.format(**substitutions)


    @classmethod
    def translate_to_nl_scheme(cls, expression: Expression) -> str:
        """translate a logic expression to natural language scheme"""

        def is_negated_atomic_expression(expression: Expression) -> bool:
            """checks if expression is negated atomic expression (unary predicate)"""
            if isinstance(expression, NegatedExpression):
                if isinstance(expression.term, ApplicationExpression):
                    return isinstance(expression.term.function, FunctionVariableExpression) 
            return False

        if isinstance(expression, ApplicationExpression):
            if isinstance(expression.function, FunctionVariableExpression):
                # unary predicate
                return "{%s} is {%s}" % (
                    expression.argument.variable.name, expression.function.variable.name
                )
            if isinstance(expression.function, ApplicationExpression):
                # n-ary predicate
                objects: list[str] = []
                subexpression = expression
                while isinstance(subexpression.function, ApplicationExpression):
                    objects = [subexpression.argument.variable.name] + objects
                    subexpression = subexpression.function
                if isinstance(subexpression.function, FunctionVariableExpression):
                    return "{%s} stands in relation {%s} to %s" % (
                        subexpression.argument.variable.name,
                        subexpression.function.variable.name,
                        ", ".join([f"{{{o}}}" for o in objects]),
                    )
        if isinstance(expression, ConstantExpression):
            return "{%s}" % str(expression)
        if isinstance(expression, EqualityExpression):
            return "{%s} is identical with {%s}" % (
                expression.first.variable.name,
                expression.second.variable.name,
            )
        if isinstance(expression, NegatedExpression):
            if isinstance(expression.term, ApplicationExpression):  # negate atomic sentence
                if isinstance(expression.term.function, FunctionVariableExpression):
                    return "{%s} is not {%s}" % (
                        expression.term.argument.variable.name, expression.term.function.variable.name
                    )
            return "it is false that %s" % (cls.translate_to_nl_scheme(expression.negate()))
        if isinstance(expression, AndExpression):
            junctor = " and "
            if (
                isinstance(expression.first, (OrExpression,NegatedExpression)) or
                isinstance(expression.second, (OrExpression))
            ) and not is_negated_atomic_expression(expression.first):
                junctor = " and also "
            return expression.visit(
                cls.translate_to_nl_scheme,
                junctor.join
            )
        if isinstance(expression, OrExpression):
            junctor = " or "
            if (
                isinstance(expression.first, (AndExpression,NegatedExpression)) or
                isinstance(expression.second, (AndExpression))
            ) and not is_negated_atomic_expression(expression.first):
                junctor = " or else "
            return expression.visit(
                cls.translate_to_nl_scheme,
                junctor.join
            )
        if isinstance(expression, ImpExpression):
            if isinstance(expression.first, OrExpression):
                return "if %s, or if %s, then %s" % (
                    cls.translate_to_nl_scheme(expression.first.first),
                    cls.translate_to_nl_scheme(expression.first.second),
                    cls.translate_to_nl_scheme(expression.second),
                )
            if isinstance(expression.first, AndExpression):
                return "if %s, and if %s, then %s" % (
                    cls.translate_to_nl_scheme(expression.first.first),
                    cls.translate_to_nl_scheme(expression.first.second),
                    cls.translate_to_nl_scheme(expression.second),
                )
            return expression.visit(
                cls.translate_to_nl_scheme,
                lambda x: f"if {x[0]}, then {x[1]}"
            )
        if isinstance(expression, IffExpression):
            return expression.visit(
                cls.translate_to_nl_scheme,
                lambda x: f"if and only if {x[0]}, then {x[1]}"
            )
        if isinstance(expression, ExistsExpression):
            return "there exists a {%s} such that %s" % (
                expression.variable.name, cls.translate_to_nl_scheme(expression.term)
            )
        if isinstance(expression, AllExpression):
            return "for every {%s} it holds that %s" % (
                expression.variable.name, cls.translate_to_nl_scheme(expression.term)
            )
        return str(expression)  
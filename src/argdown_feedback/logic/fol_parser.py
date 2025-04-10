"""Parser for first-order logic formulae using NLTK"""

from nltk.sem.logic import (  # type: ignore
    Expression,
    LogicalExpressionException,
)


class FOLParser:
    """parser methods for first-order-logic formulae
    based on NLTK parser"""

    @staticmethod
    def parse(form: str) -> Expression:
        """parses string formalizationsas NLTK first-order-logic formula"""
        try:
            return Expression.fromstring(form)
        except LogicalExpressionException as e:
            raise ValueError(
                f"Invalid formula: {form}. Error: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Unexpected error while parsing formula: {form}. Error: {e}"
            ) from e                


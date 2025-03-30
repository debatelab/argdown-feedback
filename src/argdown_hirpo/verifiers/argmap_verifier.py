from textwrap import shorten

from pyargdown import (
    Argdown,
    ArgdownMultiDiGraph,
    Argument,
    Proposition,
    Conclusion,
    DialecticalType,
)
from pyargdown.parser.base import ArgdownParser

from .base import BaseArgdownVerifier


class ArgMapVerifier(BaseArgdownVerifier):
    """
    ArgMapVerifier is a specialized verifier for informal argument maps.
    """

    def __init__(self, argdown: Argdown):
        super().__init__(argdown)
        if not isinstance(self.argdown, ArgdownMultiDiGraph):
            raise TypeError("argdown must be of type ArgdownMultiDiGraph")

    def has_complete_claims(self) -> tuple[bool | None, str | None]:
        """
        Check if all claims have labels and are not empty.
        """
        incomplete_claims: list[str] = []
        for claim in self.argdown.propositions:
            assert isinstance(claim, Proposition)
            if ArgdownParser.is_unlabeled(claim):
                if not claim.texts or not claim.texts[0]:
                    incomplete_claims.append("Empty claim")
                else:
                    incomplete_claims.append(shorten(claim.texts[0], width=40))
        if incomplete_claims:
            return (
                False,
                f"Missing labels for nodes: {', '.join(incomplete_claims)}",
            )
        return True, None

    def has_no_duplicate_labels(self) -> tuple[bool | None, str | None]:
        """
        Check for duplicate labels in claims and arguments.
        """
        duplicate_labels: list[str] = []
        for claim in self.argdown.propositions:
            if len(claim.texts) > 1 and claim.label:
                duplicate_labels.append(claim.label)
        for argument in self.argdown.arguments:
            if len(argument.gists) > 1 and argument.label:
                duplicate_labels.append(argument.label)
        if duplicate_labels:
            return (
                False,
                f"Duplicate labels: {', '.join(duplicate_labels)}",
            )
        return True, None

    def has_no_pcs(self) -> tuple[bool | None, str | None]:
        """
        Check if any argument has a premise-conclusion structure.
        """
        arguments_with_pcs = [
            argument for argument in self.argdown.arguments if argument.pcs
        ]
        if arguments_with_pcs:
            arg_labels = [
                f"<{argument.label}>" if argument.label else "<unlabeled_argument>"
                for argument in arguments_with_pcs
            ]
            return (
                False,
                f"Found detailed reconstruction of individual argument(s) {', '.join(arg_labels)} as premise-conclusion-structures.",
            )
        return True, None

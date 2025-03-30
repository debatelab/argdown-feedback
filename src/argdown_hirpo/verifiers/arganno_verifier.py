from difflib import unified_diff
from textwrap import shorten
from typing import Sequence

from bs4 import BeautifulSoup


from .base import Verifier


class ArgAnnoVerifier(Verifier):
    """
    ArgAnnoVerifier is a specialized verifier for argumentative text annotations.
    """

    def __init__(self, sources: str, soup: BeautifulSoup):
        self.sources = sources
        self.soup = soup
        if not isinstance(self.soup, BeautifulSoup):
            raise TypeError("soup must be of type BeautifulSoup")

    def source_text_not_altered(self) -> tuple[bool | None, str | None]:
        """
        Check that the source text has not been altered.
        """
        lines_o = " ".join(self.sources.split()).splitlines(keepends=True)
        lines_a = " ".join(self.soup.get_text().split()).splitlines(keepends=True)
        lines_o = [line for line in lines_o if line.strip()]
        lines_a = [line for line in lines_a if line.strip()]

        diff = list(unified_diff(lines_o, lines_a, n=0))
        if diff:
            return (
                False,
                "Source text was altered. Diff:\n" + "".join(diff),
            )
        return True, None

    def no_nested_proposition_annotations(self) -> tuple[bool | None, str | None]:
        """
        Check for nested proposition annotations.
        """
        nested_props = [
            f"'{shorten(str(proposition), 256)}'"
            for proposition in self.soup.find_all("proposition")
            if proposition.find_all("proposition")  # type: ignore
        ]
        if nested_props:
            return (
                False,
                "Nested annotations in proposition(s) {}".format(", ".join(nested_props)),
            )
        return True, None

    def every_proposition_has_id(self) -> tuple[bool | None, str | None]:
        """
        Check that every proposition has an id.
        """
        props_without_id = [
            f"'{shorten(str(proposition), 64)}'"
            for proposition in self.soup.find_all("proposition")
            if not proposition.get("id")  # type: ignore
        ]
        if props_without_id:
            return (
                False,
                "Missing id in proposition(s) {}".format(", ".join(props_without_id)),
            )
        return True, None

    def every_proposition_has_unique_id(self) -> tuple[bool | None, str | None]:
        """
        Check that every proposition has a unique id.
        """
        ids = [
            str(proposition.get("id"))  # type: ignore
            for proposition in self.soup.find_all("proposition")
        ]
        duplicates = {id for id in ids if ids.count(id) > 1}
        if duplicates:
            return (
                False,
                "Duplicate ids: {}".format(", ".join(duplicates)),
            )
        return True, None
    
    def every_support_reference_is_valid(self) -> tuple[bool | None, str | None]:
        """
        Check that every "supports" reference is a valid id.
        """
        ids = [
            proposition.get("id")  # type: ignore
            for proposition in self.soup.find_all("proposition")
        ]
        msgs = []
        for proposition in self.soup.find_all("proposition"):
            for support in proposition.get("supports", []): # type: ignore
                if support not in ids:
                    msgs.append(
                        f"Supported proposition with id '{support}' in proposition '{shorten(str(proposition), 64)}' does not exist."
                    )
        if msgs:
            return False, " ".join(msgs)
        return True, None
    
    def every_attack_reference_is_valid(self) -> tuple[bool | None, str | None]:
        """
        Check that every "attacks" reference is a valid id.
        """
        ids = [
            proposition.get("id")  # type: ignore
            for proposition in self.soup.find_all("proposition")
        ]
        msgs = []
        for proposition in self.soup.find_all("proposition"):
            for attack in proposition.get("attacks", []):  # type: ignore
                if attack not in ids:
                    msgs.append(
                        f"Attacked proposition with id '{attack}' in proposition '{shorten(str(proposition), 64)}' does not exist."
                    )
        if msgs:
            return False, " ".join(msgs)
        return True, None
    
    def no_unknown_attributes(self) -> tuple[bool | None, str | None]:
        """
        Check for unknown attributes in propositions.
        """
        unknown_attrs = []
        for proposition in self.soup.find_all("proposition"):
            for attr in proposition.attrs:  # type: ignore
                if attr not in {
                    "id",
                    "supports",
                    "attacks",
                    "argument_label",
                    "ref_reco_label",
                }:
                    unknown_attrs.append(
                        f"Unknown attribute '{attr}' in proposition '{shorten(str(proposition), 64)}'"
                    )
        if unknown_attrs:
            return False, " ".join(unknown_attrs)
        return True, None
    
    def no_unknown_elements(self) -> tuple[bool | None, str | None]:
        """
        Check for unknown elements in the soup.
        """
        unknown_elements = []
        for element in self.soup.find_all():
            element_name = element.name  # type: ignore
            if element_name not in {"proposition"}:
                unknown_elements.append(
                    f"Unknown element '{element_name}' at '{shorten(str(element), 64)}'"
                )
        if unknown_elements:
            return False, " ".join(unknown_elements)
        return True, None
    
    def has_legal_argument_labels(self, legal_labels: Sequence[str]) -> tuple[bool | None, str | None]:
        """
        Check that every argument label is one of the legal labels.
        """
        illegal_labels = []
        for proposition in self.soup.find_all("proposition"):
            if proposition.get("argument_label") not in legal_labels:  # type: ignore
                illegal_labels.append(
                    f"Illegal argument label '{proposition.get('argument_label')}' "  # type: ignore
                    f"in proposition '{shorten(str(proposition), 64)}'"
                )
        if illegal_labels:
            return False, " ".join(illegal_labels)
        return True, None
    
    def has_legal_ref_reco_labels(self, legal_labels: Sequence[str]) -> tuple[bool | None, str | None]:
        """
        Check that every ref_reco label is one of the legal labels.
        """
        illegal_labels = []
        for proposition in self.soup.find_all("proposition"):
            if proposition.get("ref_reco_label") not in legal_labels:  # type: ignore
                illegal_labels.append(
                    f"Illegal ref_reco label '{proposition.get('ref_reco_label')}' "  # type: ignore
                    f"in proposition '{shorten(str(proposition), 64)}'"
                )
        if illegal_labels:
            return False, " ".join(illegal_labels)
        return True, None
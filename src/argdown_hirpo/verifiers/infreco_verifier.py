from pyargdown import (
    Argdown,
    ArgdownMultiDiGraph,
    Argument,
    Conclusion,
    DialecticalType,
)
from pyargdown.parser.base import ArgdownParser

from .base import BaseArgdownVerifier


class InfRecoVerifier(BaseArgdownVerifier):
    """
    InfRecoVerifier is a specialized verifier for informal reconstructions.
    """

    def __init__(self, argdown: Argdown, from_key: str = "from"):
        super().__init__(argdown)
        # Additional initialization for InfReco can be added here
        if not isinstance(self.argdown, ArgdownMultiDiGraph):
            raise TypeError("argdown must be of type ArgdownMultiDiGraph")

        self.argument: Argument | None = None
        if len(argdown.arguments) == 1:
            self.argument = argdown.arguments[0]

        self.from_key = from_key

    def has_unique_argument(self) -> tuple[bool | None, str | None]:
        if len(self.argdown.arguments) > 1:
            return False, "More than one argument in argdown snippet."
        if len(self.argdown.arguments) == 0:
            return False, "No argument in argdown snippet."
        if self.argument is None:
            return None, None
        return True, None

    def has_pcs(self) -> tuple[bool | None, str | None]:
        if self.argument is None:
            return None, None
        if not self.argument.pcs:
            return (
                False,
                "Argument lacks premise conclusion structure, i.e., is not reconstructed in standard form.",
            )
        return True, None

    def starts_with_premise(self) -> tuple[bool | None, str | None]:
        if self.argument is None or not self.argument.pcs:
            return None, None
        if isinstance(self.argument.pcs[0], Conclusion):
            return False, "Argument does not start with a premise."
        return True, None

    def ends_with_conclusion(self) -> tuple[bool | None, str | None]:
        if self.argument is None or not self.argument.pcs:
            return None, None
        if not isinstance(self.argument.pcs[-1], Conclusion):
            return False, "Argument does not end with a conclusion."
        return True, None

    def has_not_multiple_gists(self) -> tuple[bool | None, str | None]:
        if self.argument is None:
            return None, None
        if len(self.argument.gists) > 1:
            return False, "Argument has more than one gist."
        return True, None

    def has_no_duplicate_pcs_labels(self) -> tuple[bool | None, str | None]:
        if self.argument is None or not self.argument.pcs:
            return None, None
        pcs_labels = [p.label for p in self.argument.pcs]
        duplicates = [label for label in pcs_labels if pcs_labels.count(label) > 1]
        if duplicates:
            return (
                False,
                f"Duplicate labels in the argument's standard form: {', '.join([f'({lbl})' for lbl in duplicates])}.",
            )
        return True, None

    def has_label(self) -> tuple[bool | None, str | None]:
        if self.argument is None:
            return None, None
        if ArgdownParser.is_unlabeled(self.argument):
            return False, "Argument lacks a label / title."
        return True, None

    def has_gist(self) -> tuple[bool | None, str | None]:
        if self.argument is None:
            return None, None
        if not self.argument.gists:
            return False, "Argument lacks a gist / summary."
        return True, None

    def has_inference_data(self) -> tuple[bool | None, str | None]:
        if self.argument is None or not self.argument.pcs:
            return None, None
        msgs = []
        for c in self.argument.pcs:
            if not isinstance(c, Conclusion):
                continue
            inf_data = c.inference_data
            if not inf_data:
                msgs.append(f"Conclusion {c.label} lacks yaml inference information.")
            else:
                from_list = inf_data.get(self.from_key)
                if from_list is None:
                    msgs.append(
                        f"Conclusion {c.label} inference information lacks '{self.from_key}' key."
                    )
                elif not isinstance(from_list, list):
                    msgs.append(
                        f"Conclusion {c.label} inference information '{self.from_key}' value is not a list."
                    )
                elif len(from_list) == 0:
                    msgs.append(
                        f"Conclusion {c.label} inference information '{self.from_key}' value is empty."
                    )
        if msgs:
            return False, " ".join(msgs)
        return True, None

    def uses_all_props(self) -> tuple[bool | None, str | None]:
        if self.argument is None or not self.argument.pcs:
            return None, None

        used_labels = set()
        for c in self.argument.pcs:
            if isinstance(c, Conclusion):
                used_labels.update(c.inference_data.get(self.from_key, []))
        unused_props = [f"({p.label})" for p in self.argument.pcs[:-1] if p.label not in used_labels]
        if unused_props:
            return (
                False,
                f"Some propositions are not explicitly used in any of the argument's inferences: {', '.join(unused_props)}.",
            )
        return True, None



    def prop_refs_exist(self) -> tuple[bool | None, str | None]:
        if self.argument is None or not self.argument.pcs:
            return None, None
        msgs = []
        for enum, c in enumerate(self.argument.pcs):
            if isinstance(c, Conclusion):
                inf_data = c.inference_data
                from_list = inf_data.get(self.from_key, [])
                if isinstance(from_list, list):
                    for ref in from_list:
                        if str(ref) not in [p.label for p in self.argument.pcs[:enum]]:
                            msgs.append(
                                f"Item '{ref}' in inference information of conclusion {c.label} does "
                                "not refer to a previously introduced premise or conclusion."
                            )
        if msgs:
            return False, " ".join(msgs)
        return True, None

    def no_extra_propositions(self) -> tuple[bool | None, str | None]:
        if self.argument is None:
            return None, None
        if self.argument.pcs and len(self.argdown.propositions) > len(
            self.argument.pcs
        ):
            return (
                False,
                "Argdown snippet contains propositions other than the ones in the argument.",
            )
        if not self.argument.pcs and len(self.argdown.propositions) > 0:
            return False, "Argdown snippet contains propositions outside the argument."
        return True, None

    def only_grounded_dialectical_relations(self) -> tuple[bool | None, str | None]:
        if any(
            set(d.dialectics) != {DialecticalType.GROUNDED}
            for d in self.argdown.dialectical_relations
        ):
            return False, "Argdown snippet defines dialectical relations."
        return True, None

    def no_prop_inline_data(self) -> tuple[bool | None, str | None]:
        if any(prop.data for prop in self.argdown.propositions):
            return False, "Some propositions contain yaml inline data."
        return True, None

    def no_arg_inline_data(self) -> tuple[bool | None, str | None]:
        if any(arg.data for arg in self.argdown.arguments):
            return False, "Some arguments contain yaml inline data."
        return True, None

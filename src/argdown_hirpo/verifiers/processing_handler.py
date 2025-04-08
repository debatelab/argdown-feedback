from typing import Optional, List
import logging
import uuid
import yaml  # type: ignore[import]

from .verification_request import (
    VerificationRequest,
    PrimaryVerificationData,
    VerificationDType,
)
from .base import BaseHandler, CompositeHandler

_CODE_MARKERS = {
    VerificationDType.argdown: "```argdown",
    VerificationDType.xml: "```xml",
}
_MULTI_VALUED_ATTRIBUTES = {"*": {"supports", "attacks"}}


class ProcessingHandler(BaseHandler):
    """Base handler interface for processing operations in the Chain of Responsibility pattern."""


class FencedCodeBlockExtractor(ProcessingHandler):
    """Handler that extracts fenced code blocks from text."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        supported_languages: List[VerificationDType] | None = None,
    ):
        super().__init__(name, logger)
        self.supported_languages = supported_languages or [
            VerificationDType.argdown,
            VerificationDType.xml,
        ]

    def handle(self, request: VerificationRequest) -> VerificationRequest:
        """Extract fenced code blocks of specified languages."""
        input_text = request.inputs
        if isinstance(input_text, list):
            input_text = "\n\n".join(input_text)

        # Process each supported language
        extracted_blocks = 0

        for language in self.supported_languages:
            code_marker = _CODE_MARKERS[language]
            close_marker = "\n```"

            needs_to_be_parsed = input_text
            if code_marker in needs_to_be_parsed:
                splits = needs_to_be_parsed.split(code_marker)[1].split(close_marker, 1)
                if len(splits) > 1:
                    snippet = code_marker + splits[0] + close_marker
                    needs_to_be_parsed = splits[1]

                    # try to parse yaml code metadata after code marker
                    try:
                        lines = splits[0].split("\n")
                        metadata = yaml.safe_load(lines[0]) if lines else None
                    except (IndexError, yaml.YAMLError) as e:
                        self.logger.debug(f"No metadata found in code snippet {lines[0] if lines else ''} ({e})")
                        metadata = None

                    request.verification_data.append(
                        PrimaryVerificationData(
                            id=f"{language.value}_{str(uuid.uuid4())}",
                            dtype=language,
                            data=None,
                            code_snippet=snippet,
                            metadata=metadata,
                        )
                    )
                    extracted_blocks += 1
            del needs_to_be_parsed

            if extracted_blocks == 0:
                self.logger.debug(f"No {language.value} code blocks found to extract")

        return request


class ArgdownParser(ProcessingHandler):
    """Handler that parses all Argdown code snippets into an ArgdownMultiDiGraph."""

    def __init__(
        self, name: Optional[str] = None, logger: Optional[logging.Logger] = None
    ):
        super().__init__(name, logger)

    def handle(self, request: VerificationRequest) -> VerificationRequest:
        """Parse Argdown code snippets."""
        from pyargdown import parse_argdown

        for vdata in request.verification_data:
            if vdata.data is not None:
                # Skip if data is already parsed
                self.logger.warning(f"Data for {vdata.id} is already parsed. Skipping.")
                continue
            if vdata.code_snippet is None:
                self.logger.debug(f"Code snippet for {vdata.id} is None. Skipping.")
                continue
            if vdata.dtype == VerificationDType.argdown:
                code_snippet = vdata.code_snippet
                code_marker = _CODE_MARKERS[vdata.dtype]
                # remove metadata from code snippet
                if "\n" in code_snippet and code_snippet.startswith(code_marker):
                    # remove the first line (metadata) from the code snippet, add back the code marker
                    code_snippet = code_marker + "\n" + code_snippet.split("\n", 1)[1]
                try:
                    argdown = parse_argdown(code_snippet)
                    vdata.data = argdown
                except Exception as e:
                    metadata_text = f" {vdata.metadata}" if vdata.metadata else ""
                    request.add_result(
                        self.name,
                        [vdata.id],
                        False,
                        f"Failed to parse argdown code snippet{metadata_text}: {str(e)}",
                    )

        return request


class XMLParser(ProcessingHandler):
    """Handler that parses any XML code into a BeautifulSoup object."""

    def __init__(
        self, name: Optional[str] = None, logger: Optional[logging.Logger] = None
    ):
        super().__init__(name, logger)

    def handle(self, request: VerificationRequest) -> VerificationRequest:
        """Parse XML code blocks."""
        from bs4 import BeautifulSoup

        for vdata in request.verification_data:
            if vdata.data is not None:
                # Skip if data is already parsed
                self.logger.warning(f"Data for {vdata.id} is already parsed. Skipping.")
                continue
            if vdata.code_snippet is None:
                self.logger.debug(f"Code snippet for {vdata.id} is None. Skipping.")
                continue
            if vdata.dtype == VerificationDType.xml:
                code_snippet = vdata.code_snippet
                code_marker = _CODE_MARKERS[vdata.dtype]
                # remove metadata from code snippet
                if "\n" in code_snippet and code_snippet.startswith(code_marker):
                    # remove the first line (metadata) from the code snippet, add back the code marker
                    code_snippet = code_marker + "\n" + code_snippet.split("\n", 1)[1]
                try:
                    
                    soup = BeautifulSoup(
                        code_snippet,
                        "html.parser",
                        multi_valued_attributes=_MULTI_VALUED_ATTRIBUTES,
                    )
                    vdata.data = soup
                except Exception as e:
                    metadata_text = f" {vdata.metadata}" if vdata.metadata else ""
                    request.add_result(
                        self.name,
                        [vdata.id],
                        False,
                        f"Failed to parse XML code snippet{metadata_text}: {str(e)}",
                    )

        return request


class CompositeProcessingHandler(CompositeHandler):
    """Processing handler with default pipeline."""

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        handlers: Optional[List[ProcessingHandler]] = None,
    ):
        if handlers is None:
            handlers = [
                FencedCodeBlockExtractor(
                    name="FencedCodeBlockExtractor", logger=logger
                ),
                ArgdownParser(name="ArgdownParser", logger=logger),
                XMLParser(name="XMLParser", logger=logger),
            ]
        super().__init__(name, logger, handlers)

"""HTML to the flat text a chunker can read.

Two parsers, on purpose. bs4 walks the tree and reproduces what a browser would
render: newlines where the layout implies them, tables as tab-separated rows,
list items as hyphens. trafilatura is better at finding the article inside a
page full of chrome, but it returns nothing on pages it does not recognize, so
it is opt-in (`parse_with_trafilatura`) and always falls back to bs4.

Which classes and elements to drop, and whether links survive as markdown, come
from `BrainSettings` rather than module globals, so one process can parse a wiki
export and a marketing site with different rules.
"""

from __future__ import annotations

import logging
import re
from copy import copy
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import IO

import bs4

from brain.config import BrainSettings

logger = logging.getLogger(__name__)

# Mintlify-built docs sites wrap a duplicate copy of the page in these classes,
# so every paragraph would otherwise be indexed twice.
MINTLIFY_UNWANTED = ["sticky", "hidden"]

_ANCHOR_ELEMENT = "a"
_HREF_ATTRIBUTE = "href"
_TABLE_ELEMENT = "table"


class HtmlLinkStrategy(str, Enum):
    """What to do with an `<a href>` around a run of text."""

    # Drop the href; index only the anchor text.
    STRIP = "strip"
    # Keep it as a markdown link, so the URL is retrievable from the chunk.
    MARKDOWN = "markdown"


@dataclass
class ParsedHTML:
    title: str | None
    cleaned_text: str


def strip_excessive_newlines_and_spaces(document: str) -> str:
    # collapse repeated spaces into one
    document = re.sub(r" +", " ", document)
    # remove trailing spaces
    document = re.sub(r" +[\n\r]", "\n", document)
    # remove repeated newlines
    document = re.sub(r"[\n\r]+", "\n", document)
    return document.strip()


def strip_newlines(document: str) -> str:
    # HTML might contain newlines which are just whitespaces to a browser
    return re.sub(r"[\n\r]+", " ", document)


def format_element_text(element_text: str, link_href: str | None, *, settings: BrainSettings) -> str:
    element_text_no_newlines = strip_newlines(element_text)

    if not link_href or settings.html_link_strategy != HtmlLinkStrategy.MARKDOWN:
        return element_text_no_newlines

    return f"[{element_text_no_newlines}]({link_href})"


def _get_ancestor_link_href(element: bs4.element.PageElement, in_table: bool) -> str | None:
    if in_table:
        return None

    link = element.find_parent(_ANCHOR_ELEMENT)
    if not link:
        return None

    href = link.get(_HREF_ATTRIBUTE)
    return href[0] if isinstance(href, list) else href


def parse_html_with_trafilatura(html_content: str) -> str:
    """Extract the main article, discarding navigation and boilerplate."""
    import trafilatura
    from trafilatura.settings import use_config

    config = use_config()
    config.set("DEFAULT", "include_links", "True")
    config.set("DEFAULT", "include_tables", "True")
    config.set("DEFAULT", "include_images", "True")
    config.set("DEFAULT", "include_formatting", "True")

    extracted_text = trafilatura.extract(html_content, config=config)
    return strip_excessive_newlines_and_spaces(extracted_text) if extracted_text else ""


def format_document_soup(
    document: bs4.BeautifulSoup,
    *,
    settings: BrainSettings,
    table_cell_separator: str = "\t",
) -> str:
    """Flatten a parsed document the way a browser would render it.

    - Newlines inside the HTML are whitespace to a browser, so they are removed.
    - Newlines appear only around headings and paragraphs, or where the markup
      is explicit about them (`br`, `pre`).
    - Table rows are newline separated and cells tab separated, which is what
      makes a table survive chunking as something still readable.
    - List items start with a hyphen.
    """
    text = ""
    list_element_start = False
    verbatim_output = 0
    last_added_newline = False

    for e in document.descendants:
        verbatim_output -= 1
        if isinstance(e, bs4.element.NavigableString):
            if isinstance(e, bs4.element.Comment | bs4.element.Doctype):
                continue
            in_table = e.find_parent(_TABLE_ELEMENT) is not None
            element_text = e.text
            if in_table:
                element_text = element_text.replace("\n", " ").strip()

            # Some tags become spaces here but newlines below (br, headings).
            # Without this, the newline would be followed by a stray space.
            if last_added_newline and element_text.startswith(" "):
                element_text = element_text[1:]
                last_added_newline = False

            if element_text:
                content_to_add = (
                    element_text
                    if verbatim_output > 0
                    else format_element_text(
                        element_text,
                        _get_ancestor_link_href(e, in_table),
                        settings=settings,
                    )
                )

                # Don't join separate elements without any spacing
                if (text and not text[-1].isspace()) and (
                    content_to_add and not content_to_add[0].isspace()
                ):
                    text += " "

                text += content_to_add

                list_element_start = False
        elif isinstance(e, bs4.element.Tag):
            in_table = e.find_parent(_TABLE_ELEMENT) is not None
            if e.name == "tr" and in_table:
                text += "\n"
            elif e.name in ["td", "th"] and in_table:
                text += table_cell_separator
            elif in_table:
                # don't handle other cases while in table
                pass
            elif e.name in ["p", "div"]:
                if not list_element_start:
                    text += "\n"
            elif e.name in ["h1", "h2", "h3", "h4"] or e.name == "br":
                text += "\n"
                list_element_start = False
                last_added_newline = True
            elif e.name == "li":
                text += "\n- "
                list_element_start = True
            elif e.name == "pre" and verbatim_output <= 0:
                verbatim_output = len(list(e.childGenerator()))
    return strip_excessive_newlines_and_spaces(text)


def parse_html_page_basic(
    text: str | BytesIO | IO[bytes], *, settings: BrainSettings
) -> str:
    soup = bs4.BeautifulSoup(text, "lxml")
    return format_document_soup(soup, settings=settings)


def web_html_cleanup(
    page_content: str | bs4.BeautifulSoup,
    *,
    settings: BrainSettings,
    mintlify_cleanup_enabled: bool = True,
    additional_element_types_to_discard: list[str] | None = None,
) -> ParsedHTML:
    """Strip navigation chrome, then flatten what is left."""
    soup = bs4.BeautifulSoup(page_content, "lxml") if isinstance(page_content, str) else page_content

    title_tag = soup.find("title")
    title = None
    if title_tag and title_tag.text:
        title = title_tag.text
        title_tag.extract()

    unwanted_classes = copy(settings.html_ignored_classes)
    if mintlify_cleanup_enabled:
        unwanted_classes.extend(MINTLIFY_UNWANTED)
    for undesired_element in unwanted_classes:
        for tag in soup.find_all(
            class_=lambda x, cls=undesired_element: x and cls in x.split()
        ):
            tag.extract()

    discard = list(settings.html_ignored_elements) + list(additional_element_types_to_discard or [])
    for undesired_tag in discard:
        for tag in soup.find_all(undesired_tag):
            tag.extract()

    soup_string = str(soup)
    page_text = ""

    if settings.parse_with_trafilatura:
        try:
            page_text = parse_html_with_trafilatura(soup_string)
            if not page_text:
                raise ValueError("Empty content returned by trafilatura.")
        except Exception as e:
            logger.info("Trafilatura parsing failed: %s. Falling back on bs4.", e)
            page_text = format_document_soup(soup, settings=settings)
    else:
        page_text = format_document_soup(soup, settings=settings)

    # 200B is ZeroWidthSpace which we don't care for
    cleaned_text = page_text.replace("​", "")

    return ParsedHTML(title=title, cleaned_text=cleaned_text)

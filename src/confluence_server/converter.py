"""Convert Confluence HTML content to Markdown with attachment handling."""

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify as md


class ConfluenceMarkdownConverter:
    """Convert Confluence HTML to clean Markdown."""

    @staticmethod
    def convert(html_content: str) -> str:
        """Convert Confluence HTML to Markdown.

        Handles Confluence-specific elements:
        - Info/warning/note macros → blockquotes
        - Code blocks with language
        - Tables
        - Internal links
        - Images and attachments
        """
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, "html.parser")

        ConfluenceMarkdownConverter._process_macros(soup)
        ConfluenceMarkdownConverter._process_code_blocks(soup)
        ConfluenceMarkdownConverter._process_links(soup)
        ConfluenceMarkdownConverter._process_images(soup)
        ConfluenceMarkdownConverter._process_lists(soup)
        ConfluenceMarkdownConverter._remove_unwanted(soup)

        markdown = md(str(soup), heading_style="ATX", bullets="-", code_language_callback=True)

        markdown = ConfluenceMarkdownConverter._cleanup(markdown)

        return markdown.strip()

    @staticmethod
    def _process_macros(soup: BeautifulSoup) -> None:
        """Convert Confluence macros to standard HTML."""
        for macro in soup.find_all("div", class_=re.compile(r"confluence-information-macro")):
            macro_type = ""
            for cls in macro.get("class", []):
                if "information-macro-" in cls:
                    macro_type = cls.split("information-macro-")[-1].lower()
                    break

            title_map = {
                "info": "ℹ️ Info",
                "note": "📝 Note",
                "tip": "💡 Tip",
                "warning": "⚠️ Warning",
                "danger": "🚨 Danger",
            }
            title = title_map.get(macro_type, "Note")

            macro.name = "blockquote"
            title_tag = soup.new_tag("strong")
            title_tag.string = f"{title}: "
            macro.insert(0, title_tag)
            macro.insert(1, soup.new_tag("br"))

        for status in soup.find_all("span", class_="aui-lozenge"):
            status.name = "code"

        for task in soup.find_all("li", class_=re.compile(r"task-list-item")):
            checkbox = task.find("input", type="checkbox")
            if checkbox:
                checked = "[x]" if checkbox.get("checked") else "[ ]"
                text = task.get_text(strip=True)
                task.string = f"{checked} {text}"

    @staticmethod
    def _process_code_blocks(soup: BeautifulSoup) -> None:
        """Process Confluence code blocks."""
        for pre in soup.find_all("pre"):
            code = pre.find("code")
            if not code:
                code = pre

            language = ""
            for attr in ["data-language", "class"]:
                val = code.get(attr, "")
                if isinstance(val, list):
                    for v in val:
                        if v.startswith("language-"):
                            language = v.replace("language-", "")
                            break
                elif isinstance(val, str) and val.startswith("language-"):
                    language = val.replace("language-", "")

            code_text = code.get_text()

            wrapper = soup.new_tag("div")
            wrapper["class"] = "code-block"
            wrapper["data-language"] = language
            wrapper.string = f"\n```{language}\n{code_text}\n```\n"

            pre.replace_with(wrapper)

    @staticmethod
    def _process_links(soup: BeautifulSoup) -> None:
        """Process links, especially Confluence internal links."""
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if "/pages/viewpage.action" in href:
                match = re.search(r"pageId=(\d+)", href)
                if match:
                    a["href"] = f"confluence://page/{match.group(1)}"
            elif "/display/" in href or "/spaces/" in href:
                a["href"] = f"confluence://{href.split('/display/')[-1].split('/spaces/')[-1]}"

    @staticmethod
    def _process_images(soup: BeautifulSoup) -> None:
        """Process images and attachments."""
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if "/download/attachments/" in src:
                match = re.search(r"/attachments/thumbnail/[^/]+/(\d+)/(.+)", src)
                if not match:
                    match = re.search(r"/download/attachments/[^/]+/(.+?)(\?|$)", src)
                if match:
                    filename = match.group(1)
                    img["src"] = f"./attachments/{filename}"
                    img["alt"] = img.get("alt", filename)

    @staticmethod
    def _process_lists(soup: BeautifulSoup) -> None:
        """Ensure lists are properly formatted."""
        pass

    @staticmethod
    def _remove_unwanted(soup: BeautifulSoup) -> None:
        """Remove Confluence-specific noise."""
        for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        for tag in soup.find_all(attrs={"data-macro-name": True}):
            macro_name = tag.get("data-macro-name", "")
            if macro_name in ("details", "expand", "toc", "anchor"):
                tag.decompose()

    @staticmethod
    def _cleanup(markdown: str) -> str:
        """Clean up the resulting Markdown."""
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        markdown = re.sub(r" +$", "", markdown, flags=re.MULTILINE)
        markdown = markdown.strip()

        return markdown


def build_frontmatter(
    title: str,
    space_key: str,
    page_id: str,
    parent_id: str | None = None,
    created_date: str | None = None,
    last_modified: str | None = None,
    created_by: str | None = None,
    last_modified_by: str | None = None,
) -> str:
    """Build YAML frontmatter for a page."""
    lines = [
        "---",
        f"title: {title}",
        f"space: {space_key}",
        f"page_id: {page_id}",
    ]
    if parent_id:
        lines.append(f"parent_id: {parent_id}")
    if created_date:
        lines.append(f"created_date: {created_date}")
    if last_modified:
        lines.append(f"last_modified: {last_modified}")
    if created_by:
        lines.append(f"created_by: {created_by}")
    if last_modified_by:
        lines.append(f"last_modified_by: {last_modified_by}")
    lines.append("---")
    return "\n".join(lines)


def convert_page_to_markdown(
    page: dict[str, Any],
    space_key: str,
) -> str:
    """Convert a Confluence page API response to Markdown with frontmatter."""
    body = page.get("body", {})
    view = body.get("view", {})
    html = view.get("value", "")

    title = page.get("title", "Untitled")
    page_id = page.get("id", "")

    ancestors = page.get("ancestors", [])
    parent_id = ancestors[-1].get("id") if ancestors else None

    history = page.get("history", {})
    version = page.get("version", {})
    created_by_info = history.get("createdBy", {})
    last_updated_by = history.get("lastUpdated", {}).get("by", {})

    frontmatter = build_frontmatter(
        title=title,
        space_key=space_key,
        page_id=page_id,
        parent_id=parent_id,
        created_date=history.get("createdDate"),
        last_modified=version.get("when"),
        created_by=created_by_info.get("displayName", created_by_info.get("username")),
        last_modified_by=last_updated_by.get("displayName", last_updated_by.get("username")),
    )

    converter = ConfluenceMarkdownConverter()
    content = converter.convert(html)

    return f"{frontmatter}\n\n{content}"


def generate_page_path(
    page: dict[str, Any],
    space_key: str,
) -> str:
    """Generate a file path for a page based on its hierarchy."""
    title = page.get("title", "Untitled")
    ancestors = page.get("ancestors", [])

    path_parts = [space_key]
    for ancestor in ancestors:
        ancestor_title = ancestor.get("title", "")
        if ancestor_title:
            path_parts.append(_sanitize_filename(ancestor_title))

    path_parts.append(_sanitize_filename(title) + ".md")
    return "/".join(path_parts)


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", " ", name)
    name = name[:200]
    return name

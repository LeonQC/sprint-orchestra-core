#!/usr/bin/env python3
"""Generate, validate, and install the canonical AtlasCart M1 Jira dataset.

Only the Python standard library is required. Database installation uses the
PostgreSQL client already available in the local `orchestra-postgres` container.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import subprocess
import sys
import textwrap
import uuid
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from html import escape as html_escape
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape
from zoneinfo import ZoneInfo

from m1_comments import COMMENTS as ALL_COMMENTS
from m1_quality import SUBTASK_RATIONALE
from m1_visual_evidence import (
    FRONTEND_VISUAL_EVIDENCE,
    all_visual_attachments,
    frontend_visual_issue_keys,
    verification_filenames,
)
from m1_scenario import (
    ASSIGNEE_CHANGES,
    AS_OF_DATE,
    BUG_REPORT_DETAILS,
    CURATED_ATTACHMENTS,
    DATASET_ID,
    DATASET_VERSION,
    ESTIMATE_CHANGES,
    FIXTURE_MARKER,
    FIXTURE_NAME,
    GROUPS,
    ISSUE_BY_KEY,
    ISSUE_LINKS as CANONICAL_ISSUE_LINKS,
    ISSUES,
    LABELS,
    LINK_NARRATIVES,
    POINT_RATIONALES,
    REPOSITORIES,
    S1_ADDED,
    S1_CARRIED,
    S1_INITIAL,
    S2_ADDED,
    S2_CARRIED,
    S2_CARRY_REASONS,
    S2_INITIAL,
    S2_REMOVED,
    S3_INITIAL,
    SNAPSHOT_AT,
    SPACE_KEY,
    SPRINTS,
    STATUS_OVERRIDES,
    USERS,
    USER_BY_ID,
    ts,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "generated" / "m1"
ATTACHMENT_DIR = OUTPUT_DIR / "attachments"
INJECT_SQL_PATH = OUTPUT_DIR / "inject_atlas_cart_m1.sql"
CLEANUP_SQL_PATH = OUTPUT_DIR / "cleanup_atlas_cart_m1.sql"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
VALIDATION_PATH = OUTPUT_DIR / "validation-report.md"
REVIEW_PATH = OUTPUT_DIR / "jira-review-path.md"
ESTIMATE_AUDIT_PATH = OUTPUT_DIR / "estimate-audit.md"
# bcrypt of the demo password "123"; jira-backend only accepts hashed passwords (V35).
DEMO_PASSWORD_HASH = "$2a$10$.7G43/qDfW6zKRPSphqVg.eF9SXvKTV9HLmlrSOsa48.W1sFXqYoi"

ALLOWED_TYPES = {"epic", "story", "task", "bug", "subtask"}
ALLOWED_STATUSES = {"planned", "in_progress", "done"}
ALLOWED_PRIORITIES = {"low", "medium", "high", "highest"}
ALLOWED_LINKS = {"blocks", "clones", "duplicates", "relates to"}
EXPECTED_TYPE_COUNTS = {"epic": 6, "story": 17, "task": 11, "bug": 8, "subtask": 6}


@dataclass(frozen=True)
class CodeLink:
    issue: str
    creator: str
    kind: str
    repo: str
    ref_id: str
    title: str
    state: str
    author_login: str
    created_at: datetime
    last_activity_at: datetime
    url: str


@dataclass(frozen=True)
class History:
    issue: str
    actor: str | None
    event_type: str
    field_name: str | None
    from_value: str | None
    to_value: str | None
    description: str
    created_at: datetime


def code_repo(issue: dict[str, Any]) -> str:
    labels = set(issue["labels"])
    if "platform" in labels:
        return "atlascart-infra"
    if "frontend" in labels and "backend" not in labels:
        return "atlascart-web"
    return "atlascart-api"


CODE_ISSUE_KEYS = [
    "ATC-7", "ATC-8", "ATC-9", "ATC-10", "ATC-11", "ATC-12",
    "ATC-13", "ATC-14", "ATC-15", "ATC-16", "ATC-17", "ATC-19",
    "ATC-20", "ATC-21", "ATC-23", "ATC-24", "ATC-25", "ATC-26",
    "ATC-27", "ATC-28", "ATC-29", "ATC-30", "ATC-31", "ATC-33",
    "ATC-44", "ATC-45",
]


def build_code_links() -> list[CodeLink]:
    links: list[CodeLink] = []
    pr_candidates: list[dict[str, Any]] = []
    for key in CODE_ISSUE_KEYS:
        item = ISSUE_BY_KEY[key]
        started_at = item["started_at"]
        if started_at is None:
            raise ValueError(f"Code-backed issue {key} has no start time")
        repo = code_repo(item)
        opened_at = started_at + timedelta(hours=4)
        if item["done_at"]:
            last_activity = item["done_at"] - timedelta(minutes=20)
            if opened_at >= last_activity:
                opened_at = started_at + timedelta(minutes=45)
            state = "closed"
        else:
            last_activity = min(SNAPSHOT_AT - timedelta(hours=2), opened_at + timedelta(days=2))
            state = "open"
        creator = item["assignee"] or item["reporter"]
        login = USER_BY_ID[creator]["username"].replace("atlascart.", "")
        title = f"{key} {item['title']}"
        pr_candidates.append({"issue": key, "creator": creator, "repo": repo, "title": title, "state": state, "login": login, "created_at": opened_at, "last_activity_at": last_activity})

    next_number: defaultdict[str, int] = defaultdict(lambda: 101)
    for candidate in sorted(pr_candidates, key=lambda row: (row["repo"], row["created_at"], row["issue"])):
        repo = candidate["repo"]
        pr_number = next_number[repo]
        next_number[repo] += 1
        url = f"https://github.com/northstar-labs-sim/{repo}/pull/{pr_number}"
        links.append(CodeLink(candidate["issue"], candidate["creator"], "pull_request", repo, str(pr_number), candidate["title"], candidate["state"], candidate["login"], candidate["created_at"], candidate["last_activity_at"], url))

    for key, sha, at in [
        ("ATC-10", "4ad719c", ts(18, 14, 50)),
        ("ATC-16", "c21e085", ts(5, 12, 50)),
        ("ATC-30", "8f6c0d1", ts(27, 14, 35)),
        ("ATC-44", "d30ac72", ts(27, 12, 20)),
    ]:
        item = ISSUE_BY_KEY[key]
        repo = code_repo(item)
        creator = item["assignee"] or item["reporter"]
        login = USER_BY_ID[creator]["username"].replace("atlascart.", "")
        title = f"{key} {item['title']}"
        url = f"https://github.com/northstar-labs-sim/{repo}/commit/{sha}"
        links.append(CodeLink(key, creator, "commit", repo, sha, title, "committed", login, at, at, url))
    return sorted(links, key=lambda link: (link.created_at, link.issue, link.kind))


CODE_LINKS = build_code_links()


DEFAULT_LINK_RATIO = 20.0
LINK_RATIO = DEFAULT_LINK_RATIO
LINK_PRIORITY = [
    ("ATC-8", "ATC-9", "blocks"),
    ("ATC-13", "ATC-14", "blocks"),
    ("ATC-22", "ATC-10", "blocks"),
    ("ATC-33", "ATC-34", "blocks"),
    ("ATC-48", "ATC-27", "duplicates"),
    ("ATC-9", "ATC-10", "blocks"),
    ("ATC-9", "ATC-11", "blocks"),
    ("ATC-11", "ATC-12", "blocks"),
    ("ATC-15", "ATC-12", "relates to"),
    ("ATC-15", "ATC-14", "relates to"),
    ("ATC-23", "ATC-24", "blocks"),
    ("ATC-23", "ATC-26", "blocks"),
    ("ATC-24", "ATC-25", "relates to"),
    ("ATC-27", "ATC-30", "relates to"),
    ("ATC-28", "ATC-30", "blocks"),
    ("ATC-29", "ATC-30", "blocks"),
    ("ATC-30", "ATC-31", "blocks"),
    ("ATC-30", "ATC-32", "blocks"),
    ("ATC-43", "ATC-30", "relates to"),
    ("ATC-44", "ATC-43", "blocks"),
    ("ATC-45", "ATC-43", "blocks"),
    ("ATC-43", "ATC-46", "relates to"),
    ("ATC-43", "ATC-47", "relates to"),
]


def link_target_count(ratio: float) -> int:
    max_coverage = len({key for source, target, _kind, _at in CANONICAL_ISSUE_LINKS for key in (source, target)})
    max_ratio = max_coverage / len(ISSUES) * 100
    if not math.isfinite(ratio) or ratio < 0 or ratio > max_ratio + 1e-9:
        raise ValueError(
            f"Issue-link ratio must be from 0 through {max_ratio:.2f}. "
            "The upper bound keeps the graph limited to relationships supported by the story."
        )
    if ratio == 0:
        return 0
    return max(2, min(max_coverage, math.ceil(len(ISSUES) * ratio / 100.0)))


def build_issue_link_set(ratio: float) -> list[tuple[str, str, str, datetime]]:
    """Choose a sparse semantic graph whose unique-issue coverage matches the ratio."""
    target = link_target_count(ratio)
    if target == 0:
        return []
    by_identity = {(source, target_key, kind): (source, target_key, kind, at) for source, target_key, kind, at in CANONICAL_ISSUE_LINKS}
    candidates = [by_identity[identity] for identity in LINK_PRIORITY]
    assert len(candidates) == len(CANONICAL_ISSUE_LINKS)

    # The canonical graph is small. Depth-first selection lets odd targets use
    # an existing node plus one new dependency instead of rounding to an
    # unrelated pair. Candidate order keeps the default graph easy to review.
    memo: set[tuple[int, frozenset[str]]] = set()

    def select(index: int, nodes: frozenset[str]) -> tuple[int, ...] | None:
        if len(nodes) == target:
            return ()
        if index >= len(candidates) or len(nodes) > target:
            return None
        state = (index, nodes)
        if state in memo:
            return None
        memo.add(state)
        source, target_key, _kind, _at = candidates[index]
        expanded = nodes | {source, target_key}
        if len(expanded) <= target and expanded != nodes:
            chosen = select(index + 1, expanded)
            if chosen is not None:
                return (index,) + chosen
        return select(index + 1, nodes)

    selected_indexes = select(0, frozenset())
    if selected_indexes is None:
        raise ValueError(f"No semantic issue-link subset can cover exactly {target} issues")
    return sorted((candidates[index] for index in selected_indexes), key=lambda row: row[3])


def comments_for_links(
    issue_links: list[tuple[str, str, str, datetime]],
    attachments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected = {(source, target, kind) for source, target, kind, _at in issue_links}
    selected_files = None if attachments is None else {attachment["filename"] for attachment in attachments}
    result: list[dict[str, Any]] = []
    for source_comment in ALL_COMMENTS:
        if source_comment.get("link_context") and source_comment["link_context"] not in selected:
            continue
        comment = source_comment.copy()
        if selected_files is not None:
            kept_lines = []
            for line in comment["content"].splitlines():
                match = re.search(r"\[\[ATTACHMENT:([^]]+)\]\]", line)
                if match and match.group(1) not in selected_files:
                    if line.strip().startswith("Evidence:") or line.strip().startswith("Evidence AC"):
                        continue
                    line = line.replace(match.group(0), "").rstrip().rstrip(":") + "."
                kept_lines.append(line)
            comment["content"] = "\n".join(kept_lines)
        result.append(comment)
    return result


ISSUE_LINKS = build_issue_link_set(DEFAULT_LINK_RATIO)
COMMENTS = comments_for_links(ISSUE_LINKS)


def status_events(item: dict[str, Any]) -> list[tuple[datetime, str, str]]:
    key = item["key"]
    if key in STATUS_OVERRIDES:
        return STATUS_OVERRIDES[key]
    if item["status"] == "done" and item["started_at"] and item["done_at"]:
        return [
            (item["started_at"], "planned", "in_progress"),
            (item["done_at"], "in_progress", "done"),
        ]
    if item["status"] == "in_progress" and item["started_at"]:
        return [(item["started_at"], "planned", "in_progress")]
    return []


def points_at(issue_key: str, at: datetime) -> int | None:
    item = ISSUE_BY_KEY[issue_key]
    value = item["initial_points"]
    change = ESTIMATE_CHANGES.get(issue_key)
    if change and change[2] <= at:
        value = change[1]
    return value


def make_pdf(title: str, lines: list[str], sections: list[dict[str, Any]] | None = None) -> bytes:
    """Build a high-contrast, human-readable report PDF without third-party packages."""
    def pdf_text(value: str) -> str:
        return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    if sections is None:
        wrapped = [piece for line in lines for piece in (textwrap.wrap(line, width=82) or [""])]
        sections = [
            {"heading": f"Page {index // 22 + 1}", "lines": wrapped[index:index + 22]}
            for index in range(0, max(1, len(wrapped)), 22)
        ]

    streams: list[bytes] = []
    page_count = len(sections)
    regular_font_id = 3 + page_count * 2
    bold_font_id = regular_font_id + 1
    for page_number, section in enumerate(sections, start=1):
        heading = section.get("heading") or title
        raw_lines = [str(line) for line in section.get("lines", [])]
        commands = [
            # Opaque warm-white canvas and a dark report masthead.
            "0.985 0.988 0.992 rg", "0 0 612 792 re f",
            "0.07 0.12 0.22 rg", "0 688 612 104 re f",
            "0.05 0.55 0.52 rg", "0 688 10 104 re f",
        ]

        def add_text(x: float, y: float, size: int, value: str, *, bold: bool = False, color: str = "0.10 0.14 0.22") -> None:
            font = "F2" if bold else "F1"
            commands.extend([f"{color} rg", "BT", f"/{font} {size} Tf", f"{x:.1f} {y:.1f} Td", f"({pdf_text(value)}) Tj", "ET"])

        # Masthead: the report name, section label, and page position remain
        # visually stable across every PDF attachment.
        add_text(34, 758, 8, "ATLASCART  /  SYNTHETIC PROJECT RECORD", bold=True, color="0.45 0.91 0.86")
        for line_index, title_line in enumerate(textwrap.wrap(title, width=56)[:2]):
            add_text(34, 730 - line_index * 20, 17 if line_index == 0 else 14, title_line, bold=True, color="1 1 1")
        add_text(520, 758, 8, f"{page_number:02d} / {page_count:02d}", bold=True, color="0.82 0.88 0.96")

        # Section identity uses a small accent label and a large readable title.
        add_text(40, 655, 8, f"SECTION {page_number:02d}", bold=True, color="0.05 0.45 0.43")
        commands.extend(["0.05 0.55 0.52 rg", "40 642 32 3 re f"])
        heading_lines = textwrap.wrap(str(heading), width=52)[:2]
        for line_index, heading_line in enumerate(heading_lines):
            add_text(40, 615 - line_index * 20, 15, heading_line, bold=True)
        y = 570 - max(0, len(heading_lines) - 1) * 18

        # The first paragraph is presented as a summary card. Remaining text
        # uses comfortable line spacing and clear bullet/code treatments.
        body: list[str] = []
        for raw in raw_lines:
            if raw.startswith("- "):
                wrapped = textwrap.wrap(raw[2:], width=72) or [""]
                body.extend(["- " + wrapped[0], *["  " + part for part in wrapped[1:]]])
            elif raw.startswith("CODE: "):
                body.extend(["CODE: " + part for part in (textwrap.wrap(raw[6:], width=76) or [""])])
            else:
                body.extend(textwrap.wrap(raw, width=78) or [""])

        if body:
            # Keep the first three source facts together even when the last
            # one wraps; a summary sentence must never spill outside its card.
            summary_line_count = 0
            for raw in raw_lines[:3]:
                source = raw[2:] if raw.startswith("- ") else raw[6:] if raw.startswith("CODE: ") else raw
                width = 72 if raw.startswith("- ") else 76 if raw.startswith("CODE: ") else 78
                summary_line_count += len(textwrap.wrap(source, width=width) or [""])
            summary_lines = body[: min(6, summary_line_count)]
            card_height = 28 + len(summary_lines) * 15
            commands.extend(["0.91 0.96 0.96 rg", f"40 {y - card_height + 8} 532 {card_height} re f", "0.05 0.55 0.52 rg", f"40 {y - card_height + 8} 5 {card_height} re f"])
            add_text(56, y - 7, 8, "AT A GLANCE", bold=True, color="0.05 0.45 0.43")
            for line_index, line in enumerate(summary_lines):
                clean = line[2:] if line.startswith("- ") else line.strip()
                add_text(56, y - 24 - line_index * 15, 9, clean)
            y -= card_height + 12

        for line in body[len(summary_lines) if body else 0:18]:
            if line.startswith("- "):
                commands.extend(["0.05 0.55 0.52 rg", f"48 {y + 3} 5 5 re f"])
                add_text(62, y, 9, line[2:])
            elif line.startswith("  "):
                add_text(62, y, 9, line.strip(), color="0.22 0.28 0.38")
            elif line.startswith("CODE: "):
                commands.extend(["0.92 0.94 0.97 rg", f"48 {y - 7} 516 23 re f", "0.32 0.39 0.50 RG", "0.6 w", f"48 {y - 7} 516 23 re S"])
                add_text(58, y, 8, line[6:], bold=True)
            elif not line:
                y -= 5
                continue
            else:
                add_text(48, y, 9, line, color="0.18 0.23 0.32")
            y -= 16

        table = section.get("table")
        if table:
            headers = [str(value) for value in table["headers"]]
            rows = [[str(value) for value in row] for row in table["rows"]]
            column_count = len(headers)
            table_x, table_width = 40, 532
            column_width = table_width / column_count
            table_top = min(y - 10, 480)
            header_height = 30
            commands.extend(["0.07 0.12 0.22 rg", f"{table_x} {table_top - header_height} {table_width} {header_height} re f"])
            for column_index, header in enumerate(headers):
                shown = header if len(header) <= max(8, int(column_width / 5.4)) else header[: max(5, int(column_width / 5.4) - 3)] + "..."
                add_text(table_x + column_index * column_width + 7, table_top - 19, 8, shown, bold=True, color="1 1 1")
            table_y = table_top - header_height
            for row_index, row in enumerate(rows[:8]):
                max_chars = max(9, int(column_width / 4.8) - 1)
                wrapped_cells = [(textwrap.wrap(cell, width=max_chars) or [""])[:2] for cell in row]
                row_height = 26 + 12 * (max(len(cell_lines) for cell_lines in wrapped_cells) - 1)
                bottom = table_y - row_height
                fill = "1 1 1" if row_index % 2 == 0 else "0.95 0.97 0.99"
                commands.extend([f"{fill} rg", f"{table_x} {bottom} {table_width} {row_height} re f", "0.77 0.81 0.87 RG", "0.5 w", f"{table_x} {bottom} {table_width} {row_height} re S"])
                for column_index in range(1, column_count):
                    x = table_x + column_index * column_width
                    commands.extend([f"{x:.1f} {bottom} m {x:.1f} {table_y} l S"])
                for column_index, cell_lines in enumerate(wrapped_cells):
                    for line_index, cell_line in enumerate(cell_lines):
                        add_text(table_x + column_index * column_width + 7, table_y - 17 - line_index * 11, 7, cell_line, bold=(column_index == 0))
                table_y = bottom

        commands.extend(["0.78 0.82 0.88 RG", "0.6 w", "40 49 m 572 49 l S"])
        add_text(40, 31, 7, "ATC  /  AtlasCart synthetic project record", bold=True, color="0.30 0.36 0.46")
        add_text(522, 31, 7, f"PAGE {page_number} OF {page_count}", bold=True, color="0.30 0.36 0.46")
        streams.append("\n".join(commands).encode("latin-1", errors="replace"))

    kids = " ".join(f"{3 + index * 2} 0 R" for index in range(page_count))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"),
    ]
    for index, stream in enumerate(streams):
        content_id = 4 + index * 2
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {regular_font_id} 0 R /F2 {bold_font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii")
        )
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode("ascii"))
        result.extend(obj)
        result.extend(b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    result.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(result)


def make_xlsx(sheet_name: str, rows: list[list[Any]]) -> bytes:
    """Build a readable one-sheet XLSX with filters, frozen headers, and wrapping."""
    sheet_rows: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_number, value in enumerate(row, start=1):
            column = ""
            number = column_number
            while number:
                number, remainder = divmod(number - 1, 26)
                column = chr(65 + remainder) + column
            ref = f"{column}{row_number}"
            style = 1 if row_number == 1 else (2 if str(value).upper() == "PASS" else 3 if str(value).upper() in {"FAIL", "PENDING"} else 4)
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}" s="{style}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{xml_escape(str(value))}</t></is></c>')
        height = 30 if row_number == 1 else 45
        sheet_rows.append(f'<row r="{row_number}" ht="{height}" customHeight="1">{"".join(cells)}</row>')

    column_count = max(len(row) for row in rows)
    widths = []
    for index in range(column_count):
        longest = max(len(str(row[index])) if index < len(row) else 0 for row in rows)
        widths.append(min(45, max(12, longest + 2)))
    columns_xml = "".join(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>' for index, width in enumerate(widths, start=1))
    last_column = ""
    number = column_count
    while number:
        number, remainder = divmod(number - 1, 26)
        last_column = chr(65 + remainder) + last_column
    table_range = f"A1:{last_column}{len(rows)}"

    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{xml_escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
        "xl/styles.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Aptos"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Aptos"/></font></fonts><fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF2563EB"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFE8F5E9"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border/><border><left style="thin"><color rgb="FFD7DEE8"/></left><right style="thin"><color rgb="FFD7DEE8"/></right><top style="thin"><color rgb="FFD7DEE8"/></top><bottom style="thin"><color rgb="FFD7DEE8"/></bottom></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>',
        "xl/worksheets/sheet1.xml": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="18"/><cols>{columns_xml}</cols><sheetData>{"".join(sheet_rows)}</sheetData><autoFilter ref="{table_range}"/></worksheet>',
    }
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, content in files.items():
            info = zipfile.ZipInfo(filename, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content.encode("utf-8"))
    return output.getvalue()


def make_screenshot(
    caption: str,
    screenshot_type: str = "bug_ui",
    *,
    ac_index: int = 1,
    phase: str = "before",
    context: str = "",
) -> bytes:
    """Create a deterministic, readable product/CI/architecture PNG."""
    width, height = 640, 360
    pixels = [[(246, 248, 251) for _ in range(width)] for _ in range(height)]

    def rect(x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y1), min(height, y2)):
            pixels[y][max(0, x1):min(width, x2)] = [color] * max(0, min(width, x2) - max(0, x1))

    font = {
        "A":"01110/10001/10001/11111/10001/10001/10001", "B":"11110/10001/10001/11110/10001/10001/11110",
        "C":"01111/10000/10000/10000/10000/10000/01111", "D":"11110/10001/10001/10001/10001/10001/11110",
        "E":"11111/10000/10000/11110/10000/10000/11111", "F":"11111/10000/10000/11110/10000/10000/10000",
        "G":"01111/10000/10000/10111/10001/10001/01111", "H":"10001/10001/10001/11111/10001/10001/10001",
        "I":"11111/00100/00100/00100/00100/00100/11111", "J":"00111/00010/00010/00010/10010/10010/01100",
        "K":"10001/10010/10100/11000/10100/10010/10001", "L":"10000/10000/10000/10000/10000/10000/11111",
        "M":"10001/11011/10101/10101/10001/10001/10001", "N":"10001/11001/10101/10011/10001/10001/10001",
        "O":"01110/10001/10001/10001/10001/10001/01110", "P":"11110/10001/10001/11110/10000/10000/10000",
        "Q":"01110/10001/10001/10001/10101/10010/01101", "R":"11110/10001/10001/11110/10100/10010/10001",
        "S":"01111/10000/10000/01110/00001/00001/11110", "T":"11111/00100/00100/00100/00100/00100/00100",
        "U":"10001/10001/10001/10001/10001/10001/01110", "V":"10001/10001/10001/10001/10001/01010/00100",
        "W":"10001/10001/10001/10101/10101/11011/10001", "X":"10001/10001/01010/00100/01010/10001/10001",
        "Y":"10001/10001/01010/00100/00100/00100/00100", "Z":"11111/00001/00010/00100/01000/10000/11111",
        "0":"01110/10001/10011/10101/11001/10001/01110", "1":"00100/01100/00100/00100/00100/00100/01110",
        "2":"01110/10001/00001/00010/00100/01000/11111", "3":"11110/00001/00001/01110/00001/00001/11110",
        "4":"00010/00110/01010/10010/11111/00010/00010", "5":"11111/10000/10000/11110/00001/00001/11110",
        "6":"01110/10000/10000/11110/10001/10001/01110", "7":"11111/00001/00010/00100/01000/01000/01000",
        "8":"01110/10001/10001/01110/10001/10001/01110", "9":"01110/10001/10001/01111/00001/00001/01110",
        "-":"00000/00000/00000/11111/00000/00000/00000", ":":"00000/00100/00100/00000/00100/00100/00000",
        "+":"00000/00100/00100/11111/00100/00100/00000",
        ".":"00000/00000/00000/00000/00000/00110/00110", "/":"00001/00010/00100/01000/10000/00000/00000",
        "#":"01010/11111/01010/01010/11111/01010/00000", " ":"00000/00000/00000/00000/00000/00000/00000",
    }

    def text(x: int, y: int, value: str, color: tuple[int, int, int], scale: int = 2) -> None:
        cursor = x
        for char in value.upper():
            rows = font.get(char, font[" "]).split("/")
            for row_index, row in enumerate(rows):
                for col_index, bit in enumerate(row):
                    if bit == "1":
                        rect(cursor + col_index * scale, y + row_index * scale, cursor + (col_index + 1) * scale, y + (row_index + 1) * scale, color)
            cursor += 6 * scale

    def outline(x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], thickness: int = 2) -> None:
        rect(x1, y1, x2, y1 + thickness, color); rect(x1, y2 - thickness, x2, y2, color)
        rect(x1, y1, x1 + thickness, y2, color); rect(x2 - thickness, y1, x2, y2, color)

    dark, muted, green, red, blue = (30, 41, 59), (100, 116, 139), (22, 163, 74), (220, 38, 38), (37, 99, 235)

    if screenshot_type == "web_shell":
        rect(0, 0, width, 60, dark); text(22, 20, "ATLAS CART", (255, 255, 255), 2); text(505, 20, "CART 0", (255, 255, 255), 2)
        rect(28, 82, 612, 315, (255, 255, 255)); outline(28, 82, 612, 315, (203, 213, 225))
        text(52, 105, "PRODUCTS", dark, 2); text(52, 142, "ROUTE /PRODUCTS", muted, 1)
        rect(52, 184, 588, 234, (254, 242, 242)); outline(52, 184, 588, 234, red)
        text(70, 201, "CATALOG API IS UNAVAILABLE", red, 1); text(70, 218, "TRY AGAIN IN A MOMENT", dark, 1)
        text(52, 268, "SHARED HEADER / ROUTE / ERROR BOUNDARY", green, 1)
    elif screenshot_type == "product_list_fixed":
        rect(0, 0, width, 54, dark); text(22, 19, "PRODUCTS / STAGING / 375PX", (255, 255, 255), 2)
        for index, (name, price, stock) in enumerate([("OAK TRAY", "USD 28.00", "IN STOCK"), ("MONITOR STAND", "USD 74.50", "OUT OF STOCK")]):
            x = 44 + index * 292
            rect(x, 80, x + 260, 316, (255, 255, 255)); outline(x, 80, x + 260, 316, (203, 213, 225))
            rect(x + 18, 98, x + 242, 184, (226, 232, 240)); text(x + 25, 205, name, dark, 1)
            text(x + 25, 230, price, dark, 1); text(x + 25, 255, stock, green if index == 0 else red, 1)
            rect(x + 25, 278, x + 205, 304, blue if index == 0 else muted); text(x + 43, 286, "VIEW DETAILS", (255, 255, 255), 1)
    elif screenshot_type in {"product_detail_current", "product_detail_expected", "product_detail_fixed"}:
        label = {"product_detail_current": "CURRENT", "product_detail_expected": "EXPECTED", "product_detail_fixed": "STAGING AFTER"}[screenshot_type]
        rect(0, 0, width, 54, dark); text(22, 19, f"PRODUCT DETAIL / {label}", (255, 255, 255), 2)
        rect(120, 74, 520, 336, (255, 255, 255)); outline(120, 74, 520, 336, (203, 213, 225))
        if screenshot_type == "product_detail_current":
            text(145, 104, "C-311", muted, 1); text(145, 135, "MONITOR STAND", dark, 2)
            text(145, 180, "74.5", dark, 2); text(145, 215, "0", red, 2)
            text(145, 262, "RAW DATA / NO SHOPPER GUIDANCE", red, 1)
        else:
            rect(145, 96, 300, 232, (226, 232, 240)); text(325, 104, "MONITOR STAND", dark, 1)
            text(325, 139, "USD 74.50", dark, 2); text(325, 177, "OUT OF STOCK", red, 1)
            rect(325, 210, 490, 246, muted); text(341, 221, "ADD TO CART", (255, 255, 255), 1)
            text(145, 274, "IMAGE / NAME / PRICE / STOCK / ACTION", green if screenshot_type == "product_detail_fixed" else blue, 1)
            text(145, 302, "IMPLEMENTED IN STAGING" if screenshot_type == "product_detail_fixed" else "DESIGN REVIEW TARGET", dark, 1)
    elif screenshot_type == "bug_ui_fixed":
        rect(0, 0, width, 54, dark); text(22, 19, "STAGING AFTER / PRODUCT CARD 375PX", (255, 255, 255), 2)
        rect(32, 78, 608, 322, (255, 255, 255)); rect(56, 102, 208, 278, (226, 232, 240))
        text(235, 104, "ADJUSTABLE BAMBOO", dark, 2); text(235, 128, "MONITOR STAND", dark, 2)
        text(235, 170, "USD 39.00", dark, 2); text(235, 201, "IN STOCK", green, 2)
        rect(235, 242, 375, 276, blue); text(252, 252, "ADD TO CART", (255, 255, 255), 1)
        text(235, 300, "NO OVERLAP AT 320PX OR 375PX", green, 1)
    elif screenshot_type == "missing_image_fixed":
        rect(0, 0, width, 54, dark); text(22, 19, "CHROME / SLOW 3G / AFTER FIX", (255, 255, 255), 2)
        rect(32, 78, 608, 322, (255, 255, 255)); outline(32, 78, 608, 322, (208, 215, 222))
        rect(58, 105, 225, 274, (226, 232, 240)); outline(58, 105, 225, 274, (148, 163, 184))
        text(93, 178, "NO IMAGE", muted, 1); text(258, 112, "OAK TRAY", dark, 2)
        text(258, 151, "USD 28.00", dark, 2); text(258, 188, "IN STOCK", green, 2)
        text(258, 242, "PLACEHOLDER SHOWN DIRECTLY", green, 1); text(258, 260, "NO BROKEN ICON FLASH", green, 1)
    elif screenshot_type == "ci_pipeline":
        rect(0, 0, width, 58, (13, 17, 23)); text(22, 19, "ATLAS CART / CI RUN #184", (255, 255, 255), 2)
        rect(24, 78, 616, 332, (255, 255, 255)); outline(24, 78, 616, 332, (208, 215, 222))
        text(48, 96, "PULL REQUEST ATC-16", dark, 2); text(455, 96, "PASSED", green, 2)
        rows = [("BACKEND TESTS", "2M 18S"), ("FRONTEND CHECKS", "1M 06S"), ("IMAGE BUILD", "3M 41S"), ("MERGE GATE", "12S")]
        for index, (label, duration) in enumerate(rows):
            y = 136 + index * 43
            rect(48, y, 70, y + 22, (220, 252, 231)); outline(48, y, 70, y + 22, green)
            rect(54, y + 11, 59, y + 16, green); rect(58, y + 14, 64, y + 18, green); rect(63, y + 8, 67, y + 17, green)
            text(86, y + 4, label, dark, 2); text(488, y + 4, duration, muted, 2)
    elif screenshot_type == "architecture":
        rect(0, 0, width, 56, dark); text(22, 19, "ATC-30 CHECKOUT IDEMPOTENCY", (255, 255, 255), 2)
        boxes = [
            (30, 105, 145, 165, "WEB APP"), (190, 105, 315, 165, "GATEWAY"), (360, 95, 515, 175, "ORDER API"),
            (360, 235, 505, 295, "POSTGRES"), (535, 105, 630, 165, "PAYMENT"),
        ]
        for x1, y1, x2, y2, label in boxes:
            rect(x1, y1, x2, y2, (255, 255, 255)); outline(x1, y1, x2, y2, blue, 3)
            text(x1 + 12, y1 + 22, label, dark, 1)
        for x1, x2, y in [(145, 190, 135), (315, 360, 135), (515, 535, 135)]:
            rect(x1, y - 2, x2 - 8, y + 2, muted); rect(x2 - 12, y - 7, x2, y + 8, muted)
        rect(430, 175, 434, 235, muted); rect(424, 225, 440, 238, muted)
        text(348, 195, "IDEMPOTENCY KEY", muted, 1); text(350, 310, "ONE ORDER / ONE STOCK CHANGE", green, 1)
    elif screenshot_type == "order_incident":
        rect(0, 0, width, 54, dark); text(22, 19, "ATLAS CART / RECENT ORDERS", (255, 255, 255), 2)
        rect(32, 78, 608, 310, (255, 255, 255)); text(55, 94, "ORDER", muted, 1); text(210, 94, "STATUS", muted, 1); text(370, 94, "TOTAL", muted, 1)
        for y, reference in [(132, "BETA-1042"), (196, "BETA-1043")]:
            rect(50, y, 590, y + 48, (248, 250, 252)); text(67, y + 16, reference, dark, 2)
            text(218, y + 16, "PAID", green, 2); text(372, y + 16, "USD 39.00", dark, 2); text(520, y + 16, "09:05", muted, 1)
        outline(44, 124, 596, 252, red, 4); text(55, 272, "ONE CHECKOUT / TWO ORDER ROWS", red, 1)
    elif screenshot_type == "missing_image":
        rect(0, 0, width, 54, dark); text(22, 19, "CHROME / SLOW 3G / SKU B-208", (255, 255, 255), 2)
        rect(32, 78, 608, 322, (255, 255, 255)); outline(32, 78, 608, 322, (208, 215, 222))
        rect(58, 105, 225, 274, (248, 250, 252)); outline(58, 105, 225, 274, (203, 213, 225))
        outline(104, 145, 177, 218, red, 4)
        rect(115, 158, 127, 170, red); rect(154, 158, 166, 170, red)
        rect(124, 205, 158, 211, red); rect(119, 194, 130, 205, red); rect(152, 194, 163, 205, red)
        text(258, 112, "OAK TRAY", dark, 2); text(258, 151, "USD 28.00", dark, 2); text(258, 188, "IN STOCK", green, 2)
        outline(50, 96, 233, 282, red, 4); text(258, 242, "BROKEN ICON FLASHES", red, 1); text(258, 260, "BEFORE PLACEHOLDER", red, 1)
    elif screenshot_type == "empty_cart":
        rect(0, 0, width, 54, dark); text(22, 19, "EMPTY CART / 320PX VIEW", (255, 255, 255), 2)
        rect(176, 76, 464, 332, (255, 255, 255)); outline(176, 76, 464, 332, (203, 213, 225))
        text(228, 105, "YOUR CART IS EMPTY", dark, 1)
        outline(206, 136, 434, 268, red, 3); text(233, 188, "UNUSED EMPTY SPACE", red, 1)
        rect(220, 286, 420, 320, blue); text(250, 296, "BROWSE PRODUCTS", (255, 255, 255), 1)
        text(220, 337, "BUTTON STARTS BELOW VIEW", red, 1)
    elif screenshot_type == "m3_frontend":
        issue_key = context.split(" / ", 1)[0]
        is_before = phase != "after"
        accent = red if is_before else green
        state = "CURRENT" if is_before else "VERIFIED AFTER"
        rect(0, 0, width, 54, dark); text(22, 19, f"{issue_key} / {state}", (255, 255, 255), 2)
        rect(26, 72, 614, 338, (255, 255, 255)); outline(26, 72, 614, 338, (203, 213, 225))
        if issue_key in {"ATC-54", "ATC-55"}:
            text(54, 92, "ATLAS CART ACCOUNT", dark, 2)
            text(54, 132, "EMAIL", muted, 1); outline(54, 150, 390, 184, (148, 163, 184))
            text(68, 161, "shopper@example.test", dark, 1)
            text(54, 205, "PASSWORD", muted, 1); outline(54, 223, 390, 257, (148, 163, 184))
            text(68, 234, "************", dark, 1)
            rect(54, 278, 230, 312, blue); text(80, 289, "CREATE ACCOUNT" if issue_key == "ATC-54" else "SIGN IN", (255, 255, 255), 1)
            if is_before:
                text(420, 153, "ACCOUNT FLOW", red, 1); text(420, 176, "NOT AVAILABLE", red, 1)
            else:
                text(420, 153, "SESSION ACTIVE", green, 1); text(420, 176, "GUEST PATH KEPT", green, 1)
        elif issue_key == "ATC-62":
            text(52, 92, "CHECKOUT / SHIPPING", dark, 2)
            for index, (label, price, eta) in enumerate([("STANDARD", "$6.00", "4-6 DAYS"), ("EXPEDITED", "$14.00", "2 DAYS")]):
                y = 136 + index * 72; outline(52, y, 580, y + 52, blue if (not is_before and index == 0) else (148, 163, 184), 3)
                text(74, y + 17, label, dark, 2); text(310, y + 17, price, dark, 2); text(442, y + 17, eta, muted, 1)
            text(52, 294, "NO SHIPPING CHOICE" if is_before else "STANDARD SELECTED / TOTAL $81.50", accent, 1)
        elif issue_key == "ATC-64":
            text(52, 92, "OPERATIONS / ORDER SEARCH", dark, 2); outline(52, 132, 430, 170, (148, 163, 184)); text(68, 145, "BETA-1187", dark, 2)
            rect(452, 132, 570, 170, blue); text(478, 145, "SEARCH", (255, 255, 255), 1)
            rect(52, 198, 570, 286, (248, 250, 252)); text(72, 218, "BETA-1187", dark, 2)
            text(72, 250, "REFUNDED / USD 81.50", green if not is_before else red, 1)
            text(350, 250, "NO PRIVATE ADDRESS", accent, 1)
        elif issue_key == "ATC-69":
            text(52, 92, "SELLER PRODUCT EDITOR / C-311", dark, 2)
            for index, (label, value) in enumerate([("NAME", "BAMBOO DISPLAY RISER"), ("PRICE", "74.50"), ("STOCK", "12")]):
                y = 130 + index * 50; text(52, y, label, muted, 1); outline(145, y - 8, 540, y + 24, (148, 163, 184)); text(160, y + 2, value, dark, 1)
            rect(410, 292, 540, 324, blue if not is_before else muted); text(447, 302, "SAVE", (255, 255, 255), 1)
            text(52, 302, "READ ONLY" if is_before else "AUDIT REASON REQUIRED", accent, 1)
        elif issue_key == "ATC-74":
            text(52, 92, "CATALOG IMPORT / IMP-204", dark, 2)
            phases = [("QUEUED", True), ("VALIDATING 5,000 ROWS", not is_before), ("APPLIED", not is_before)]
            for index, (label, complete) in enumerate(phases):
                y = 138 + index * 58; rect(54, y, 78, y + 24, (220, 252, 231) if complete else (241, 245, 249)); outline(54, y, 78, y + 24, green if complete else muted)
                text(100, y + 5, label, dark, 2)
            text(52, 310, "STATUS STALE UNTIL REFRESH" if is_before else "LIVE STATUS / AUDIT LINK READY", accent, 1)
        else:  # ATC-78 accessibility pass
            text(52, 92, "CHECKOUT / KEYBOARD + SCREEN READER", dark, 2)
            outline(52, 132, 570, 182, accent, 4); text(72, 149, "CARD NUMBER / REQUIRED", dark, 2)
            outline(52, 208, 570, 258, blue if not is_before else red, 4); text(72, 225, "PLACE ORDER", dark, 2)
            text(52, 294, "FOCUS LOST / ERROR NOT ANNOUNCED" if is_before else "VISIBLE FOCUS / ERROR ANNOUNCED", accent, 1)
        text(440, 318, f"AC{ac_index}", muted, 1)
    elif screenshot_type == "m3_regression":
        issue_key = context.split(" / ", 1)[0]
        rect(0, 0, width, 58, (13, 17, 23)); text(22, 19, f"{issue_key} / FRONTEND REGRESSION", (255, 255, 255), 2)
        rect(24, 78, 616, 332, (255, 255, 255)); outline(24, 78, 616, 332, (208, 215, 222))
        rows = [("VISUAL ACCEPTANCE", "PASS"), ("KEYBOARD FLOW", "PASS"), ("API ERROR STATE", "PASS"), ("CHROME + SAFARI", "PASS")]
        for index, (label, result) in enumerate(rows):
            y = 105 + index * 50; rect(48, y, 72, y + 24, (220, 252, 231)); outline(48, y, 72, y + 24, green)
            text(92, y + 5, label, dark, 2); text(500, y + 5, result, green, 2)
        text(48, 306, "SYNTHETIC CI / BUILD beta-m3.7", muted, 1)
    elif screenshot_type == "frontend_evidence":
        is_before = phase != "after"
        accent = red if is_before else green
        header = context or f"AC{ac_index} / {'BEFORE' if is_before else 'AFTER'}"
        rect(0, 0, width, 54, dark)
        text(22, 19, header[:34], (255, 255, 255), 2)
        rect(32, 78, 608, 322, (255, 255, 255))
        outline(32, 78, 608, 322, accent, 4)
        text(52, 102, "CURRENT PROBLEM" if is_before else "FIX VERIFIED", accent, 2)
        y = 142
        for chunk in textwrap.wrap(caption, width=42)[:4]:
            text(52, y, chunk, dark, 1)
            y += 22
        text(52, 300, "SYNTHETIC STAGING SCREENSHOT", muted, 1)
    elif screenshot_type == "cart_controls":
        rect(0, 0, width, 54, dark); text(22, 19, "CART ROW / 320PX / QTY 10", (255, 255, 255), 2)
        rect(42, 85, 598, 300, (255, 255, 255)); outline(42, 85, 598, 300, (203, 213, 225))
        text(66, 108, "ADJUSTABLE BAMBOO", dark, 2); text(66, 133, "MONITOR STAND", dark, 2)
        rect(66, 185, 112, 229, (226, 232, 240)); text(82, 199, "-", dark, 2)
        outline(112, 185, 180, 229, (148, 163, 184)); text(133, 199, "10", dark, 2)
        rect(180, 185, 226, 229, (226, 232, 240)); text(195, 199, "+", dark, 2)
        rect(208, 185, 340, 229, (254, 226, 226)); text(233, 199, "REMOVE", red, 1)
        outline(174, 177, 348, 237, red, 4); text(66, 255, "PLUS OVERLAPS REMOVE", red, 1)
    else:
        rect(0, 0, width, 54, dark); text(22, 19, "STAGING / PRODUCT CARD 375PX", (255, 255, 255), 2)
        rect(32, 78, 608, 322, (255, 255, 255)); rect(56, 102, 208, 278, (226, 232, 240))
        text(235, 104, "ADJUSTABLE BAMBOO", dark, 2); text(235, 128, "MONITOR STAND", dark, 2)
        text(235, 164, "USD 39.00", red, 2); text(315, 164, "IN STOCK", green, 2)
        rect(235, 218, 375, 252, blue); text(252, 228, "ADD TO CART", (255, 255, 255), 1)
        outline(225, 154, 520, 194, red, 4); text(235, 285, "PRICE OVERLAPS STOCK LABEL", red, 1)

    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in pixels)

    def chunk(name: bytes, payload: bytes) -> bytes:
        body = name + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"tEXt", b"Description\x00" + caption.encode("latin-1", errors="replace"))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def attachment_bytes(attachment: dict[str, Any]) -> bytes:
    kind = attachment["kind"]
    if kind in {"text", "log", "markdown", "csv"}:
        return attachment["content"].encode("utf-8")
    if kind == "pdf":
        return make_pdf(attachment["title"], attachment.get("lines", []), attachment.get("sections"))
    if kind == "xlsx":
        return make_xlsx(attachment["sheet"], attachment["rows"])
    if kind == "screenshot":
        return make_screenshot(
            attachment["caption"],
            attachment.get("screenshot_type", "bug_ui"),
            ac_index=int(attachment.get("ac_index", 1)),
            phase=str(attachment.get("phase", "before")),
            context=str(attachment.get("context", "")),
        )
    raise ValueError(f"Unsupported attachment kind: {kind}")


DEFAULT_ATTACHMENT_RATIO = 20.0
ATTACHMENT_RATIO = DEFAULT_ATTACHMENT_RATIO


def attachment_target_count(ratio: float) -> int:
    if not math.isfinite(ratio) or ratio < 0 or ratio > 100:
        raise ValueError("Attachment ratio must be a number from 0 through 100")
    return min(len(ISSUES), math.ceil(len(ISSUES) * ratio / 100.0))


def issue_completion_time(item: dict[str, Any]) -> datetime | None:
    done_times = [at for at, _old, new in status_events(item) if new == "done"]
    return max(done_times) if done_times else None


def issue_evidence_cases(item: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    cases = []
    for index, criterion in enumerate(item["acceptance_criteria"], start=1):
        cases.append((
            f"AC{index}",
            f"AtlasCart M1 fixture prepared for {item['title']}",
            f"run the issue-specific check for AC{index}",
            criterion,
        ))
    return cases


def supplemental_table_rows(
    key: str,
    item: dict[str, Any],
    cases: list[tuple[str, str, str, str]],
) -> list[tuple[str, str]]:
    """Build CSV/XLSX rows tied to ACs, with context padding when an issue has fewer ACs."""
    rows = [(name, expected) for name, _setup, _action, expected in cases]
    padding = [
        ("Environment", "Run against the AtlasCart M1 synthetic fixture with no customer or payment data."),
        ("Scope boundary", f"Evidence stays within the acceptance criteria written for {key}."),
        ("Traceability", f"Captured from Jira issue {key} on snapshot date {AS_OF_DATE.isoformat()}."),
    ]
    pad_index = 0
    # Keep generated tabular evidence substantive even when an issue has only one or two ACs.
    while len(rows) < 3:
        label, text = padding[pad_index % len(padding)]
        rows.append((label, text))
        pad_index += 1
    return rows


def supplemental_attachment(item: dict[str, Any], extra_index: int) -> dict[str, Any]:
    """Create issue-specific evidence when coverage exceeds curated examples."""
    key = item["key"]
    done = item["status"] == "done"
    completion = issue_completion_time(item)
    if completion:
        created_at = max(item["created_at"] + timedelta(minutes=10), completion - timedelta(minutes=8))
    elif item["started_at"]:
        created_at = min(SNAPSHOT_AT - timedelta(hours=2), item["started_at"] + timedelta(hours=3))
    else:
        created_at = min(SNAPSHOT_AT - timedelta(hours=2), item["created_at"] + timedelta(minutes=10))

    placement = "comment"
    title = f"{key} - {item['title']} - {'acceptance report' if done else 'work plan'}"
    result = "PASS" if done else item["status"].replace("_", " ").upper()
    cases = issue_evidence_cases(item)
    kind_cycle = ("pdf", "markdown", "csv", "xlsx")
    kind = kind_cycle[extra_index % len(kind_cycle)]
    slug = "-".join("".join(char.lower() if char.isalnum() else " " for char in item["title"]).split()[:8])
    purpose_focus = "; ".join(item["acceptance_criteria"])

    common = {
        "issue": key,
        "uploader": "U10" if done else (item["reporter"] or "U02"),
        "placement": placement,
        "created_at": created_at,
        "generated": True,
        "comment_mode": "append",
        "purpose": (
            f"{'Completion evidence' if done else 'Working reference'} for {key} ({item['title']}) covering these issue-specific checks: "
            f"{purpose_focus} Attached to the timeline at the current {item['status'].replace('_', ' ')} state."
        ),
    }
    stem = f"{slug}-{'acceptance-report' if done else 'work-plan'}"
    if kind == "pdf":
        sections = [
            {"heading": "1. Context and scope", "lines": [
                f"Issue: {key} - {item['title']}",
                f"Current result: {result}",
                f"Owner: {USER_BY_ID[item['assignee'] or item['reporter']]['name']}",
                f"Primary labels: {', '.join(item['labels'])}",
                "- Scope and acceptance come from the Jira issue; this report does not add hidden requirements.",
                "- All examples use synthetic fixture data and contain no real customer details.",
                f"- Estimate decision: {item['estimate_rationale']}",
            ]},
            {"heading": "2. Acceptance matrix", "lines": [
                "The table keeps each check tied to the numbered acceptance criterion in Jira.",
                "CODE: build=atlascart-m1  environment=synthetic-local",
            ], "table": {
                "headers": ["AC", "Check", "Observed", "Result"],
                "rows": [[name, expected, expected if done else "Not run yet", "PASS" if done else "PENDING"] for name, _setup, _action, expected in cases],
            }},
            {"heading": "3. Result, risks, and follow-up", "lines": [
                f"Overall result at attachment time: {result}.",
                *([f"- {name} passed: {expected}" for name, _setup, _action, expected in cases] if done else [f"- {name} still needs evidence: {expected}" for name, _setup, _action, expected in cases]),
                "- Any failure remains visible on Jira instead of being edited out of this report.",
                "- Logs and screenshots use synthetic references and omit personal and payment data.",
            ]},
        ]
        return common | {"filename": f"{stem}.pdf", "content_type": "application/pdf", "kind": "pdf", "title": title, "sections": sections}
    if kind == "markdown":
        lines = [f"# {title}", "", "## Why this file exists", "", common["purpose"], "", "## Current state", "", f"- **Status:** {item['status'].replace('_', ' ')}", f"- **Owner:** {USER_BY_ID[item['assignee'] or item['reporter']]['name']}"]
        lines.extend([f"- **Estimate decision:** {item['estimate_rationale']}"])
        lines.extend(["", "## Acceptance checks", ""])
        for name, _setup, _action, expected in cases:
            lines.extend([f"### {name}", "", f"- Expected: {expected}", f"- Observed: {expected if done else 'Not run yet'}", f"- Result: {'PASS' if done else 'PENDING'}", ""])
        lines.extend(["## Reproduction command", "", "```text", f"dataset={DATASET_ID} issue={key} as_of={AS_OF_DATE.isoformat()}", "```", "", "## Data handling", "", "Synthetic references only. No real customer or payment data."])
        return common | {"filename": f"{stem}.md", "content_type": "text/markdown", "kind": "markdown", "content": "\n".join(lines) + "\n"}
    if kind == "csv":
        output = StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["issue", "acceptance_criterion", "plain_english_check", "observed_result", "status", "owner"])
        owner = USER_BY_ID[item['assignee'] or item['reporter']]["name"]
        for name, expected in supplemental_table_rows(key, item, cases):
            writer.writerow([key, name, expected, expected if done else "Not run yet", "PASS" if done else "PENDING", owner])
        writer.writerow([key, "Overall", "All numbered acceptance checks", "All passed" if done else "Work remains open", "PASS" if done else result, owner])
        return common | {"filename": f"{stem}.csv", "content_type": "text/csv", "kind": "csv", "content": output.getvalue()}
    if kind == "xlsx":
        owner = USER_BY_ID[item['assignee'] or item['reporter']]["name"]
        rows = [["Issue", "Acceptance criterion", "Plain-English check", "Observed result", "Status", "Owner"]]
        rows.extend([[key, name, expected, expected if done else "Not run yet", "PASS" if done else "PENDING", owner] for name, expected in supplemental_table_rows(key, item, cases)])
        rows.append([key, "Overall", "All numbered acceptance checks", "All passed" if done else "Work remains open", "PASS" if done else result, owner])
        return common | {"filename": f"{stem}.xlsx", "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "kind": "xlsx", "sheet": "Acceptance evidence", "rows": rows}
    raise AssertionError(f"Unexpected supplemental attachment kind: {kind}")


def merge_visual_evidence(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visual_issues = frontend_visual_issue_keys()
    selected_issues = {attachment["issue"] for attachment in attachments}
    filtered = [
        attachment for attachment in attachments
        if not (
            attachment["issue"] in visual_issues
            and attachment["kind"] == "screenshot"
            and attachment["placement"] in {"description", "comment"}
        )
    ]
    filtered.extend(
        row for row in all_visual_attachments(ISSUE_BY_KEY)
        if row["issue"] in selected_issues
    )
    return filtered


def build_attachment_set(ratio: float) -> list[dict[str, Any]]:
    target = attachment_target_count(ratio)
    all_curated_keys = list(dict.fromkeys(attachment["issue"] for attachment in CURATED_ATTACHMENTS))
    default_evidence_priority = [
        "ATC-7", "ATC-10", "ATC-12", "ATC-13", "ATC-14",
        "ATC-16", "ATC-19", "ATC-21", "ATC-30", "ATC-36",
    ]
    curated_keys = default_evidence_priority + [key for key in all_curated_keys if key not in default_evidence_priority]
    evidence_priority = [key for key in [
        "ATC-9", "ATC-30", "ATC-44", "ATC-45", "ATC-17", "ATC-34",
        "ATC-20", "ATC-26", "ATC-29", "ATC-32", "ATC-47", "ATC-42",
    ] if key not in curated_keys]
    remaining_keys = evidence_priority + [
        item["key"] for item in ISSUES
        if item["key"] not in curated_keys and item["key"] not in evidence_priority
    ]
    coverage_order = curated_keys + remaining_keys
    selected_keys = set(coverage_order[:target])
    selected = [attachment.copy() for attachment in CURATED_ATTACHMENTS if attachment["issue"] in selected_keys]
    for extra_index, key in enumerate(key for key in coverage_order[:target] if key not in curated_keys):
        selected.append(supplemental_attachment(ISSUE_BY_KEY[key], extra_index))
    return sorted(
        merge_visual_evidence(selected),
        key=lambda attachment: (attachment["created_at"], attachment["issue"], attachment["filename"]),
    )


ATTACHMENTS = build_attachment_set(DEFAULT_ATTACHMENT_RATIO)
COMMENTS = comments_for_links(ISSUE_LINKS, ATTACHMENTS)


def sprint_change_events() -> list[tuple[str, datetime, str | None, str | None, str]]:
    events = [
        ("ATC-10", ts(13, 16, 50), "AtlasCart Sprint 1", "AtlasCart Sprint 2", "Carried into Sprint 2 after the duplicate-SKU rule was clarified late"),
        ("ATC-22", ts(13, 16, 52), "AtlasCart Sprint 1", "AtlasCart Sprint 2", "Carried with its catalog import parent"),
        ("ATC-36", S2_REMOVED["ATC-36"][1], "AtlasCart Sprint 2", "AtlasCart Sprint 3", S2_REMOVED["ATC-36"][2]),
        ("ATC-37", S2_REMOVED["ATC-37"][1], "AtlasCart Sprint 2", "AtlasCart Sprint 3", S2_REMOVED["ATC-37"][2]),
        ("ATC-38", S2_REMOVED["ATC-38"][1], "AtlasCart Sprint 2", None, S2_REMOVED["ATC-38"][2]),
        ("ATC-39", ts(16, 15, 40), None, "AtlasCart Sprint 2", "Added after QA found the narrow-screen problem"),
        ("ATC-39", S2_REMOVED["ATC-39"][1], "AtlasCart Sprint 2", "AtlasCart Sprint 3", S2_REMOVED["ATC-39"][2]),
    ]
    for offset, key in enumerate(S2_CARRIED):
        events.append((
            key,
            ts(27, 17, offset),
            "AtlasCart Sprint 2",
            "AtlasCart Sprint 3",
            S2_CARRY_REASONS[key],
        ))
    return events


def build_histories() -> list[History]:
    rows: list[History] = []

    for item in ISSUES:
        rows.append(History(item["key"], item["reporter"], "issue_created", None, None, None, "Issue created", item["created_at"]))
        for at, old, new in status_events(item):
            rows.append(History(item["key"], item["assignee"], "field_change", "status", old, new, "Updated status", at))

    for key, (old, new, at, reason) in ESTIMATE_CHANGES.items():
        rows.append(History(key, "U03", "field_change", "storyPoints", str(old), str(new), f"Updated storyPoints — {reason}", at))

    for key, (old_user, new_user, at, reason) in ASSIGNEE_CHANGES.items():
        rows.append(History(key, "U03", "field_change", "assignee", USER_BY_ID[old_user]["name"], USER_BY_ID[new_user]["name"], f"Updated assignee — {reason}", at))

    for key, at, old, new, reason in sprint_change_events():
        rows.append(History(key, "U03", "field_change", "sprint", old, new, f"Updated sprint — {reason}", at))

    for comment in COMMENTS:
        rows.append(History(comment["issue"], comment["author"], "comment_created", None, None, None, "added a comment", comment["created_at"]))

    for source, target, link_type, at in ISSUE_LINKS:
        actor = ISSUE_BY_KEY[source]["reporter"]
        rows.append(History(source, actor, "link_created", None, None, None, "updated the Link", at))
        rows.append(History(source, actor, "field_change", "Link", "None", f"{link_type} {target}", "Updated Link", at + timedelta(seconds=1)))
        rows.append(History(target, actor, "link_created", None, None, None, "updated the Link", at + timedelta(seconds=2)))

    for link in CODE_LINKS:
        label = f"{link.kind.replace('_', ' ')} {link.ref_id}"
        rows.append(History(link.issue, link.creator, "code_link_added", None, None, None, f"linked {label}", link.created_at))
        rows.append(History(link.issue, link.creator, "field_change", "Development", "None", label, "Updated Development", link.created_at + timedelta(seconds=1)))

    for attachment in ATTACHMENTS:
        if attachment["placement"] != "panel":
            continue
        rows.append(History(attachment["issue"], attachment["uploader"], "attachment_uploaded", None, None, None, "attached a file", attachment["created_at"]))
        rows.append(History(attachment["issue"], attachment["uploader"], "field_change", "Attachment", "None", attachment["filename"], "Updated Attachment", attachment["created_at"] + timedelta(seconds=1)))

    return sorted(rows, key=lambda row: (row.created_at, row.issue, row.event_type, row.field_name or ""))


HISTORIES = build_histories()


def configure_dataset(attachment_ratio: float, link_ratio: float) -> None:
    global ATTACHMENT_RATIO, LINK_RATIO, ATTACHMENTS, ISSUE_LINKS, COMMENTS, HISTORIES
    link_target_count(link_ratio)
    ISSUE_LINKS = build_issue_link_set(link_ratio)
    LINK_RATIO = link_ratio
    attachment_target_count(attachment_ratio)
    ATTACHMENT_RATIO = attachment_ratio
    ATTACHMENTS = build_attachment_set(attachment_ratio)
    COMMENTS = comments_for_links(ISSUE_LINKS, ATTACHMENTS)
    HISTORIES = build_histories()


def sprint_history_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    s1_started = ts(2, 10, 30)
    s1_completed = ts(13, 16, 45)
    for key in S1_INITIAL + S1_ADDED:
        item = ISSUE_BY_KEY[key]
        initial = key in S1_INITIAL
        carried = key in S1_CARRIED
        points_at_start = points_at(key, s1_started) if initial else None
        points_at_end = points_at(key, s1_completed)
        rows.append({
            "sprint": "S1",
            "issue": key,
            "type": item["type"],
            "initial_scope": initial,
            "points_at_start": points_at_start,
            "points_at_end": points_at_end,
            "status_at_end": "in_progress" if carried else "done",
            "outcome": "carried_over" if carried else "completed",
            "added_at": s1_started if initial else item["created_at"] + timedelta(minutes=10),
            "removed_at": None,
            "finalized_at": s1_completed,
        })

    s2_started = ts(16, 10, 30)
    s2_completed = ts(27, 16, 55)
    for key in S2_INITIAL + S2_ADDED:
        item = ISSUE_BY_KEY[key]
        initial = key in S2_INITIAL
        removed = key in S2_REMOVED
        carried = key in S2_CARRIED
        rows.append({
            "sprint": "S2",
            "issue": key,
            "type": item["type"],
            "initial_scope": initial,
            "points_at_start": points_at(key, s2_started) if initial else None,
            "points_at_end": points_at(key, s2_completed),
            "status_at_end": "in_progress" if carried else (item["status"] if removed else "done"),
            "outcome": "removed" if removed else ("carried_over" if carried else "completed"),
            "added_at": s2_started if initial else item["created_at"] + timedelta(minutes=10),
            "removed_at": S2_REMOVED[key][1] if removed else None,
            "finalized_at": s2_completed,
        })

    s3_started = ts(30, 10, 0)
    for key in S3_INITIAL:
        item = ISSUE_BY_KEY[key]
        rows.append({
            "sprint": "S3",
            "issue": key,
            "type": item["type"],
            "initial_scope": True,
            "points_at_start": points_at(key, s3_started),
            "points_at_end": None,
            "status_at_end": None,
            "outcome": None,
            "added_at": s3_started,
            "removed_at": None,
            "finalized_at": None,
        })
    return rows


SPRINT_HISTORY = sprint_history_rows()


def completed_sprint_metrics(sprint_id: str) -> tuple[int, int, int, int, int, int, int, int]:
    rows = [row for row in SPRINT_HISTORY if row["sprint"] == sprint_id]
    estimatable = [row for row in rows if row["type"] in {"story", "task", "bug"}]
    initial_points = sum(row["points_at_start"] or 0 for row in estimatable if row["initial_scope"])
    initial_completed = sum(row["points_at_start"] or 0 for row in estimatable if row["initial_scope"] and row["outcome"] == "completed")
    final_rows = [row for row in rows if row["outcome"] != "removed"]
    final_estimatable = [row for row in estimatable if row["outcome"] != "removed"]
    final_points = sum(row["points_at_end"] or 0 for row in final_estimatable)
    completed_points = sum(row["points_at_end"] or 0 for row in estimatable if row["outcome"] == "completed")
    initial_issue_count = len([row for row in rows if row["initial_scope"]])
    completed_issue_count = len([row for row in rows if row["outcome"] == "completed"])
    unestimated_issue_count = len([row for row in final_estimatable if not row["points_at_end"]])
    return (
        initial_points, initial_completed, final_points, completed_points,
        initial_issue_count, completed_issue_count, len(final_rows), unestimated_issue_count,
    )


def validate() -> list[str]:
    checks: list[str] = []

    def passed(message: str) -> None:
        checks.append(message)

    keys = [item["key"] for item in ISSUES]
    assert keys == [f"ATC-{number}" for number in range(1, 49)]
    assert len(set(keys)) == 48
    passed("48 unique sequential issue keys")

    type_counts = Counter(item["type"] for item in ISSUES)
    assert dict(type_counts) == EXPECTED_TYPE_COUNTS, (type_counts, EXPECTED_TYPE_COUNTS)
    assert set(type_counts) <= ALLOWED_TYPES
    passed(f"Issue type mix matches the approved design: {dict(type_counts)}")

    estimatable_keys = {item["key"] for item in ISSUES if item["type"] in {"story", "task", "bug"}}
    assert set(POINT_RATIONALES) == estimatable_keys
    assert len(set(POINT_RATIONALES.values())) == len(POINT_RATIONALES)
    for key in estimatable_keys:
        points = ISSUE_BY_KEY[key]["points"]
        assert points in {1, 2, 3, 5, 8, 13}
        assert POINT_RATIONALES[key].startswith(f"{points} — ")
        assert len(POINT_RATIONALES[key]) >= 150
        assert POINT_RATIONALES[key] in ISSUE_BY_KEY[key]["description"]
    passed(f"All {len(estimatable_keys)} estimated issues have unique, issue-specific scope, risk, and boundary reasoning")

    user_ids = set(USER_BY_ID)
    sprint_ids = {sprint["id"] for sprint in SPRINTS}
    label_names = set(LABELS)
    for item in ISSUES:
        assert item["status"] in ALLOWED_STATUSES
        assert item["priority"] in ALLOWED_PRIORITIES
        assert item["reporter"] in user_ids
        assert item["assignee"] is None or item["assignee"] in user_ids
        assert item["sprint"] is None or item["sprint"] in sprint_ids
        assert set(item["labels"]) <= label_names
        assert item["created_at"] <= SNAPSHOT_AT
        assert "Estimate reasoning" in item["description"]
        assert item["estimate_rationale"] in item["description"]
        assert 1 <= len(item["acceptance_criteria"]) <= 3
        assert "Acceptance criteria" in item["description"]
        for index, criterion in enumerate(item["acceptance_criteria"], start=1):
            assert len(criterion) >= 35
            assert f"AC{index}: {criterion}" in item["description"]
        if item["type"] == "bug":
            for heading in ("Bug example", "When / environment:", "Steps to reproduce:", "Expected behavior:", "Actual behavior:"):
                assert heading in item["description"]
        if item["parent"]:
            assert item["parent"] in ISSUE_BY_KEY
            assert int(item["parent"].split("-")[1]) < int(item["key"].split("-")[1])
        if item["type"] == "epic":
            assert item["parent"] is None and item["sprint"] is None and item["points"] is None
        if item["type"] == "subtask":
            assert item["parent"] and ISSUE_BY_KEY[item["parent"]]["type"] != "epic"
            assert item["points"] is None
        if item["sprint"] and item["type"] in {"story", "task", "bug"}:
            assert item["points"] and item["points"] > 0
        if item["status"] == "done" and item["type"] != "epic":
            if item["started_at"] or item["done_at"]:
                assert item["started_at"] and item["done_at"] and item["started_at"] < item["done_at"]
            else:
                # A duplicate closed during triage never entered implementation.
                assert status_events(item) == [(STATUS_OVERRIDES[item["key"]][0][0], "planned", "done")]
                assert any(item["key"] in (source, target) and kind == "duplicates" for source, target, kind, _at in CANONICAL_ISSUE_LINKS)
        for at, old, new in status_events(item):
            assert item["created_at"] <= at <= SNAPSHOT_AT
            assert old in ALLOWED_STATUSES and new in ALLOWED_STATUSES
    parent_subtasks = {
        item["key"]: sorted(
            child["key"] for child in ISSUES
            if child["parent"] == item["key"] and child["type"] == "subtask"
        )
        for item in ISSUES
        if any(child["parent"] == item["key"] and child["type"] == "subtask" for child in ISSUES)
    }
    assert set(SUBTASK_RATIONALE) == set(parent_subtasks)
    for parent_key, child_keys in parent_subtasks.items():
        rationale_keys = [child for child, _reason in SUBTASK_RATIONALE[parent_key]]
        assert rationale_keys == child_keys, f"{parent_key} subtask rationale keys must match children"
        rendered = description_html(ISSUE_BY_KEY[parent_key])
        assert "<h3>Subtasks</h3>" in rendered
        issue_comments = [comment["content"] for comment in COMMENTS if comment["issue"] == parent_key]
        for child_key, reason in SUBTASK_RATIONALE[parent_key]:
            assert child_key in rendered
            assert reason[:48] in rendered
            assert any(child_key in comment for comment in issue_comments), (
                f"{parent_key} should mention {child_key} in at least one comment"
            )
    passed(
        f"All {len(parent_subtasks)} parent issues explain why each of their "
        f"{sum(len(keys) for keys in parent_subtasks.values())} subtasks exists"
    )
    ac_counts = [len(item["acceptance_criteria"]) for item in ISSUES]
    ones = sum(1 for count in ac_counts if count == 1)
    twos = sum(1 for count in ac_counts if count == 2)
    threes = sum(1 for count in ac_counts if count == 3)
    assert ones >= 10, f"expected at least 10 single-AC issues, got {ones}"
    assert twos >= 20, f"expected at least 20 two-AC issues, got {twos}"
    assert threes >= 4, f"expected at least 4 three-AC issues, got {threes}"
    assert ones + twos + threes == len(ISSUES)
    passed(
        f"Acceptance criteria mix is realistic: {ones} with 1 AC, {twos} with 2 AC, {threes} with 3 AC"
    )
    passed("All issue descriptions include plain-English AC; every bug includes environment, example, steps, expected, and actual behavior")

    for comment in COMMENTS:
        assert comment["issue"] in ISSUE_BY_KEY
        assert comment["author"] in user_ids
        assert ISSUE_BY_KEY[comment["issue"]]["created_at"] <= comment["created_at"] <= SNAPSHOT_AT
        assert comment["content"].strip()
        rendered = comment_html(comment)
        assert rendered.startswith("<p>")
        assert "\n" not in rendered or "</p>\n" in rendered or "</li>\n" in rendered
    passed(f"{len(COMMENTS)} comments occur after issue creation and before the snapshot")

    human_events: list[tuple[str, str, datetime]] = []
    for item in ISSUES:
        human_events.append((f"{item['key']} created", item["reporter"], item["created_at"]))
        if item["started_at"]:
            human_events.append((f"{item['key']} started", item["assignee"], item["started_at"]))
        completion = issue_completion_time(item)
        if completion:
            human_events.append((f"{item['key']} completed", item["assignee"] or item["reporter"], completion))
    human_events.extend((f"{comment['issue']} comment", comment["author"], comment["created_at"]) for comment in COMMENTS)
    human_events.extend((f"{link.issue} code link", link.creator, link.created_at) for link in CODE_LINKS)
    human_events.extend((f"{row['issue']} attachment", row["uploader"], row["created_at"]) for row in ATTACHMENTS)
    human_events.extend((f"{row.issue} {row.event_type}", row.actor, row.created_at) for row in HISTORIES if row.actor)
    for label, actor, at in human_events:
        local = at.astimezone(ZoneInfo(USER_BY_ID[actor]["timezone"]))
        assert local.weekday() < 5, f"Weekend human event without incident reason: {label} at {local.isoformat()}"
        assert 7 <= local.hour <= 20, f"Human event outside working window: {label} at {local.isoformat()}"
    assert len({at.utcoffset() for _label, _actor, at in human_events}) >= 3
    passed(f"All {len(human_events)} human events occur on weekdays in the actor's Pacific, Mountain, Central, or Eastern work window")

    for item in ISSUES:
        rendered = description_html(item)
        assert rendered.count("<h3>") >= 3
        assert rendered.count("<li>") >= len(item["acceptance_criteria"])
    passed("Issue descriptions and comments render as browser-visible paragraphs and lists")

    done_keys = {item["key"] for item in ISSUES if item["status"] == "done"}
    verification_by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for comment in COMMENTS:
        if comment.get("verification"):
            verification_by_issue[comment["issue"]].append(comment)
            assert comment["content"].startswith("Verification result: ")
    assert set(verification_by_issue) == done_keys
    for key in done_keys:
        assert len(verification_by_issue[key]) == 1
        verification = verification_by_issue[key][0]
        rendered = comment_html(verification)
        assert "<p><strong>Verification result</strong></p>" in rendered
        assert rendered.count("<li>") == len(ISSUE_BY_KEY[key]["acceptance_criteria"])
        assert verification["created_at"] <= issue_completion_time(ISSUE_BY_KEY[key])
        for index, criterion in enumerate(ISSUE_BY_KEY[key]["acceptance_criteria"], start=1):
            assert f"AC{index} — PASS — {criterion}" in verification["content"]
    passed(f"All {len(done_keys)} Done issues have one close comment that verifies every numbered AC")

    for source, target, link_type, at in ISSUE_LINKS:
        assert source in ISSUE_BY_KEY and target in ISSUE_BY_KEY and source != target
        assert link_type in ALLOWED_LINKS
        assert max(ISSUE_BY_KEY[source]["created_at"], ISSUE_BY_KEY[target]["created_at"]) <= at <= SNAPSHOT_AT
    assert len(set((s, t, kind) for s, t, kind, _ in ISSUE_LINKS)) == len(ISSUE_LINKS)
    assert {(s, t, kind) for s, t, kind, _ in ISSUE_LINKS} <= set(LINK_NARRATIVES)
    for source, target, kind, _at in ISSUE_LINKS:
        contextual = [comment for comment in COMMENTS if comment.get("link_context") == (source, target, kind)]
        phases = {comment.get("link_phase") for comment in contextual}
        if kind == "blocks":
            assert {"blocked-source", "blocked-target"} <= phases
            if issue_completion_time(ISSUE_BY_KEY[source]):
                assert {"unblocked-source", "unblocked-target"} <= phases
        elif kind == "relates to":
            assert phases == {"related-source", "related-target"}
        else:
            assert phases == {"duplicate-source", "duplicate-target"}
        assert all(source in comment["content"] or target in comment["content"] for comment in contextual)
    linked_issue_count = len({key for source, target, _kind, _at in ISSUE_LINKS for key in (source, target)})
    assert linked_issue_count == link_target_count(LINK_RATIO)
    passed(
        f"{len(ISSUE_LINKS)} links cover {linked_issue_count}/{len(ISSUES)} issues for requested {LINK_RATIO:g}%; "
        "selected links retain two-sided lifecycle comments"
    )

    for link in CODE_LINKS:
        assert link.issue in ISSUE_BY_KEY and link.creator in user_ids
        assert ISSUE_BY_KEY[link.issue]["created_at"] <= link.created_at <= link.last_activity_at <= SNAPSHOT_AT
        assert link.kind in {"pull_request", "commit"}
    pull_requests_by_repo: defaultdict[str, list[CodeLink]] = defaultdict(list)
    for link in CODE_LINKS:
        if link.kind == "pull_request":
            pull_requests_by_repo[link.repo].append(link)
    for repo, pull_requests in pull_requests_by_repo.items():
        chronological = sorted(pull_requests, key=lambda link: (link.created_at, int(link.ref_id)))
        numbers = [int(link.ref_id) for link in chronological]
        assert all(left < right for left, right in zip(numbers, numbers[1:])), f"{repo} PR numbers do not follow creation order"
    passed(f"{len(CODE_LINKS)} deterministic GitHub fixture links have causal timestamps and repo-local PR numbering")

    for attachment in ATTACHMENTS:
        assert attachment["issue"] in ISSUE_BY_KEY
        assert attachment["uploader"] in user_ids
        assert ISSUE_BY_KEY[attachment["issue"]]["created_at"] <= attachment["created_at"] <= SNAPSHOT_AT
        assert attachment["placement"] in {"panel", "description", "comment"}
        assert len(attachment.get("purpose", "")) >= 60
        if attachment["placement"] == "description":
            item = ISSUE_BY_KEY[attachment["issue"]]
            # A description may gain a reproduction/design artifact during
            # refinement, but it must be present before implementation starts.
            assert item["started_at"] is None or attachment["created_at"] <= item["started_at"] + timedelta(hours=4)
            assert "verification" not in attachment["filename"].lower()
        binary = attachment_bytes(attachment)
        assert binary
        if attachment["kind"] == "pdf":
            assert binary.startswith(b"%PDF-1.4")
            assert binary.count(b"/Type /Page ") >= 3
            assert len(binary) >= 4000
            assert any(section.get("table") for section in attachment.get("sections", []))
            assert b"/BaseFont /Helvetica-Bold" in binary
            assert binary.count(b"0.07 0.12 0.22 rg") >= 3
            assert binary.count(b"0.05 0.55 0.52 rg") >= 3
        elif attachment["kind"] == "csv":
            rows = list(csv.reader(StringIO(attachment["content"])))
            assert len(rows) >= 5 and len(rows[0]) >= 4
            assert all(len(row) == len(rows[0]) for row in rows)
        elif attachment["kind"] == "xlsx":
            with zipfile.ZipFile(BytesIO(binary)) as workbook:
                assert workbook.testzip() is None
                sheet_xml = workbook.read("xl/worksheets/sheet1.xml")
                assert sheet_xml.count(b"<row ") >= 5
        elif attachment["kind"] == "screenshot":
            assert binary.startswith(b"\x89PNG\r\n\x1a\n") and len(binary) >= 2000
            assert attachment.get("screenshot_type") in {
                "bug_ui", "bug_ui_fixed", "ci_pipeline", "order_incident", "architecture",
                "missing_image", "missing_image_fixed", "empty_cart", "cart_controls",
                "web_shell", "product_list_fixed", "product_detail_current",
                "product_detail_expected", "product_detail_fixed", "frontend_evidence",
            }
        elif attachment["kind"] == "log":
            assert len(attachment["content"].splitlines()) >= 5
            assert "request_id=" in attachment["content"] and "retry_count=" in attachment["content"]
        elif attachment["kind"] == "markdown":
            assert attachment["content"].startswith("# ")
            assert attachment["content"].count("## ") >= 4
            assert "```" in attachment["content"]
        elif attachment["kind"] == "text":
            assert len(attachment["content"].splitlines()) >= 4
    allowed_content_types = {
        "application/pdf", "text/plain", "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "image/png", "text/markdown",
    }
    assert {attachment["content_type"] for attachment in ATTACHMENTS} <= allowed_content_types
    covered_issue_keys = {attachment["issue"] for attachment in ATTACHMENTS}
    assert len(covered_issue_keys) == attachment_target_count(ATTACHMENT_RATIO)
    for attachment in ATTACHMENTS:
        placeholder = f"[[ATTACHMENT:{attachment['filename']}]]"
        matches = [comment for comment in COMMENTS if placeholder in comment["content"]]
        if attachment["placement"] == "comment":
            if attachment.get("comment_mode") == "append":
                assert not matches
                assert any(comment["issue"] == attachment["issue"] and comment["created_at"] >= attachment["created_at"] for comment in COMMENTS)
            else:
                assert len(matches) == 1 and matches[0]["issue"] == attachment["issue"]
                assert attachment["created_at"] <= matches[0]["created_at"]
        else:
            assert not matches
            if attachment["placement"] == "panel":
                issue_comments = [comment["content"].lower() for comment in COMMENTS if comment["issue"] == attachment["issue"]]
                filename = attachment["filename"].lower()
                assert any(filename in content or ("attached" in content and any(word in content for word in filename.replace(".", "-").split("-") if len(word) >= 6)) for content in issue_comments)
    actual_ratio = len(covered_issue_keys) / len(ISSUES) * 100
    passed(f"Attachment coverage is {len(covered_issue_keys)}/{len(ISSUES)} issues ({actual_ratio:.2f}%) for requested {ATTACHMENT_RATIO:g}%")
    if ATTACHMENTS:
        passed("Attachments have stated purposes and pass depth checks: designed high-contrast PDFs, styled XLSX, contextual CSV/Markdown/log, or issue-specific PNG")

    for issue_key in frontend_visual_issue_keys():
        if issue_key not in covered_issue_keys:
            continue
        item = ISSUE_BY_KEY[issue_key]
        ac_count = len(item["acceptance_criteria"])
        description_pngs = sorted(
            [
                attachment for attachment in ATTACHMENTS
                if attachment["issue"] == issue_key
                and attachment["placement"] == "description"
                and attachment["kind"] == "screenshot"
            ],
            key=lambda attachment: int(attachment.get("ac_index", 0)),
        )
        assert len(description_pngs) == ac_count, f"{issue_key} description PNG count must match AC count"
        assert [int(attachment.get("ac_index", 0)) for attachment in description_pngs] == list(range(1, ac_count + 1))
        if item["status"] == "done":
            verification = verification_by_issue[issue_key][0]
            after_pngs = sorted(
                [
                    attachment for attachment in ATTACHMENTS
                    if attachment["issue"] == issue_key
                    and attachment["placement"] == "comment"
                    and attachment["kind"] == "screenshot"
                    and attachment.get("phase") == "after"
                ],
                key=lambda attachment: int(attachment.get("ac_index", 0)),
            )
            assert len(after_pngs) == ac_count, f"{issue_key} verification PNG count must match AC count"
            for index, attachment in enumerate(after_pngs, start=1):
                assert f"Evidence AC{index}: [[ATTACHMENT:{attachment['filename']}]]" in verification["content"]
    passed(
        f"{len(frontend_visual_issue_keys())} frontend issues embed one problem PNG per AC in description "
        "and one fixed PNG per AC in verification when Done"
    )

    sprint_pairs = [(row["sprint"], row["issue"]) for row in SPRINT_HISTORY]
    assert len(sprint_pairs) == len(set(sprint_pairs))
    assert set(S1_CARRIED) <= set(S2_INITIAL)
    assert set(S2_CARRY_REASONS) == set(S2_CARRIED)
    expected_carry_transitions = {
        **{key: ("AtlasCart Sprint 1", "AtlasCart Sprint 2") for key in S1_CARRIED},
        **{key: ("AtlasCart Sprint 2", "AtlasCart Sprint 3") for key in S2_CARRIED},
    }
    for key, (expected_old, expected_new) in expected_carry_transitions.items():
        matches = [
            reason for event_key, _at, old, new, reason in sprint_change_events()
            if event_key == key and old == expected_old and new == expected_new
        ]
        assert len(matches) == 1 and len(matches[0]) >= 20
        if key in S2_CARRY_REASONS:
            assert matches[0] == S2_CARRY_REASONS[key]
    assert {row["issue"] for row in SPRINT_HISTORY if row["sprint"] == "S1"} == set(S1_INITIAL + S1_ADDED)
    assert {row["issue"] for row in SPRINT_HISTORY if row["sprint"] == "S2"} == set(S2_INITIAL + S2_ADDED)
    assert {row["issue"] for row in SPRINT_HISTORY if row["sprint"] == "S3"} == set(S3_INITIAL)
    current_s2 = {item["key"] for item in ISSUES if item["sprint"] == "S2"}
    completed_s2 = {row["issue"] for row in SPRINT_HISTORY if row["sprint"] == "S2" and row["outcome"] == "completed"}
    current_s3 = {item["key"] for item in ISSUES if item["sprint"] == "S3"}
    assert current_s2 == completed_s2
    assert current_s3 == set(S3_INITIAL)
    assert set(S2_CARRIED) <= current_s3
    assert all(item["status"] == "done" for item in ISSUES if item["sprint"] == "S1")
    assert all(item["status"] == "done" for item in ISSUES if item["sprint"] == "S2")
    passed(f"{len(SPRINT_HISTORY)} sprint-history rows preserve two closed sprints, active work, and issue-specific carry-over reasons")

    assert completed_sprint_metrics("S1") == (42, 37, 50, 42, 12, 14, 16, 0)
    passed("Sprint 1 metrics reconcile after estimate review: 42 committed, 37 initial completed, 50 final scope, 42 completed")

    s2_metrics = completed_sprint_metrics("S2")
    assert all(metric >= 0 for metric in s2_metrics)
    assert s2_metrics[2] >= s2_metrics[3]
    passed(f"Sprint 2 metrics reconcile with completed, removed, and {len(S2_CARRIED)} carried items")

    assert len(HISTORIES) >= 320
    assert all(ISSUE_BY_KEY[row.issue]["created_at"] <= row.created_at <= SNAPSHOT_AT for row in HISTORIES)
    passed(f"{len(HISTORIES)} audit rows form a valid historical timeline")

    active_sprint = next(sprint for sprint in SPRINTS if sprint["status"] == "active")
    assert SNAPSHOT_AT.date() == AS_OF_DATE
    active_start = date.fromisoformat(active_sprint["start_date"])
    active_end = date.fromisoformat(active_sprint["end_date"])
    assert active_start <= AS_OF_DATE <= active_end + timedelta(days=2)
    assert AS_OF_DATE <= active_end or AS_OF_DATE.weekday() in {5, 6}
    assert Counter(sprint["status"] for sprint in SPRINTS) == {"completed": 2, "active": 1}
    passed(f"Snapshot {AS_OF_DATE.isoformat()} is inside the active sprint window or its weekend close grace after two closed sprints")
    return checks


def q(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, datetime):
        value = value.isoformat()
    text = str(value).replace("'", "''")
    return f"'{text}'"


def description_html(item: dict[str, Any]) -> str:
    """Render issue sections as readable rich text instead of collapsed newlines."""
    parts = ["<h3>Context</h3>", f"<p>{html_escape(item['context'])}</p>"]
    report = BUG_REPORT_DETAILS.get(item["key"])
    if report:
        parts.extend([
            "<h3>Bug report</h3>",
            "<ul>",
            f"<li><strong>When / environment:</strong> {html_escape(report['when'])}</li>",
            f"<li><strong>Steps to reproduce:</strong> {html_escape(report['steps'])}</li>",
            f"<li><strong>Expected behavior:</strong> {html_escape(report['expected'])}</li>",
            f"<li><strong>Actual behavior:</strong> {html_escape(report['actual'])}</li>",
            "</ul>",
        ])
    subtasks = SUBTASK_RATIONALE.get(item["key"])
    if subtasks:
        parts.extend(["<h3>Subtasks</h3>", "<ul>"])
        for child_key, reason in subtasks:
            child = ISSUE_BY_KEY[child_key]
            parts.append(
                f"<li><strong>{html_escape(child_key)} — {html_escape(child['title'])}</strong>"
                f"<br>{html_escape(reason)}</li>"
            )
        parts.extend(["</ul>"])
    parts.extend(["<h3>Acceptance criteria</h3>", "<ul>"])
    parts.extend(
        f"<li><strong>AC{index}</strong> — {html_escape(criterion)}</li>"
        for index, criterion in enumerate(item["acceptance_criteria"], start=1)
    )
    parts.extend([
        "</ul>",
        "<h3>Estimate reasoning</h3>",
        f"<p>{html_escape(item['estimate_rationale'])}</p>",
    ])
    return "\n".join(parts)


def _comment_paragraphs(text: str) -> list[str]:
    """Split status prose into short, browser-visible paragraphs."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if not sentences:
        return []
    paragraphs: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > 190:
            paragraphs.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        paragraphs.append(current)
    return paragraphs


def comment_html(comment: dict[str, Any]) -> str:
    """Render every timeline entry with paragraphs, lists, and evidence blocks."""
    content = comment["content"]
    if comment.get("verification"):
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        summary = lines[0].removeprefix("Verification result: ")
        ac_lines = [line for line in lines if re.match(r"^AC\d+ — ", line)]
        close_line = next(line for line in lines if line.startswith("Close decision: "))
        other_lines = [line for line in lines[1:] if line not in ac_lines and line != close_line]
        parts = [
            "<p><strong>Verification result</strong></p>",
            f"<p>{html_escape(summary)}</p>",
            "<p><strong>Acceptance criteria checked</strong></p>",
            "<ul>",
        ]
        for line in ac_lines:
            ac, result, detail = line.split(" — ", 2)
            parts.append(f"<li><strong>{html_escape(ac)} — {html_escape(result)}</strong><br>{html_escape(detail)}</li>")
        parts.extend(["</ul>", f"<p><strong>Close decision</strong><br>{html_escape(close_line.removeprefix('Close decision: '))}</p>"])
        for extra in other_lines:
            placeholder = re.search(r"\[\[ATTACHMENT:[^]]+\]\]", extra)
            if placeholder:
                prefix = extra[:placeholder.start()].strip().removeprefix("Evidence:").strip()
                parts.append("<p><strong>Visual evidence</strong></p>")
                if prefix:
                    parts.append(f"<p>{html_escape(prefix)}</p>")
                parts.append(f"<p>{placeholder.group(0)}</p>")
            else:
                parts.append(f"<p>{html_escape(extra)}</p>")
        return "\n".join(parts)

    placeholder = re.search(r"\[\[ATTACHMENT:[^]]+\]\]", content)
    prose = content
    marker = None
    if placeholder:
        marker = placeholder.group(0)
        prose = (content[:placeholder.start()] + content[placeholder.end():]).strip()
    parts: list[str] = []
    if comment.get("link_context"):
        heading_by_phase = {
            "blocked-source": "Dependency created",
            "blocked-target": "Blocked status",
            "unblocked-source": "Dependency cleared",
            "unblocked-target": "Work can resume",
            "related-source": "Related work",
            "related-target": "Related context",
            "duplicate-source": "Duplicate decision",
            "duplicate-target": "Duplicate linked",
        }
        parts.append(f"<p><strong>{heading_by_phase[comment['link_phase']]}</strong></p>")
    parts.extend(f"<p>{html_escape(paragraph)}</p>" for paragraph in _comment_paragraphs(prose))
    if marker:
        parts.extend(["<p><strong>Attached evidence</strong></p>", f"<p>{marker}</p>"])
    return "\n".join(parts)


def user_expr(user_id: str | None) -> str:
    if user_id is None:
        return "NULL"
    return f"(SELECT id FROM users WHERE username = {q(USER_BY_ID[user_id]['username'])})"


def space_expr() -> str:
    return f"(SELECT id FROM spaces WHERE space_key = {q(SPACE_KEY)} AND deleted_at IS NULL)"


def sprint_expr(sprint_id: str | None) -> str:
    if sprint_id is None:
        return "NULL"
    name = next(sprint["name"] for sprint in SPRINTS if sprint["id"] == sprint_id)
    return f"(SELECT id FROM sprints WHERE space_id = {space_expr()} AND name = {q(name)})"


def issue_expr(issue_key: str | None) -> str:
    if issue_key is None:
        return "NULL"
    return f"(SELECT id FROM issues WHERE issue_key = {q(issue_key)})"


def group_expr(group_name: str) -> str:
    return f"(SELECT id FROM user_groups WHERE name = {q(group_name)} AND description LIKE {q(FIXTURE_MARKER + '%')} ORDER BY id LIMIT 1)"


def label_expr(label: str) -> str:
    return f"(SELECT id FROM labels WHERE space_id = {space_expr()} AND normalized_name = {q(label.lower())})"


def updated_at_by_issue() -> dict[str, datetime]:
    values = {item["key"]: item["created_at"] for item in ISSUES}
    for row in HISTORIES:
        values[row.issue] = max(values[row.issue], row.created_at)
    for comment in COMMENTS:
        values[comment["issue"]] = max(values[comment["issue"]], comment["created_at"])
    for link in CODE_LINKS:
        values[link.issue] = max(values[link.issue], link.last_activity_at)
    for attachment in ATTACHMENTS:
        values[attachment["issue"]] = max(values[attachment["issue"]], attachment["created_at"])
    for source, target, _, at in ISSUE_LINKS:
        values[source] = max(values[source], at)
        values[target] = max(values[target], at)
    return values


def build_sql(*, commit: bool) -> str:
    validate()
    lines: list[str] = [
        "\\set ON_ERROR_STOP on",
        "BEGIN;",
        "",
        "-- Refuse to replace a user-owned ATC project.",
        "DO $$",
        "BEGIN",
        f"  IF EXISTS (SELECT 1 FROM spaces WHERE space_key = {q(SPACE_KEY)} AND deleted_at IS NULL AND name <> {q(FIXTURE_NAME)}) THEN",
        "    RAISE EXCEPTION 'Active space key ATC already belongs to a non-fixture project';",
        "  END IF;",
        f"  IF EXISTS (SELECT 1 FROM issues i WHERE i.issue_key LIKE 'ATC-%' AND NOT EXISTS (SELECT 1 FROM spaces s WHERE s.id = i.space_id AND s.space_key = {q(SPACE_KEY)} AND s.name = {q(FIXTURE_NAME)})) THEN",
        "    RAISE EXCEPTION 'ATC issue keys already belong to a non-fixture project';",
        "  END IF;",
    ]
    for user in USERS:
        lines.extend([
            f"  IF EXISTS (SELECT 1 FROM users WHERE username = {q(user['username'])} AND email <> {q(user['email'])}) THEN",
            f"    RAISE EXCEPTION 'Fixture username {user['username']} belongs to another email';",
            "  END IF;",
        ])
    lines.extend(["END $$;", ""])

    lines.extend([
        "CREATE TEMP TABLE _atlascart_spaces ON COMMIT DROP AS",
        f"SELECT id FROM spaces WHERE space_key = {q(SPACE_KEY)} AND name = {q(FIXTURE_NAME)};",
        "CREATE TEMP TABLE _atlascart_issues ON COMMIT DROP AS",
        "SELECT id FROM issues WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "CREATE TEMP TABLE _atlascart_sprints ON COMMIT DROP AS",
        "SELECT id FROM sprints WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "",
        "DELETE FROM issue_labels WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issue_history WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM comments WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issue_attachments WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issue_code_links WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issue_links WHERE source_issue_id IN (SELECT id FROM _atlascart_issues) OR target_issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM sprint_issue_history WHERE sprint_id IN (SELECT id FROM _atlascart_sprints) OR issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issues WHERE id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM labels WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "DELETE FROM space_github_repos WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "DELETE FROM space_groups WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "DELETE FROM space_members WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "DELETE FROM sprints WHERE id IN (SELECT id FROM _atlascart_sprints);",
        "DELETE FROM spaces WHERE id IN (SELECT id FROM _atlascart_spaces);",
        f"DELETE FROM group_members WHERE group_id IN (SELECT id FROM user_groups WHERE description LIKE {q(FIXTURE_MARKER + '%')});",
        f"DELETE FROM user_groups WHERE description LIKE {q(FIXTURE_MARKER + '%')};",
        "",
    ])

    joined_at = ts(2, 8, 0)
    for user in USERS:
        lines.append(
            "INSERT INTO users (username, name, email, password, avatar_color, google_sub, github_id, created_at, updated_at) VALUES "
            f"({q(user['username'])}, {q(user['name'])}, {q(user['email'])}, {q(DEMO_PASSWORD_HASH)}, {q(user['color'])}, NULL, NULL, {q(ts(1, 12, 0))}, {q(ts(1, 12, 0))}) "
            "ON CONFLICT (username) DO UPDATE SET name = EXCLUDED.name, email = EXCLUDED.email, password = EXCLUDED.password, avatar_color = EXCLUDED.avatar_color, updated_at = EXCLUDED.updated_at;"
        )

    lines.extend([
        "",
        "INSERT INTO spaces (name, space_key, color, owner_id, issue_counter, created_at, updated_at, github_pat, deleted_at) VALUES "
        f"({q(FIXTURE_NAME)}, {q(SPACE_KEY)}, '#2563EB', {user_expr('U01')}, 48, {q(ts(2, 8, 5))}, {q(SNAPSHOT_AT)}, NULL, NULL);",
    ])

    for user in USERS:
        lines.append(
            "INSERT INTO space_members (space_id, user_id, role, joined_at) VALUES "
            f"({space_expr()}, {user_expr(user['id'])}, {q(user['role'])}, {q(joined_at)});"
        )

    for group in GROUPS:
        lines.append(
            "INSERT INTO user_groups (name, description, owner_id, created_at, updated_at) VALUES "
            f"({q(group['name'])}, {q(group['description'])}, {user_expr(group['owner'])}, {q(joined_at)}, {q(joined_at)});"
        )
        lines.append(
            "INSERT INTO space_groups (space_id, group_id, added_at) VALUES "
            f"({space_expr()}, {group_expr(group['name'])}, {q(joined_at)});"
        )
        for member in group["members"]:
            role = "owner" if member == group["owner"] else "member"
            lines.append(
                "INSERT INTO group_members (group_id, user_id, role, joined_at) VALUES "
                f"({group_expr(group['name'])}, {user_expr(member)}, {q(role)}, {q(joined_at)});"
            )

    for sprint in SPRINTS:
        if sprint["status"] == "completed":
            metrics = ", ".join(str(value) for value in completed_sprint_metrics(sprint["id"]))
        else:
            metrics = "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL"
        lines.append(
            "INSERT INTO sprints (space_id, name, goal, start_date, end_date, status, sprint_order, initial_committed_points, initial_completed_points, final_scope_points, completed_points, initial_issue_count, completed_issue_count, final_issue_count, unestimated_issue_count, created_at, updated_at) VALUES "
            f"({space_expr()}, {q(sprint['name'])}, {q(sprint['goal'])}, {q(sprint['start_date'])}, {q(sprint['end_date'])}, {q(sprint['status'])}, {sprint['order']}, {metrics}, {q(sprint['created_at'])}, {q(sprint['updated_at'])});"
        )

    for label in LABELS:
        lines.append(
            "INSERT INTO labels (space_id, name, normalized_name, created_at, deleted_at) VALUES "
            f"({space_expr()}, {q(label)}, {q(label.lower())}, {q(ts(2, 8, 10))}, NULL);"
        )

    updated = updated_at_by_issue()
    for index, item in enumerate(ISSUES):
        start_date = item["started_at"].date().isoformat() if item["started_at"] else None
        if item["sprint"]:
            sprint = next(s for s in SPRINTS if s["id"] == item["sprint"])
            due_date = sprint["end_date"]
        else:
            due_date = None
        lines.append(
            "INSERT INTO issues (issue_key, space_id, sprint_id, parent_id, title, description, issue_type, status, priority, assignee_id, reporter_id, story_points, start_date, due_date, issue_order, created_at, updated_at) VALUES "
            f"({q(item['key'])}, {space_expr()}, {sprint_expr(item['sprint'])}, {issue_expr(item['parent'])}, {q(item['title'])}, {q(description_html(item))}, {q(item['type'])}, {q(item['status'])}, {q(item['priority'])}, {user_expr(item['assignee'])}, {user_expr(item['reporter'])}, {q(item['points'])}, {q(start_date)}, {q(due_date)}, {index * 10}, {q(item['created_at'])}, {q(updated[item['key']])});"
        )
        for label in item["labels"]:
            lines.append(
                "INSERT INTO issue_labels (issue_id, label_id) VALUES "
                f"({issue_expr(item['key'])}, {label_expr(label)});"
            )

    for comment in COMMENTS:
        lines.append(
            "INSERT INTO comments (issue_id, author_id, content, created_at, updated_at) VALUES "
            f"({issue_expr(comment['issue'])}, {user_expr(comment['author'])}, {q(comment_html(comment))}, {q(comment['created_at'])}, {q(comment['created_at'])});"
        )

    for history in HISTORIES:
        lines.append(
            "INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES "
            f"({issue_expr(history.issue)}, {user_expr(history.actor)}, {q(history.event_type)}, {q(history.field_name)}, {q(history.from_value)}, {q(history.to_value)}, {q(history.description)}, {q(history.created_at)});"
        )

    for source, target, link_type, at in ISSUE_LINKS:
        lines.append(
            "INSERT INTO issue_links (source_issue_id, target_issue_id, link_type, created_at) VALUES "
            f"({issue_expr(source)}, {issue_expr(target)}, {q(link_type)}, {q(at)});"
        )

    for link in CODE_LINKS:
        lines.append(
            "INSERT INTO issue_code_links (issue_id, creator_id, url, kind, provider, owner, repo, ref_id, title, state, author_login, created_at, last_activity_at) VALUES "
            f"({issue_expr(link.issue)}, {user_expr(link.creator)}, {q(link.url)}, {q(link.kind)}, 'github', 'northstar-labs-sim', {q(link.repo)}, {q(link.ref_id)}, {q(link.title)}, {q(link.state)}, {q(link.author_login)}, {q(link.created_at)}, {q(link.last_activity_at)});"
        )

    for attachment in ATTACHMENTS:
        storage_name = attachment_storage_name(attachment)
        size = len(attachment_bytes(attachment))
        list_in_panel = attachment["placement"] == "panel"
        lines.append(
            "INSERT INTO issue_attachments (issue_id, uploader_id, original_filename, storage_filename, content_type, size_bytes, list_in_attachment_panel, created_at) VALUES "
            f"({issue_expr(attachment['issue'])}, {user_expr(attachment['uploader'])}, {q(attachment['filename'])}, {q(storage_name)}, {q(attachment['content_type'])}, {size}, {q(list_in_panel)}, {q(attachment['created_at'])});"
        )
        safe_name = html_escape(attachment["filename"], quote=True).replace("'", "''")
        if attachment["content_type"].startswith("image/"):
            embed = (
                "'<img data-attachment-id=\"' || a.id || '\" data-attachment-name=\""
                + safe_name + "\" alt=\"" + safe_name + "\" class=\"ticket-rich-image\" />'"
            )
        else:
            embed = (
                "'<span data-attachment-id=\"' || a.id || '\" data-attachment-name=\""
                + safe_name + "\">[file] " + safe_name + "</span>'"
            )
        if attachment["placement"] == "description":
            purpose = html_escape(attachment["purpose"], quote=True).replace("'", "''")
            lines.append(
                "UPDATE issues i SET description = i.description || E'\\n\\n<p><strong>Attached reference:</strong> " + purpose + " ' || "
                f"{embed} || '</p>' FROM issue_attachments a "
                f"WHERE a.issue_id = i.id AND i.issue_key = {q(attachment['issue'])} AND a.storage_filename = {q(storage_name)};"
            )
        elif attachment["placement"] == "comment":
            if attachment.get("comment_mode") == "append":
                purpose = html_escape(attachment["purpose"], quote=True).replace("'", "''")
                lines.append(
                    "UPDATE comments c SET content = c.content || E'\\n\\n<p><strong>Attached file:</strong> " + purpose + " ' || "
                    f"{embed} || '</p>' FROM issue_attachments a, issues i "
                    f"WHERE c.issue_id = i.id AND a.issue_id = i.id AND i.issue_key = {q(attachment['issue'])} "
                    f"AND a.storage_filename = {q(storage_name)} AND c.id = "
                    "(SELECT c2.id FROM comments c2 WHERE c2.issue_id = i.id ORDER BY c2.created_at DESC, c2.id DESC LIMIT 1);"
                )
            else:
                placeholder = f"[[ATTACHMENT:{attachment['filename']}]]"
                lines.append(
                    f"UPDATE comments c SET content = REPLACE(c.content, {q(placeholder)}, {embed}) "
                    "FROM issue_attachments a, issues i "
                    f"WHERE c.issue_id = i.id AND a.issue_id = i.id AND i.issue_key = {q(attachment['issue'])} "
                    f"AND a.storage_filename = {q(storage_name)} AND c.content LIKE {q('%' + placeholder + '%')};"
                )

    for owner, repo in REPOSITORIES:
        lines.append(
            "INSERT INTO space_github_repos (space_id, owner, repo, created_at, last_scanned_at) VALUES "
            f"({space_expr()}, {q(owner)}, {q(repo)}, {q(ts(2, 8, 20))}, {q(SNAPSHOT_AT - timedelta(hours=1))});"
        )

    for row in SPRINT_HISTORY:
        lines.append(
            "INSERT INTO sprint_issue_history (sprint_id, issue_id, issue_key, issue_type, initial_scope, points_at_start, points_at_end, status_at_end, outcome, added_at, removed_at) VALUES "
            f"({sprint_expr(row['sprint'])}, {issue_expr(row['issue'])}, {q(row['issue'])}, {q(row['type'])}, {q(row['initial_scope'])}, {q(row['points_at_start'])}, {q(row['points_at_end'])}, {q(row['status_at_end'])}, {q(row['outcome'])}, {q(row['added_at'])}, {q(row['removed_at'])});"
        )

    lines.extend([
        "",
        "-- Database-level fixture assertions.",
        "DO $$",
        "DECLARE",
        "  issue_count INTEGER;",
        "  user_count INTEGER;",
        "  sprint_count INTEGER;",
        "BEGIN",
        f"  SELECT COUNT(*) INTO issue_count FROM issues WHERE space_id = {space_expr()};",
        f"  SELECT COUNT(*) INTO user_count FROM space_members WHERE space_id = {space_expr()};",
        f"  SELECT COUNT(*) INTO sprint_count FROM sprints WHERE space_id = {space_expr()};",
        "  IF issue_count <> 48 OR user_count <> 12 OR sprint_count <> 3 THEN",
        "    RAISE EXCEPTION 'AtlasCart fixture count mismatch: issues %, users %, sprints %', issue_count, user_count, sprint_count;",
        "  END IF;",
        "END $$;",
        "",
        "COMMIT;" if commit else "ROLLBACK;",
        "",
    ])
    return "\n".join(lines)


def cleanup_delete_statements() -> list[str]:
    fixture_usernames = ", ".join(q(user["username"]) for user in USERS)
    return [
        "CREATE TEMP TABLE _atlascart_spaces ON COMMIT DROP AS",
        f"SELECT id FROM spaces WHERE space_key = {q(SPACE_KEY)} AND name = {q(FIXTURE_NAME)};",
        "CREATE TEMP TABLE _atlascart_issues ON COMMIT DROP AS",
        "SELECT id FROM issues WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "CREATE TEMP TABLE _atlascart_sprints ON COMMIT DROP AS",
        "SELECT id FROM sprints WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "",
        "DELETE FROM issue_labels WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issue_history WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM comments WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issue_attachments WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issue_code_links WHERE issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issue_links WHERE source_issue_id IN (SELECT id FROM _atlascart_issues) OR target_issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM sprint_issue_history WHERE sprint_id IN (SELECT id FROM _atlascart_sprints) OR issue_id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM issues WHERE id IN (SELECT id FROM _atlascart_issues);",
        "DELETE FROM labels WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "DELETE FROM space_github_repos WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "DELETE FROM space_groups WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "DELETE FROM space_members WHERE space_id IN (SELECT id FROM _atlascart_spaces);",
        "DELETE FROM sprints WHERE id IN (SELECT id FROM _atlascart_sprints);",
        "DELETE FROM spaces WHERE id IN (SELECT id FROM _atlascart_spaces);",
        f"DELETE FROM group_members WHERE group_id IN (SELECT id FROM user_groups WHERE description LIKE {q(FIXTURE_MARKER + '%')});",
        f"DELETE FROM user_groups WHERE description LIKE {q(FIXTURE_MARKER + '%')};",
        "",
        "-- Remove fixture users only when they were not reused by another dataset or space.",
        "DELETE FROM users u",
        f"WHERE u.username IN ({fixture_usernames})",
        "  AND NOT EXISTS (SELECT 1 FROM spaces s WHERE s.owner_id = u.id)",
        "  AND NOT EXISTS (SELECT 1 FROM user_groups g WHERE g.owner_id = u.id)",
        "  AND NOT EXISTS (SELECT 1 FROM space_members sm WHERE sm.user_id = u.id)",
        "  AND NOT EXISTS (SELECT 1 FROM group_members gm WHERE gm.user_id = u.id)",
        "  AND NOT EXISTS (SELECT 1 FROM issues i WHERE i.assignee_id = u.id OR i.reporter_id = u.id)",
        "  AND NOT EXISTS (SELECT 1 FROM comments c WHERE c.author_id = u.id)",
        "  AND NOT EXISTS (SELECT 1 FROM issue_history h WHERE h.actor_id = u.id)",
        "  AND NOT EXISTS (SELECT 1 FROM issue_attachments a WHERE a.uploader_id = u.id)",
        "  AND NOT EXISTS (SELECT 1 FROM issue_code_links cl WHERE cl.creator_id = u.id);",
    ]


def cleanup_safety_sql() -> list[str]:
    return [
        "DO $$",
        "BEGIN",
        f"  IF EXISTS (SELECT 1 FROM spaces WHERE space_key = {q(SPACE_KEY)} AND deleted_at IS NULL AND name <> {q(FIXTURE_NAME)}) THEN",
        "    RAISE EXCEPTION 'Refusing cleanup: active space key ATC belongs to a non-fixture project';",
        "  END IF;",
        f"  IF EXISTS (SELECT 1 FROM issues i WHERE i.issue_key LIKE 'ATC-%' AND NOT EXISTS (SELECT 1 FROM spaces s WHERE s.id = i.space_id AND s.space_key = {q(SPACE_KEY)} AND s.name = {q(FIXTURE_NAME)})) THEN",
        "    RAISE EXCEPTION 'Refusing cleanup: ATC issue keys belong to a non-fixture project';",
        "  END IF;",
        "END $$;",
    ]


def build_cleanup_sql(*, commit: bool) -> str:
    lines = ["\\set ON_ERROR_STOP on", "BEGIN;", ""]
    lines.extend(cleanup_safety_sql())
    lines.append("")
    lines.extend(cleanup_delete_statements())
    lines.extend([
        "",
        "DO $$",
        "BEGIN",
        f"  IF EXISTS (SELECT 1 FROM spaces WHERE space_key = {q(SPACE_KEY)} AND name = {q(FIXTURE_NAME)}) THEN",
        "    RAISE EXCEPTION 'AtlasCart fixture cleanup did not remove the fixture space';",
        "  END IF;",
        "END $$;",
        "",
        "COMMIT;" if commit else "ROLLBACK;",
        "",
    ])
    return "\n".join(lines)


def transaction_body(sql: str) -> str:
    kept: list[str] = []
    for line in sql.splitlines():
        if line.strip() in {"\\set ON_ERROR_STOP on", "BEGIN;", "COMMIT;", "ROLLBACK;"}:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def lifecycle_check_sql() -> str:
    return "\n".join([
        "\\set ON_ERROR_STOP on",
        "BEGIN;",
        "",
        "-- Exercise the complete injection.",
        transaction_body(build_sql(commit=True)),
        "",
        "-- Release injection temp tables before cleanup in the same transaction.",
        "DROP TABLE IF EXISTS _atlascart_issues;",
        "DROP TABLE IF EXISTS _atlascart_sprints;",
        "DROP TABLE IF EXISTS _atlascart_spaces;",
        "",
        "-- Exercise the complete cleanup.",
        transaction_body(build_cleanup_sql(commit=True)),
        "",
        "DO $$",
        "BEGIN",
        f"  IF EXISTS (SELECT 1 FROM spaces WHERE space_key = {q(SPACE_KEY)} AND name = {q(FIXTURE_NAME)}) THEN",
        "    RAISE EXCEPTION 'Lifecycle check left the AtlasCart fixture installed';",
        "  END IF;",
        "END $$;",
        "ROLLBACK;",
        "",
    ])


def status_sql() -> str:
    return "\n".join([
        "\\set ON_ERROR_STOP on",
        "SELECT CASE",
        f"  WHEN EXISTS (SELECT 1 FROM spaces WHERE space_key = {q(SPACE_KEY)} AND name = {q(FIXTURE_NAME)} AND deleted_at IS NULL) THEN 'installed'",
        f"  WHEN EXISTS (SELECT 1 FROM spaces WHERE space_key = {q(SPACE_KEY)} AND deleted_at IS NULL) THEN 'key-used-by-another-space'",
        "  ELSE 'not-installed'",
        "END AS atlascart_m1_status;",
        "SELECT",
        "  COUNT(DISTINCT i.id) AS issues,",
        "  COUNT(DISTINCT c.id) AS comments,",
        "  COUNT(DISTINCT h.id) AS history_rows",
        "FROM spaces s",
        "LEFT JOIN issues i ON i.space_id = s.id",
        "LEFT JOIN comments c ON c.issue_id = i.id",
        "LEFT JOIN issue_history h ON h.issue_id = i.id",
        f"WHERE s.space_key = {q(SPACE_KEY)} AND s.name = {q(FIXTURE_NAME)} AND s.deleted_at IS NULL;",
        "",
    ])


def attachment_storage_name(attachment: dict[str, Any]) -> str:
    safe = attachment["filename"].lower().replace(" ", "-")
    return f"atlascart-m1-{attachment['issue'].lower()}-{safe}"


def write_outputs() -> list[str]:
    checks = validate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)

    INJECT_SQL_PATH.write_text(build_sql(commit=True), encoding="utf-8")
    CLEANUP_SQL_PATH.write_text(build_cleanup_sql(commit=True), encoding="utf-8")
    expected_names = {attachment_storage_name(attachment) for attachment in ATTACHMENTS}
    for existing in ATTACHMENT_DIR.glob("atlascart-m1-*"):
        if existing.is_file() and existing.name not in expected_names:
            existing.unlink()
    for attachment in ATTACHMENTS:
        (ATTACHMENT_DIR / attachment_storage_name(attachment)).write_bytes(attachment_bytes(attachment))

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "space_key": SPACE_KEY,
        "space_name": FIXTURE_NAME,
        "snapshot_at": SNAPSHOT_AT.isoformat(),
        "as_of_date": AS_OF_DATE.isoformat(),
        "active_sprint": next(sprint["name"] for sprint in SPRINTS if sprint["status"] == "active"),
        "requested_attachment_ratio": ATTACHMENT_RATIO,
        "attachment_covered_issue_count": len({attachment["issue"] for attachment in ATTACHMENTS}),
        "actual_attachment_ratio": round(len({attachment["issue"] for attachment in ATTACHMENTS}) / len(ISSUES) * 100, 2),
        "requested_link_ratio": LINK_RATIO,
        "link_covered_issue_count": len({key for source, target, _kind, _at in ISSUE_LINKS for key in (source, target)}),
        "actual_link_ratio": round(len({key for source, target, _kind, _at in ISSUE_LINKS for key in (source, target)}) / len(ISSUES) * 100, 2),
        "attachment_runtime": {
            "binary_store": "s3-compatible MinIO",
            "bucket": "jira-attachments",
            "kafka_topic": "jira.attachment.ingestion",
            "kafka_payload": "tracking metadata only; binary bytes remain in MinIO",
        },
        "login_password": "123",
        "recommended_review_user": "atlascart.liam",
        "counts": {
            "users": len(USERS),
            "groups": len(GROUPS),
            "sprints": len(SPRINTS),
            "labels": len(LABELS),
            "issues": len(ISSUES),
            "comments": len(COMMENTS),
            "history": len(HISTORIES),
            "issue_links": len(ISSUE_LINKS),
            "code_links": len(CODE_LINKS),
            "attachments": len(ATTACHMENTS),
            "sprint_history": len(SPRINT_HISTORY),
        },
        "issue_type_counts": dict(Counter(item["type"] for item in ISSUES)),
        "attachments": [
            {
                "issue_key": attachment["issue"],
                "filename": attachment["filename"],
                "content_type": attachment["content_type"],
                "byte_size": len(attachment_bytes(attachment)),
                "placement": attachment["placement"],
                "purpose": attachment["purpose"],
                "local_path": f"attachments/{attachment_storage_name(attachment)}",
                "minio_bucket": "jira-attachments",
                "storage_key": attachment_storage_name(attachment),
            }
            for attachment in ATTACHMENTS
        ],
        "important_issue_keys": ["ATC-10", "ATC-22", "ATC-30", "ATC-35", "ATC-43", "ATC-44", "ATC-45", "ATC-47", "ATC-48"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    validation_lines = [
        "# AtlasCart M1 Validation Report",
        "",
        f"Dataset: `{DATASET_ID}` version `{DATASET_VERSION}`",
        "",
        "All generator checks passed:",
        "",
    ] + [f"- PASS — {check}" for check in checks]
    validation_lines.extend(["", "The `check-db` command also validates the complete injection and cleanup lifecycle, all foreign keys, generated search vectors, and current Flyway V26 constraints inside one rolled-back PostgreSQL transaction.", ""])
    VALIDATION_PATH.write_text("\n".join(validation_lines), encoding="utf-8")

    audit_lines = [
        "# AtlasCart M1 Point Estimate Audit",
        "",
        "Rubric: `1` tiny/local, `2` small and isolated, `3` normal single-area work, `5` cross-layer or risky, `8` large or uncertain.",
        "",
        "Epics and subtasks intentionally have no points because that matches the host application's rules. Their rows explain where the estimate is carried so every issue has an auditable sizing decision.",
        "",
        "| Issue | Type | Points | Rationale |",
        "|---|---|---:|---|",
    ]
    for item in ISSUES:
        rationale = item["estimate_rationale"].split(" — ", 1)[1]
        audit_lines.append(f"| {item['key']} | {item['type']} | {item['points'] if item['points'] is not None else '—'} | {rationale} |")
    audit_lines.append("")
    ESTIMATE_AUDIT_PATH.write_text("\n".join(audit_lines), encoding="utf-8")

    sprint_by_id = {sprint["id"]: sprint for sprint in SPRINTS}
    REVIEW_PATH.write_text(
        f"""# AtlasCart M1 Jira Review Path

Dataset as-of date: `{AS_OF_DATE.isoformat()}`

## Login

- Username: `atlascart.liam`
- Password: `123`
- Space: `ATC — AtlasCart (Synthetic M1)`

You can also log in as `atlascart.daniel` to review the project from QA's point of view.

## 1. Start with the backlog

Confirm that AtlasCart has six epics, two completed sprints, and one active sprint. The backlog should contain later work rather than looking empty or perfectly groomed.

## 2. Open Sprint 1

Sprint goal: load a seller catalog and browse product details in staging.

Follow `ATC-10` and `ATC-22`. The supplier sample contains duplicate SKUs, the team makes the rule explicit late, and both items carry into Sprint 2. Sprint 1 metrics should show 42 initially committed points and 42 completed points after added scope, not a fake 100% plan.

## 3. Open completed Sprint 2

Look for three groups:

- Original cart, checkout, order, and beta work
- `ATC-10` and `ATC-22` carried from Sprint 1
- Incident work added late in the sprint

`ATC-36`, `ATC-37`, and `ATC-38` were removed when incident work took priority. `ATC-39` moved earlier because of planned frontend availability. At close, `ATC-27`, `ATC-32`, `ATC-35`, `ATC-43`, and `ATC-47` carried into Sprint 3.

## 4. Open active Sprint 3

Today is `{AS_OF_DATE.isoformat()}`, inside Sprint 3 (`{sprint_by_id['S3']['start_date']}` through `{sprint_by_id['S3']['end_date']}`). Review the carried work beside the planned support improvements. The board should look in motion: some work is in progress, some remains planned, and the incident is still open for explicit reasons.

## 5. Follow the duplicate-order incident

Start at `ATC-43`, then follow its comments, attachments, code links, history, and linked work:

- `ATC-30` — original order-creation story, reopened after the incident
- `ATC-44` — backend idempotency fix
- `ATC-45` — frontend repeat-click mitigation
- `ATC-46` — inventory correction follow-up in Sprint 3
- `ATC-47` — slow-response browser regression still in progress

The correct conclusion at the snapshot is: the immediate frontend mitigation and backend idempotency fix are deployed, but the incident remains open until the full regression and follow-up are complete.

## 6. Check one duplicate

Open `ATC-48`. It is closed as a duplicate of `ATC-27`, with a comment explaining why. The two issues should not appear as unrelated work.

## 7. Check the evidence and writing

Open attachment evidence from the description, the attachment panel, and comments. PDF, Markdown, log, CSV, XLSX, and PNG files should be downloadable, state why they exist, and match the issue. Check that requirement input appears near the starting description while implementation and verification evidence appears in the dated comment that introduced it. Then compare comments from Liam, Noah, Daniel, Olivia, and Jack. They should use plain English and add different kinds of evidence: scope, root cause, reproduction, infrastructure facts, and customer impact.
""",
        encoding="utf-8",
    )
    return checks


def docker_run(args: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, input=input_bytes, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("Docker was not found. Start Docker Desktop and try again.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(args)}") from exc


def check_containers(require_minio: bool, require_kafka: bool = False) -> None:
    result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("Cannot access Docker. Start Docker Desktop and run the command from a terminal with Docker access.")
    names = set(result.stdout.splitlines())
    required = {"orchestra-postgres"} | ({"orchestra-minio"} if require_minio else set()) | ({"orchestra-kafka"} if require_kafka else set())
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"Required container(s) are not running: {', '.join(missing)}. Run `docker compose up -d` first.")


def run_sql(sql: str) -> None:
    docker_run(
        ["docker", "exec", "-i", "orchestra-postgres", "psql", "-q", "-U", "poc", "-d", "pocdb"],
        input_bytes=sql.encode("utf-8"),
    )


def attachment_runtime_rows() -> list[dict[str, Any]]:
    sql = (
        "SELECT a.id, i.id, i.issue_key, s.id, a.original_filename, "
        "COALESCE(a.content_type, ''), a.size_bytes, a.storage_filename "
        "FROM issue_attachments a "
        "JOIN issues i ON i.id = a.issue_id "
        "JOIN spaces s ON s.id = i.space_id "
        f"WHERE s.space_key = {q(SPACE_KEY)} AND s.name = {q(FIXTURE_NAME)} "
        "ORDER BY a.id;"
    )
    result = subprocess.run(
        ["docker", "exec", "orchestra-postgres", "psql", "-qAt", "-F", "\t", "-U", "poc", "-d", "pocdb", "-c", sql],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"Could not read attachment runtime metadata: {result.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 8:
            raise SystemExit(f"Unexpected attachment metadata row: {line}")
        rows.append({
            "attachmentId": int(fields[0]), "issueId": int(fields[1]), "issueKey": fields[2],
            "spaceId": int(fields[3]), "filename": fields[4], "mimeType": fields[5],
            "byteSize": int(fields[6]), "storageKey": fields[7],
        })
    return rows


def publish_attachment_tracking(rows: list[dict[str, Any]], event_type: str, topic: str) -> int:
    if not rows:
        return 0
    docker_run([
        "docker", "exec", "orchestra-kafka", "kafka-topics", "--bootstrap-server", "localhost:9092",
        "--create", "--if-not-exists", "--topic", topic, "--partitions", "1", "--replication-factor", "1",
    ])
    emitted_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    payload_lines: list[str] = []
    for row in rows:
        message = {
            "eventId": str(uuid.uuid4()),
            "eventType": event_type,
            "emittedAt": emitted_at,
            **row,
            "storageBackend": "s3",
            "bucket": "jira-attachments",
            "storageUri": f"s3://jira-attachments/{row['storageKey']}",
        }
        payload_lines.append(f"{row['issueKey']}|{json.dumps(message, separators=(',', ':'))}")
    docker_run(
        [
            "docker", "exec", "-i", "orchestra-kafka", "kafka-console-producer",
            "--bootstrap-server", "localhost:9092", "--topic", topic,
            "--property", "parse.key=true", "--property", "key.separator=|",
        ],
        input_bytes=("\n".join(payload_lines) + "\n").encode("utf-8"),
    )
    return len(rows)


def delete_kafka_topic(topic: str) -> None:
    docker_run([
        "docker", "exec", "orchestra-kafka", "kafka-topics", "--bootstrap-server", "localhost:9092",
        "--delete", "--if-exists", "--topic", topic,
    ])


def preflight_sql() -> str:
    lines = [
        "\\set ON_ERROR_STOP on",
        "DO $$",
        "BEGIN",
        f"  IF EXISTS (SELECT 1 FROM spaces WHERE space_key = {q(SPACE_KEY)} AND deleted_at IS NULL AND name <> {q(FIXTURE_NAME)}) THEN",
        "    RAISE EXCEPTION 'Active space key ATC already belongs to a non-fixture project';",
        "  END IF;",
        f"  IF EXISTS (SELECT 1 FROM issues i WHERE i.issue_key LIKE 'ATC-%' AND NOT EXISTS (SELECT 1 FROM spaces s WHERE s.id = i.space_id AND s.space_key = {q(SPACE_KEY)} AND s.name = {q(FIXTURE_NAME)})) THEN",
        "    RAISE EXCEPTION 'ATC issue keys already belong to a non-fixture project';",
        "  END IF;",
    ]
    for user in USERS:
        lines.extend([
            f"  IF EXISTS (SELECT 1 FROM users WHERE username = {q(user['username'])} AND email <> {q(user['email'])}) THEN",
            f"    RAISE EXCEPTION 'Fixture username {user['username']} belongs to another email';",
            "  END IF;",
        ])
    lines.extend(["END $$;", ""])
    return "\n".join(lines)


def upload_attachments() -> None:
    docker_run(["docker", "exec", "orchestra-minio", "mc", "alias", "set", "atlascart-local", "http://127.0.0.1:9000", "minioadmin", "minioadmin"])
    docker_run(["docker", "exec", "orchestra-minio", "mc", "mb", "--ignore-existing", "atlascart-local/jira-attachments"])
    for attachment in ATTACHMENTS:
        key = attachment_storage_name(attachment)
        docker_run(
            ["docker", "exec", "-i", "orchestra-minio", "mc", "pipe", f"atlascart-local/jira-attachments/{key}"],
            input_bytes=attachment_bytes(attachment),
        )
        docker_run(["docker", "exec", "orchestra-minio", "mc", "stat", f"atlascart-local/jira-attachments/{key}"])


def remove_attachments() -> None:
    alias_result = subprocess.run(
        ["docker", "exec", "orchestra-minio", "mc", "alias", "set", "atlascart-local", "http://127.0.0.1:9000", "minioadmin", "minioadmin"],
        capture_output=True,
        text=True,
    )
    if alias_result.returncode != 0:
        raise SystemExit(f"Could not configure the local MinIO client: {alias_result.stderr.strip()}")
    # The dataset-owned prefix also catches files produced by older AtlasCart
    # generator versions, preventing orphaned evidence after a refresh.
    target = "atlascart-local/jira-attachments/atlascart-m1-"
    result = subprocess.run(
        ["docker", "exec", "orchestra-minio", "mc", "rm", "--recursive", "--force", target],
        capture_output=True,
        text=True,
    )
    combined = (result.stderr + result.stdout).lower()
    missing = "does not exist" in combined or "not found" in combined
    if result.returncode != 0 and not missing:
        raise SystemExit(f"AtlasCart MinIO objects could not be removed: {(result.stderr or result.stdout).strip()}")


def print_summary(action: str) -> None:
    counts = Counter(item["type"] for item in ISSUES)
    print(f"AtlasCart M1 {action} succeeded.")
    print(f"Space: {SPACE_KEY} — {FIXTURE_NAME}")
    print(f"Issues: {len(ISSUES)} {dict(counts)}")
    print(f"Comments: {len(COMMENTS)} | History: {len(HISTORIES)} | Links: {len(ISSUE_LINKS)} | Code links: {len(CODE_LINKS)} | Attachments: {len(ATTACHMENTS)}")
    covered = len({attachment["issue"] for attachment in ATTACHMENTS})
    print(f"Attachment coverage: {covered}/{len(ISSUES)} issues ({covered / len(ISSUES) * 100:.2f}%), requested {ATTACHMENT_RATIO:g}%")
    linked = len({key for source, target, _kind, _at in ISSUE_LINKS for key in (source, target)})
    print(f"Issue-link coverage: {linked}/{len(ISSUES)} issues ({linked / len(ISSUES) * 100:.2f}%), requested {LINK_RATIO:g}%")
    print("Review login: atlascart.liam / 123")
    print(f"Review guide: {REVIEW_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and install the canonical AtlasCart M1 Jira dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_ratio_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--attachment-ratio",
            type=float,
            default=DEFAULT_ATTACHMENT_RATIO,
            metavar="PERCENT",
            help=f"Target percentage of issues with at least one attachment (0-100, default {DEFAULT_ATTACHMENT_RATIO:g})",
        )
        max_link_ratio = len({key for source, target, _kind, _at in CANONICAL_ISSUE_LINKS for key in (source, target)}) / len(ISSUES) * 100
        command_parser.add_argument(
            "--link-ratio",
            type=float,
            default=DEFAULT_LINK_RATIO,
            metavar="PERCENT",
            help=(
                f"Target percentage of issues participating in a semantic issue link "
                f"(0-{max_link_ratio:.2f}, default {DEFAULT_LINK_RATIO:g})"
            ),
        )

    generate_parser = subparsers.add_parser("generate", help="Generate SQL, attachments, manifest, and review reports")
    add_ratio_arguments(generate_parser)
    validate_parser = subparsers.add_parser("validate", help="Run deterministic in-memory validation")
    add_ratio_arguments(validate_parser)
    check_db_parser = subparsers.add_parser("check-db", help="Validate injection SQL against pocdb and roll everything back")
    add_ratio_arguments(check_db_parser)
    subparsers.add_parser("status", help="Read the AtlasCart fixture status from pocdb without changing data")
    inject_parser = subparsers.add_parser("inject", help="Safely inject or replace only the AtlasCart fixture")
    add_ratio_arguments(inject_parser)
    inject_parser.add_argument("--yes", action="store_true", help="Confirm the fixture-scoped database write")
    inject_parser.add_argument("--skip-attachments", action="store_true", help="Insert attachment rows without uploading fixture files")
    inject_parser.add_argument("--skip-kafka", action="store_true", help="Do not publish attachment_uploaded tracking messages")
    inject_parser.add_argument("--kafka-topic", default="jira.attachment.ingestion", help="Kafka attachment tracking topic")
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove the AtlasCart fixture from pocdb, MinIO, and local Kafka")
    cleanup_parser.add_argument("--yes", action="store_true", help="Confirm the fixture-scoped deletion")
    cleanup_parser.add_argument("--skip-attachments", action="store_true", help="Keep deterministic AtlasCart objects in MinIO")
    cleanup_parser.add_argument("--skip-kafka", action="store_true", help="Do not delete the Kafka topic or publish tracking messages")
    kafka_cleanup_group = cleanup_parser.add_mutually_exclusive_group()
    kafka_cleanup_group.add_argument(
        "--delete-kafka-topic",
        action="store_true",
        help="Explicitly select the default behavior: delete the entire Kafka topic",
    )
    kafka_cleanup_group.add_argument(
        "--keep-kafka-topic",
        action="store_true",
        help="Keep the Kafka topic and publish attachment_deleted messages instead",
    )
    cleanup_parser.add_argument("--kafka-topic", default="jira.attachment.ingestion", help="Kafka attachment tracking topic")
    args = parser.parse_args()

    if hasattr(args, "attachment_ratio"):
        try:
            configure_dataset(args.attachment_ratio, args.link_ratio)
        except ValueError as exc:
            parser.error(str(exc))

    if args.command == "validate":
        checks = validate()
        for check in checks:
            print(f"PASS: {check}")
        print_summary("validation")
        return

    write_outputs()

    if args.command == "generate":
        print_summary("generation")
        print(f"Injection SQL: {INJECT_SQL_PATH}")
        print(f"Cleanup SQL: {CLEANUP_SQL_PATH}")
        return

    if args.command == "check-db":
        check_containers(require_minio=False)
        run_sql(lifecycle_check_sql())
        print_summary("database check")
        print("Injection and cleanup both succeeded inside a rolled-back PostgreSQL transaction; no database data changed.")
        return

    if args.command == "status":
        check_containers(require_minio=False)
        run_sql(status_sql())
        return

    if args.command == "inject":
        if not args.yes:
            raise SystemExit("No data was changed. Re-run with `inject --yes` when you want to write the AtlasCart fixture.")
        publish_kafka = not args.skip_attachments and not args.skip_kafka
        check_containers(require_minio=not args.skip_attachments, require_kafka=publish_kafka)
        run_sql(preflight_sql())
        replaced_runtime_rows = attachment_runtime_rows() if publish_kafka else []
        if not args.skip_attachments:
            remove_attachments()
            upload_attachments()
        run_sql(build_sql(commit=True))
        if publish_kafka:
            deleted = publish_attachment_tracking(replaced_runtime_rows, "attachment_deleted", args.kafka_topic)
            uploaded = publish_attachment_tracking(attachment_runtime_rows(), "attachment_uploaded", args.kafka_topic)
            print(f"Kafka: published {deleted} replacement deletions and {uploaded} attachment uploads to {args.kafka_topic}.")
        print_summary("injection")
        return

    if args.command == "cleanup":
        if not args.yes:
            raise SystemExit("No data was changed. Re-run with `cleanup --yes` when you want to remove the AtlasCart fixture.")
        if args.delete_kafka_topic and args.skip_kafka:
            parser.error("--delete-kafka-topic and --skip-kafka cannot be used together")
        delete_kafka = not args.skip_kafka and not args.keep_kafka_topic
        publish_kafka = not args.skip_attachments and not args.skip_kafka and args.keep_kafka_topic
        use_kafka = delete_kafka or publish_kafka
        check_containers(require_minio=not args.skip_attachments, require_kafka=use_kafka)
        runtime_rows = attachment_runtime_rows() if publish_kafka else []
        run_sql(build_cleanup_sql(commit=True))
        if not args.skip_attachments:
            remove_attachments()
        if delete_kafka:
            delete_kafka_topic(args.kafka_topic)
            print(f"Kafka: deleted the entire topic {args.kafka_topic}.")
        elif publish_kafka:
            published = publish_attachment_tracking(runtime_rows, "attachment_deleted", args.kafka_topic)
            print(f"Kafka: published {published} attachment_deleted tracking messages to {args.kafka_topic}.")
        print("AtlasCart M1 cleanup succeeded. Other database spaces and referenced shared users were preserved.")
        return


if __name__ == "__main__":
    main()

"""Live smoke test of the MCP server through the public entry point, using a real MCP client.

    .venv/bin/python -m scripts.mcp_smoke                  # all tools, AtlasCart demo user
    .venv/bin/python -m scripts.mcp_smoke --issue ATC-77 --skip-llm

Logs in (POST /api/auth/token), then over http://localhost:5173/api/ai/mcp: lists tools, calls each
one, and checks the two refusals that matter (no token; a space the user isn't in).
"""
import argparse
import asyncio
import json
import os
import sys
import time

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# The browser-facing origin external agents use: the Vite dev proxy locally, CloudFront when deployed.
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5173")


def _login(username: str, password: str) -> str:
    resp = httpx.post(f"{BASE_URL}/api/auth/token", json={"username": username, "password": password}, timeout=10)
    resp.raise_for_status()
    return resp.json()["accessToken"] if "accessToken" in resp.json() else resp.json()["token"]


def _show(label: str, result, started: float) -> None:
    status = "ERROR" if result.isError else "ok"
    text = result.content[0].text if result.content else ""
    print(f"\n=== {label}  [{status}, {time.perf_counter() - started:.1f}s]")
    try:
        print(json.dumps(json.loads(text), indent=2, ensure_ascii=False)[:2500])
    except ValueError:
        print(text[:2500])


async def main(args) -> int:
    token = _login(args.username, args.password)
    url = f"{BASE_URL}/api/ai/mcp"

    async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"}, timeout=300) as (r, w, _):
        async with ClientSession(r, w) as session:
            init = await session.initialize()
            print(f"connected: {init.serverInfo.name} (protocol {init.protocolVersion})")
            tools = await session.list_tools()
            print("tools:", ", ".join(t.name for t in tools.tools))
            prompts = await session.list_prompts()
            print("prompts:", ", ".join(p.name for p in prompts.prompts))

            calls = [
                ("get_ticket_context", {"issue_key": args.issue}),
                ("check_sprint_risk", {"space": args.space}),
                ("ask_project", {"question": "SECRET space question", "space": "NOPE"}),  # must be refused
            ]
            if not args.skip_llm:
                calls += [
                    ("find_similar_work", {"issue_key": args.issue}),
                    ("ask_project", {"question": args.question, "space": args.space}),
                ]
            failures = 0
            for name, call_args in calls:
                started = time.perf_counter()
                result = await session.call_tool(name, call_args)
                _show(f"{name}({call_args})", result, started)
                expected_error = call_args.get("space") == "NOPE"
                if result.isError != expected_error:
                    failures += 1

    # Without a token ai-service must reject before any tool runs.
    resp = httpx.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                      headers={"Accept": "application/json, text/event-stream"}, timeout=10)
    print(f"\n=== no token -> HTTP {resp.status_code}")
    if resp.status_code != 401:
        failures += 1

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=os.getenv("E2E_USERNAME", "atlascart.maya"))
    parser.add_argument("--password", default=os.getenv("E2E_PASSWORD", "123"))
    parser.add_argument("--space", default="ATC")
    parser.add_argument("--issue", default="ATC-77")
    parser.add_argument("--question", default="Which issues are blocked right now, and why?")
    parser.add_argument("--skip-llm", action="store_true", help="only the tools that make no LLM call")
    sys.exit(asyncio.run(main(parser.parse_args())))

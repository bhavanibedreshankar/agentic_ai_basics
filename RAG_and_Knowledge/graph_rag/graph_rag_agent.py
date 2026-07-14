"""
CONCEPT: Agentic graph RAG — instead of a FIXED pipeline that always runs
seed-match-then-traverse-then-generate on every question (as
./graph_retrieval.py does), give Claude direct TOOLS onto the code graph
and let it decide, per question, whether to query at all and which
strategy to use. This is the same fixed-pipeline-vs-tool-calling contrast
already drawn in this repo between ../rag/basic_rag.py and
../../Task_and_State_Management/context_management/retrieval.py's
search_notes tool — applied here to a graph instead of flat documents.

Two tools are offered, deliberately different in kind, because a graph
supports a question a flat-text retrieval tool structurally cannot answer:
  - search_code_graph(query)   — fuzzy: keyword-seed the graph, then
    traverse outward a couple of hops. Good for open-ended questions
    ("how does order confirmation work?").
  - trace_call_chain(function_name, direction) — precise: follow ONLY
    "calls" edges from a named function, either outward (what it calls)
    or inward (what calls it), with no keyword matching at all. Good for
    structural questions ("what calls charge_card?") that have an exact
    answer a similarity search could easily rank wrong. Nothing analogous
    exists in ../../Task_and_State_Management/context_management/retrieval.py's
    search_notes — flat text has no notion of "callers" to trace.

Claude picks between them itself, based on each tool's description below
— that decision-making is exactly what makes this "agentic" rather than a
fixed pipeline.

This file rebuilds the same graph as ./graph_construction.py and reuses
./graph_retrieval.py's traversal mechanics (both condensed here — see
those files for the fully annotated versions).

Type 'exit' to end the conversation.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import anthropic

# --- API settings (see ../../Core_Architecture/basics/basic.py for what each of these means) ---
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096
EFFORT = "medium"
SYSTEM_PROMPT = (
    "You are a code assistant with access to a queryable graph database of "
    "a codebase. Use search_code_graph for open-ended questions about what "
    "code does or how pieces relate; use trace_call_chain for precise "
    "structural questions like 'what does X call' or 'what calls X'. "
    "Always use a tool before answering a question about the code — don't "
    "guess from general knowledge. Cite specific function/class names from "
    "the tool results in your answer."
)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

GRAPH_FILE = Path(__file__).parent / "code_graph.json"

# ---------------------------------------------------------------------------
# Same toy codebase as ./graph_construction.py and ./graph_retrieval.py.
# ---------------------------------------------------------------------------
SOURCE_FILES: dict[str, str] = {
    "models.py": '''
class Customer:
    """A customer placing orders."""
    def __init__(self, name, email):
        self.name = name
        self.email = email


class Order:
    """A customer's order for a list of priced items."""
    def __init__(self, customer, items):
        self.customer = customer
        self.items = items

    def total(self):
        """Sum the price of every item in the order."""
        return sum(item["price"] for item in self.items)


class PriorityOrder(Order):
    """An order with expedited shipping, billed at a 10% surcharge."""
    def total(self):
        """Apply the priority surcharge on top of the base total."""
        return super().total() * 1.1
''',
    "payment.py": '''
def charge_card(customer, amount):
    """Charge the customer's card for the given amount."""
    print(f"Charging {customer.name} ${amount:.2f}")


def refund(customer, amount):
    """Refund the given amount back to the customer's card."""
    print(f"Refunding {customer.name} ${amount:.2f}")
''',
    "notifications.py": '''
def send_email(customer, subject, body):
    """Send a transactional email to the customer."""
    print(f"Emailing {customer.email}: {subject}")


def notify_order_confirmed(order):
    """Send the order confirmation email."""
    send_email(order.customer, "Order confirmed", f"Your order total is {order.total()}")


def notify_order_cancelled(order):
    """Send the order cancellation email."""
    send_email(order.customer, "Order cancelled", "Your order has been cancelled and refunded.")
''',
    "orders.py": '''
from models import Order, PriorityOrder, Customer
from payment import charge_card, refund
from notifications import notify_order_confirmed, notify_order_cancelled


def create_order(customer, items, priority=False):
    """Create a new order, charge the customer, and send a confirmation email."""
    order = PriorityOrder(customer, items) if priority else Order(customer, items)
    charge_card(customer, order.total())
    notify_order_confirmed(order)
    return order


def cancel_order(order):
    """Refund the customer and send a cancellation email."""
    refund(order.customer, order.total())
    notify_order_cancelled(order)
''',
}


def _reference_name(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


class _FileVisitor(ast.NodeVisitor):
    """Condensed — identical to ./graph_construction.py's visitor."""

    def __init__(self, filename: str, nodes: dict[str, dict], edges: list[dict], pending: list[tuple[str, str, str]]):
        self.filename = filename
        self.nodes = nodes
        self.edges = edges
        self.pending = pending
        nodes[filename] = {"id": filename, "type": "module", "name": filename, "qualname": filename, "lineno": 1, "doc": None}
        self.scope_ids = [filename]
        self.scope_class = [None]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.pending.append((self.scope_ids[0], f"{alias.name}.py", "imports"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.pending.append((self.scope_ids[0], f"{node.module}.py", "imports"))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = f"{self.filename}::{node.name}"
        self.nodes[class_id] = {
            "id": class_id, "type": "class", "name": node.name, "qualname": node.name,
            "lineno": node.lineno, "doc": ast.get_docstring(node),
        }
        self.edges.append({"src": self.scope_ids[-1], "dst": class_id, "type": "contains"})
        for base in node.bases:
            base_name = _reference_name(base)
            if base_name:
                self.pending.append((class_id, base_name, "inherits"))
        self.scope_ids.append(class_id)
        self.scope_class.append(node.name)
        self.generic_visit(node)
        self.scope_ids.pop()
        self.scope_class.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        enclosing_class = self.scope_class[-1]
        qualname = f"{enclosing_class}.{node.name}" if enclosing_class else node.name
        func_id = f"{self.filename}::{qualname}"
        self.nodes[func_id] = {
            "id": func_id, "type": "function", "name": node.name, "qualname": qualname,
            "lineno": node.lineno, "doc": ast.get_docstring(node),
        }
        self.edges.append({"src": self.scope_ids[-1], "dst": func_id, "type": "contains"})
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee_name = _reference_name(child.func)
                if callee_name:
                    self.pending.append((func_id, callee_name, "calls"))
        self.scope_ids.append(func_id)
        self.scope_class.append(enclosing_class)
        self.generic_visit(node)
        self.scope_ids.pop()
        self.scope_class.pop()


def _resolve_pending(nodes: dict[str, dict], edges: list[dict], pending: list[tuple[str, str, str]]) -> None:
    for src, target_name, kind in pending:
        if kind == "imports":
            if target_name in nodes and nodes[target_name]["type"] == "module":
                edges.append({"src": src, "dst": target_name, "type": "imports"})
        elif kind == "inherits":
            for node in nodes.values():
                if node["type"] == "class" and node["name"] == target_name:
                    edges.append({"src": src, "dst": node["id"], "type": "inherits"})
        elif kind == "calls":
            for node in nodes.values():
                if node["type"] == "function" and node["name"] == target_name and node["id"] != src:
                    edges.append({"src": src, "dst": node["id"], "type": "calls"})


def build_code_graph(source_files: dict[str, str]) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    pending: list[tuple[str, str, str]] = []
    for filename, source in source_files.items():
        tree = ast.parse(source, filename=filename)
        _FileVisitor(filename, nodes, edges, pending).visit(tree)
    _resolve_pending(nodes, edges, pending)
    return {"nodes": nodes, "edges": edges}


def load_or_build_graph() -> dict:
    if GRAPH_FILE.exists():
        return json.loads(GRAPH_FILE.read_text())
    return build_code_graph(SOURCE_FILES)


def _score_node(node: dict, query_words: set[str]) -> int:
    haystack = f"{node['name']} {node['qualname']} {node.get('doc') or ''}"
    normalized = haystack.lower().replace(".", " ").replace("_", " ")
    return len(query_words & set(normalized.split()))


def find_seed_nodes(graph: dict, query: str, top_n: int = 2) -> list[str]:
    query_words = set(query.lower().split())
    scored = [(_score_node(node, query_words), node_id) for node_id, node in graph["nodes"].items()]
    scored = [pair for pair in scored if pair[0] > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [node_id for _, node_id in scored[:top_n]]


def retrieve_subgraph(graph: dict, query: str, seed_top_n: int = 2, max_hops: int = 2, max_nodes: int = 8) -> dict:
    """Same fuzzy seed-then-traverse mechanic as ./graph_retrieval.py's
    retrieve_subgraph — see that file for the annotated version."""
    seeds = find_seed_nodes(graph, query, top_n=seed_top_n)
    if not seeds:
        return {"nodes": [], "edges": [], "seeds": []}
    visited = set(seeds)
    frontier = list(seeds)
    for _ in range(max_hops):
        if len(visited) >= max_nodes:
            break
        next_frontier: list[str] = []
        for node_id in frontier:
            for edge in graph["edges"]:
                neighbor = None
                if edge["src"] == node_id and edge["dst"] not in visited:
                    neighbor = edge["dst"]
                elif edge["dst"] == node_id and edge["src"] not in visited:
                    neighbor = edge["src"]
                if neighbor is not None:
                    if len(visited) >= max_nodes:
                        break
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    included_edges = [e for e in graph["edges"] if e["src"] in visited and e["dst"] in visited]
    return {"nodes": [graph["nodes"][nid] for nid in visited], "edges": included_edges, "seeds": seeds}


def render_subgraph(subgraph: dict) -> str:
    lines = ["Nodes:"]
    for node in subgraph["nodes"]:
        doc = f" — {node['doc']}" if node.get("doc") else ""
        lines.append(f"  [{node['type']}] {node['id']}{doc}")
    lines.append("Relationships:")
    for edge in subgraph["edges"]:
        lines.append(f"  {edge['src']} --{edge['type']}--> {edge['dst']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CONCEPT: precise structural traversal, as a SEPARATE tool from the fuzzy
# one above. Follows only "calls" edges, in one explicit direction, from
# every function whose bare name matches — including every match when the
# name is ambiguous (e.g. "total" resolves to both Order.total and
# PriorityOrder.total, the same name-based ambiguity ./graph_construction.py
# warns about when building "calls" edges in the first place). Reporting
# every match rather than guessing one is the honest choice: silently
# picking the first match would hide that ambiguity from Claude instead of
# surfacing it.
# ---------------------------------------------------------------------------
def trace_call_chain(graph: dict, function_name: str, direction: str = "callees", max_hops: int = 3) -> str:
    starts = [nid for nid, n in graph["nodes"].items() if n["type"] == "function" and n["name"] == function_name]
    if not starts:
        return f"No function named '{function_name}' found in the code graph."

    lines = []
    for start in starts:
        visited = {start}
        frontier = [start]
        chain_edges = []
        for _ in range(max_hops):
            next_frontier = []
            for node_id in frontier:
                for edge in graph["edges"]:
                    if edge["type"] != "calls":
                        continue
                    if direction == "callees" and edge["src"] == node_id and edge["dst"] not in visited:
                        chain_edges.append(edge)
                        visited.add(edge["dst"])
                        next_frontier.append(edge["dst"])
                    elif direction == "callers" and edge["dst"] == node_id and edge["src"] not in visited:
                        chain_edges.append(edge)
                        visited.add(edge["src"])
                        next_frontier.append(edge["src"])
            frontier = next_frontier
            if not frontier:
                break

        lines.append(f"From {start} ({direction}, up to {max_hops} hops):")
        if not chain_edges:
            lines.append(f"  (no {direction} found)")
        for edge in chain_edges:
            lines.append(f"  {edge['src']} --calls--> {edge['dst']}")

    if len(starts) > 1:
        lines.insert(0, f"Note: '{function_name}' matched {len(starts)} functions — showing the chain for each.")
    return "\n".join(lines)


TOOLS = [
    {
        "name": "search_code_graph",
        "description": (
            "Search the code graph for nodes relevant to an open-ended natural-language question "
            "and return the matched subgraph (functions/classes/modules plus how they relate via "
            "calls, contains, inherits, and imports edges). Use this for questions about what code "
            "does or how pieces connect, e.g. 'how is an order confirmed?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Natural-language question about the codebase"}},
            "required": ["query"],
        },
    },
    {
        "name": "trace_call_chain",
        "description": (
            "Trace the exact call chain from a specific function by name: either everything IT "
            "calls (direction='callees') or everything that calls IT (direction='callers'), up to "
            "max_hops deep. Use this for precise structural questions like 'what does charge_card "
            "call?' or 'what calls send_email?' — more reliable than search_code_graph when you "
            "already know the exact function name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {"type": "string", "description": "Bare function or method name, e.g. 'charge_card' or 'total'"},
                "direction": {"type": "string", "enum": ["callees", "callers"], "description": "Defaults to 'callees'"},
                "max_hops": {"type": "integer", "description": "How many call-edges deep to follow. Defaults to 3."},
            },
            "required": ["function_name"],
        },
    },
]

_GRAPH: dict = {}


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    try:
        if name == "search_code_graph":
            subgraph = retrieve_subgraph(_GRAPH, tool_input["query"])
            if not subgraph["nodes"]:
                return "No relevant nodes found in the code graph.", False
            return render_subgraph(subgraph), False
        if name == "trace_call_chain":
            return trace_call_chain(
                _GRAPH,
                tool_input["function_name"],
                direction=tool_input.get("direction", "callees"),
                max_hops=tool_input.get("max_hops", 3),
            ), False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to Claude
        return f"Error: {exc}", True


def run_turn(messages: list[dict]) -> None:
    """Same inner tool-calling loop as ../../Core_Architecture/tool_use/basic_agentic_tools.py."""
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": EFFORT},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\nClaude: {block.text}\n")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({block.input})")
                result_text, is_error = execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    global _GRAPH
    _GRAPH = load_or_build_graph()

    print(f"Code graph agent — {len(_GRAPH['nodes'])} nodes, {len(_GRAPH['edges'])} edges. Type 'exit' to end the conversation.\n")
    print('Try: "what calls charge_card?" then "what does create_order do end to end?" and compare which tool Claude reaches for.\n')

    messages: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()

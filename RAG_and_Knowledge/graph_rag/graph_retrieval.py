"""
CONCEPT: Graph retrieval — instead of independently ranking flat chunks by
similarity (../rag/basic_rag.py, ../coarse_to_fine_retrieval/coarse_to_fine_retrieval.py),
start from a few SEED nodes that keyword-match the query, then TRAVERSE the
code graph outward along its real edges (calls, contains, inherits,
imports) for a bounded number of hops. What comes back is a connected
SUBGRAPH, not an independently-scored top-k list — and that's the whole
point: a node can end up in the retrieved context because it's structurally
CONNECTED to something relevant, even if it shares no words with the query
at all.

Concretely: the query "how is an order confirmed" keyword-matches
notify_order_confirmed (it shares "confirmed"). Flat retrieval would stop
there. Graph retrieval instead follows notify_order_confirmed's "calls"
edge one more hop and pulls in send_email too — a function whose name and
docstring have zero words in common with the query, but which is exactly
what "confirmed" turns out to mean once you follow the actual code path.
Run this file and try that query to see it happen.

This file rebuilds the same graph as ./graph_construction.py (identical
construction mechanic, condensed here with lighter comments — see that
file for the fully annotated version) and, if graph_construction.py has
already been run in this directory, loads the code_graph.json it persisted
instead of rebuilding from scratch — the realistic "build once, query
many times" shape of a real graph database.

Run this file directly:

    python3 graph_retrieval.py
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

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

RAG_SYSTEM_PROMPT = (
    "You are a code assistant. Answer using ONLY the code graph context "
    "provided below (functions, classes, and the calls/imports/inheritance "
    "between them). If the context doesn't cover the question, say so."
)

GRAPH_FILE = Path(__file__).parent / "code_graph.json"

# ---------------------------------------------------------------------------
# Same toy codebase as ./graph_construction.py — see that file for the full
# CONCEPT commentary on each relationship this creates.
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
    """Condensed version of ./graph_construction.py's visitor — same
    behavior, see that file for the annotated walkthrough."""

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
    """Prefer the persisted database from ./graph_construction.py if it's
    there (the realistic path — build once, query many times); fall back to
    building it in memory so this file still runs completely on its own.
    """
    if GRAPH_FILE.exists():
        print(f"[loaded persisted graph from {GRAPH_FILE.name} — run graph_construction.py again to refresh it]")
        return json.loads(GRAPH_FILE.read_text())
    print("[no persisted code_graph.json found — building the graph in memory instead]")
    return build_code_graph(SOURCE_FILES)


# ---------------------------------------------------------------------------
# CONCEPT: seed matching — same keyword-overlap idea as
# ../../Task_and_State_Management/context_management/retrieval.py's
# search_notes, just scored against a graph NODE's name/qualname/docstring
# instead of a flat document's full text.
# ---------------------------------------------------------------------------
def _score_node(node: dict, query_words: set[str]) -> int:
    haystack = f"{node['name']} {node['qualname']} {node.get('doc') or ''}"
    # Code identifiers use underscores/dots where prose uses spaces
    # ("notify_order_confirmed", "Order.total") — normalize both to spaces
    # before splitting, so a query word like "confirmed" matches the name
    # it's embedded in instead of only ever matching whole identifiers.
    normalized = haystack.lower().replace(".", " ").replace("_", " ")
    haystack_words = set(normalized.split())
    return len(query_words & haystack_words)


def find_seed_nodes(graph: dict, query: str, top_n: int = 2) -> list[str]:
    query_words = set(query.lower().split())
    scored = [(_score_node(node, query_words), node_id) for node_id, node in graph["nodes"].items()]
    scored = [pair for pair in scored if pair[0] > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [node_id for _, node_id in scored[:top_n]]


# ---------------------------------------------------------------------------
# CONCEPT: bounded graph traversal — a breadth-first walk outward from the
# seeds, `max_hops` deep, stopping early if `max_nodes` is reached. Both
# limits exist for the same reason a vector search caps `top_k`: without a
# budget, "follow every edge" on a large real codebase would pull in most of
# the graph and defeat the purpose of retrieval (keeping context small and
# relevant). Edge direction is ignored here (a caller matters as much as a
# callee when explaining "what touches this function") — pass `edge_types`
# to restrict which relationships are worth following for a given question.
# ---------------------------------------------------------------------------
def retrieve_subgraph(
    graph: dict,
    query: str,
    seed_top_n: int = 2,
    max_hops: int = 2,
    max_nodes: int = 8,
    edge_types: set[str] | None = None,
) -> dict:
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
                if edge_types is not None and edge["type"] not in edge_types:
                    continue
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
    return {
        "nodes": [graph["nodes"][node_id] for node_id in visited],
        "edges": included_edges,
        "seeds": seeds,
    }


def render_subgraph(subgraph: dict) -> str:
    """Turn the retrieved subgraph into plain text Claude can read as
    context — the graph equivalent of joining retrieved chunks together in
    ../rag/basic_rag.py, just with edges spelled out alongside the nodes so
    the relationships aren't lost in translation.
    """
    lines = ["Nodes:"]
    for node in subgraph["nodes"]:
        doc = f" — {node['doc']}" if node.get("doc") else ""
        lines.append(f"  [{node['type']}] {node['id']}{doc}")
    lines.append("Relationships:")
    for edge in subgraph["edges"]:
        lines.append(f"  {edge['src']} --{edge['type']}--> {edge['dst']}")
    return "\n".join(lines)


def answer_query(query: str) -> str:
    graph = _GRAPH
    subgraph = retrieve_subgraph(graph, query)

    print(f"  [seeds: {subgraph['seeds']}]")
    print(f"  [retrieved {len(subgraph['nodes'])} of {len(graph['nodes'])} nodes, {len(subgraph['edges'])} edges]")

    if not subgraph["nodes"]:
        return "I couldn't find anything in the code graph relevant to that question."

    context = render_subgraph(subgraph)
    prompt = f"Code graph context:\n{context}\n\nQuestion: {query}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=RAG_SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


_GRAPH: dict = {}


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this script.", file=sys.stderr)
        sys.exit(1)

    global _GRAPH
    _GRAPH = load_or_build_graph()

    print(f"Code graph assistant — {len(_GRAPH['nodes'])} nodes, {len(_GRAPH['edges'])} edges.")
    print("Graph retrieval demo. Type 'exit' to end the conversation.\n")
    print('Try: "how is an order confirmed" — notice send_email gets pulled in via a 2-hop "calls" edge even though it never appears in the query.\n')

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            continue

        answer = answer_query(query)
        print(f"\nClaude: {answer}\n")


if __name__ == "__main__":
    main()

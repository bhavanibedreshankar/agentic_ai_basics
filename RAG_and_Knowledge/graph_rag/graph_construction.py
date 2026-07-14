"""
CONCEPT: Graph construction — building a queryable graph "database" out of
source code, instead of treating it as flat text to chunk (see
../chunking/chunking_strategies.py). A code graph has NODES for the code's
actual structural units (modules, classes, functions/methods) and EDGES for
the relationships between them (a module CONTAINS a class, a class INHERITS
from another, a function CALLS another). That structure is exactly what flat
text chunking throws away — chunking only knows "these characters are near
each other," never "this function calls that one," which is the kind of
question you actually ask about a codebase.

This is the "how is the database created" half of Graph RAG. The other half
— "how is it retrieved" — is ./graph_retrieval.py, which rebuilds the graph
this file produces (same construction logic, condensed inline — see the
note at the bottom of main() for why) and traverses it instead of ranking
flat chunks by similarity.

No embeddings, no external graph database, no API calls — this template is
pure static analysis using Python's own `ast` (Abstract Syntax Tree) module
against a small in-memory toy codebase (an order-processing mini-app, defined
below as strings rather than real files on disk, so this runs standalone with
no filesystem setup). `ast.parse()` turns source text into a tree Python
itself understands; walking that tree is how real tools (linters, IDEs,
code-intelligence indexers like Sourcegraph or CodeQL) build the same kind of
graph this template builds, just with far more precision than the name-based
heuristic used here (see the CONCEPT comment on call-edge resolution below
for exactly what's simplified and why).

Run this file directly to build the graph, print a summary, and persist it
to code_graph.json next to this script:

    python3 graph_construction.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# A tiny toy codebase: four modules of an order-processing mini-app, defined
# as strings (not real files) so this template needs no filesystem setup to
# run. Deliberately includes every relationship type we want to graph:
# containment (classes/functions inside modules), inheritance (PriorityOrder
# extends Order), imports (orders.py imports the other three), and calls
# (create_order calls charge_card, which chains into notify_order_confirmed
# calling send_email — a 2-hop call chain graph_retrieval.py will traverse).
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

GRAPH_FILE = Path(__file__).parent / "code_graph.json"


def _reference_name(expr: ast.AST) -> str | None:
    """Pull a bare name out of a `Call.func` or a class base expression.

    `ast.Name` covers a bare reference like `charge_card(...)` or
    `class PriorityOrder(Order)`. `ast.Attribute` covers a dotted one like
    `order.total()` — we take just the trailing `.attr` ("total"), which is
    the name-based heuristic explained in visit_FunctionDef below.
    """
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


class _FileVisitor(ast.NodeVisitor):
    """Walks one file's AST, adding its nodes and CONTAINS edges directly
    into the shared graph (both endpoints are already known within a single
    file). Anything that references ANOTHER file or a definition this
    visitor hasn't reached yet — imports, base classes, calls — gets queued
    in `pending` as a (source_id, target_name, edge_kind) tuple and resolved
    by name only after every file has been visited (see _resolve_pending).
    """

    def __init__(self, filename: str, nodes: dict[str, dict], edges: list[dict], pending: list[tuple[str, str, str]]):
        self.filename = filename
        self.nodes = nodes
        self.edges = edges
        self.pending = pending
        # The module itself is a node too — everything else nests under it.
        nodes[filename] = {"id": filename, "type": "module", "name": filename, "qualname": filename, "lineno": 1, "doc": None}
        self.scope_ids = [filename]     # stack of enclosing node ids
        self.scope_class = [None]       # stack of enclosing class name (or None), for qualnames

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

        # CONCEPT: call-edge extraction is NAME-based, not type-based. For
        # every ast.Call in this function's body, we record the CALLEE'S
        # NAME (e.g. "charge_card", or "total" for `order.total()`) as a
        # pending edge — we don't know yet whether that name will resolve
        # to exactly one function once the whole graph is built. That's a
        # real trade-off: a production code-intelligence tool resolves
        # `order.total()` via order's actual inferred type, so it never
        # confuses Order.total with an unrelated total() elsewhere. Here,
        # any function named "total" anywhere in the toy codebase would
        # match. That's an acceptable trade for a dependency-free demo on
        # code this size, where name collisions are rare — but it's exactly
        # the kind of precision a real graph RAG system over a large,
        # real-world repo has to solve for.
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee_name = _reference_name(child.func)
                if callee_name:
                    self.pending.append((func_id, callee_name, "calls"))

        self.scope_ids.append(func_id)
        self.scope_class.append(enclosing_class)  # a nested def would stay in the same class scope
        self.generic_visit(node)
        self.scope_ids.pop()
        self.scope_class.pop()


def _resolve_pending(nodes: dict[str, dict], edges: list[dict], pending: list[tuple[str, str, str]]) -> None:
    """Turn each (source_id, target_name, kind) into a real edge, now that
    every file has been visited and every node id is known. Resolution is
    plain name matching against the already-built node set — see the
    CONCEPT comment in visit_FunctionDef for what that trades away.
    """
    for src, target_name, kind in pending:
        if kind == "imports":
            if target_name in nodes and nodes[target_name]["type"] == "module":
                edges.append({"src": src, "dst": target_name, "type": "imports"})
            # else: import of something outside this toy codebase (e.g. a
            # stdlib module) — nothing to link to, so it's silently skipped.
        elif kind == "inherits":
            for node in nodes.values():
                if node["type"] == "class" and node["name"] == target_name:
                    edges.append({"src": src, "dst": node["id"], "type": "inherits"})
        elif kind == "calls":
            for node in nodes.values():
                if node["type"] == "function" and node["name"] == target_name and node["id"] != src:
                    edges.append({"src": src, "dst": node["id"], "type": "calls"})


def build_code_graph(source_files: dict[str, str]) -> dict:
    """Parse every file and return {"nodes": {id: node}, "edges": [edge]}
    — the whole graph, JSON-serializable, ready to persist or query.
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    pending: list[tuple[str, str, str]] = []

    for filename, source in source_files.items():
        tree = ast.parse(source, filename=filename)
        _FileVisitor(filename, nodes, edges, pending).visit(tree)

    _resolve_pending(nodes, edges, pending)
    return {"nodes": nodes, "edges": edges}


def save_graph(graph: dict, path: Path) -> None:
    path.write_text(json.dumps(graph, indent=2))


def describe_graph(graph: dict) -> None:
    nodes, edges = graph["nodes"], graph["edges"]

    by_type: dict[str, int] = {}
    for node in nodes.values():
        by_type[node["type"]] = by_type.get(node["type"], 0) + 1
    print(f"{len(nodes)} nodes: " + ", ".join(f"{count} {kind}" for kind, count in by_type.items()))

    edges_by_type: dict[str, int] = {}
    for edge in edges:
        edges_by_type[edge["type"]] = edges_by_type.get(edge["type"], 0) + 1
    print(f"{len(edges)} edges: " + ", ".join(f"{count} {kind}" for kind, count in edges_by_type.items()))

    print("\nEdges (src --type--> dst):")
    for edge in edges:
        print(f"  {edge['src']} --{edge['type']}--> {edge['dst']}")


def main() -> None:
    graph = build_code_graph(SOURCE_FILES)
    print(f"Built a code graph from {len(SOURCE_FILES)} source files.\n")
    describe_graph(graph)

    save_graph(graph, GRAPH_FILE)
    print(f"\nSaved to {GRAPH_FILE.name} — this is the persisted 'database' half of graph RAG.")
    print(
        "graph_retrieval.py and graph_rag_agent.py in this same directory rebuild an equivalent "
        "graph using this same construction logic (condensed inline, so each template stays "
        "runnable on its own — this repo's convention, see any two related templates elsewhere) "
        "and QUERY it, rather than requiring this script to be run first."
    )


if __name__ == "__main__":
    main()

"""
Parse a raw FUNSD annotation JSON into a DocumentRecord: entities, deduplicated undirected links,
and the scorable question->answer pairs the extraction task is actually graded on.

Confirmed empirically against the real dataset (Phase 0 spike, 2026-08-31), not assumed:
  - ``linking`` is stored symmetrically — a pair [a, b] appears on BOTH entity a's and entity b's
    own ``linking`` list. Resolving "what is X linked to" only requires scanning X's own list.
  - Gold linking is NOT a 1:1 question<->answer matching. Some questions link to 2+ answer entities
    (frequently one logical answer split across line-wrapped boxes; occasionally a genuine
    multi-option/checklist question). Those are concatenated in reading order into one scored
    answer per question — a deliberate, disclosed simplification (see ARCHITECTURE.md), not an
    attempt to hand-classify every ambiguous case under a tight deadline.
  - Headers can link to multiple question entities (a section header chaining to several
    sub-questions) — these are kept as structural context for the graph representations but are
    never scorable targets themselves (only question->answer edges are scored).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    id: int
    text: str
    label: str  # "question" | "answer" | "header" | "other"
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class ScorablePair:
    question_id: int
    question_text: str
    answer_entity_ids: tuple[int, ...]
    gold_answer_text: str


@dataclass
class DocumentRecord:
    doc_id: str
    entities: list[Entity]
    links: list[tuple[int, int]]  # deduplicated, undirected (stored as sorted (min_id, max_id))
    scorable_pairs: list[ScorablePair]

    def entity_by_id(self, entity_id: int) -> Entity:
        for e in self.entities:
            if e.id == entity_id:
                return e
        raise KeyError(f"No entity with id {entity_id} in document {self.doc_id}")


def reading_order_key(entity: Entity) -> tuple[int, int]:
    """Top-to-bottom, then left-to-right, approximating reading order from the box alone."""
    x1, y1, _, _ = entity.box
    return (y1, x1)


def parse_document(annotation: dict, doc_id: str) -> DocumentRecord:
    """Parse one raw FUNSD annotation dict (the ``json.load``ed contents of one .json file)."""
    raw_entities = annotation["form"]
    entities = [
        Entity(
            id=e["id"],
            text=e["text"],
            label=e["label"],
            box=tuple(e["box"]),  # type: ignore[arg-type]
        )
        for e in raw_entities
    ]
    by_id = {e.id: e for e in entities}

    # Deduplicate the symmetric linking pairs into a single undirected edge list.
    seen_edges: set[tuple[int, int]] = set()
    for e in raw_entities:
        for a, b in e["linking"]:
            if a in by_id and b in by_id:
                seen_edges.add((min(a, b), max(a, b)))
    links = sorted(seen_edges)

    # Build an adjacency lookup once, reused for scorable-pair resolution below.
    adjacency: dict[int, set[int]] = {e.id: set() for e in entities}
    for a, b in links:
        adjacency[a].add(b)
        adjacency[b].add(a)

    scorable_pairs: list[ScorablePair] = []
    for e in entities:
        if e.label != "question":
            continue
        answer_ids = sorted(
            (other for other in adjacency[e.id] if by_id[other].label == "answer"),
        )
        if not answer_ids:
            continue  # unlinked question — structural context only, not a scored target
        answer_entities = sorted((by_id[i] for i in answer_ids), key=reading_order_key)
        gold_text = " ".join(a.text for a in answer_entities if a.text)
        scorable_pairs.append(
            ScorablePair(
                question_id=e.id,
                question_text=e.text,
                answer_entity_ids=tuple(a.id for a in answer_entities),
                gold_answer_text=gold_text,
            )
        )

    return DocumentRecord(doc_id=doc_id, entities=entities, links=links, scorable_pairs=scorable_pairs)


def document_from_dict(d: dict) -> DocumentRecord:
    """Reconstruct a DocumentRecord from the serialized form written by funsd/download.py
    (``asdict`` of a DocumentRecord — plain dicts/lists/tuples-as-lists)."""
    return DocumentRecord(
        doc_id=d["doc_id"],
        entities=[Entity(id=e["id"], text=e["text"], label=e["label"], box=tuple(e["box"]))
                  for e in d["entities"]],
        links=[tuple(pair) for pair in d["links"]],
        scorable_pairs=[
            ScorablePair(
                question_id=p["question_id"],
                question_text=p["question_text"],
                answer_entity_ids=tuple(p["answer_entity_ids"]),
                gold_answer_text=p["gold_answer_text"],
            )
            for p in d["scorable_pairs"]
        ],
    )


def load_documents(path) -> list[DocumentRecord]:
    """Load the combined documents JSON file (config.DOCUMENTS_PATH) into a list of DocumentRecord."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [document_from_dict(d) for d in data]

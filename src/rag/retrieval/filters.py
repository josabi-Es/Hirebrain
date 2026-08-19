"""Qdrant filter builders from agent metadata dicts."""

from __future__ import annotations

from typing import Any

from qdrant_client.http import models as qmodels


def build_query_filter(
    filters: dict[str, Any] | None,
    *,
    sections: list[str] | None = None,
) -> qmodels.Filter | None:
    """Build a Qdrant Filter from flat metadata filters."""
    must: list[qmodels.Condition] = []
    must_not: list[qmodels.Condition] = []

    if filters:
        for key, value in filters.items():
            if value is None or value == "":
                continue
            if key == "exclude_sections" and isinstance(value, (list, tuple, set)):
                for section in value:
                    must_not.append(
                        qmodels.FieldCondition(
                            key="section",
                            match=qmodels.MatchValue(value=str(section)),
                        )
                    )
                continue
            if key == "sections" and isinstance(value, (list, tuple, set)):
                should = [
                    qmodels.FieldCondition(
                        key="section",
                        match=qmodels.MatchValue(value=str(section)),
                    )
                    for section in value
                ]
                if should:
                    must.append(qmodels.Filter(should=should))
                continue
            must.append(
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=str(value)),
                )
            )

    if sections:
        should = [
            qmodels.FieldCondition(
                key="section",
                match=qmodels.MatchValue(value=section),
            )
            for section in sections
        ]
        must.append(qmodels.Filter(should=should))

    if not must and not must_not:
        return None
    return qmodels.Filter(must=must or None, must_not=must_not or None)

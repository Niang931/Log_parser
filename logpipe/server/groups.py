"""Groups: several streams managed together, e.g. ``mail`` (the ldap and proxy streams) or ``iot``
(the camera and sensor streams). A stream is in at most one group (its ``group`` field holds the
group's id). A group's services are the services of its streams' sources: Ask lets a question
name a group to cover all of them, and tells the LLM which services belong together.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Server

KIND = "group"


class Groups:
    def __init__(self, server: Server):
        self.server = server
        self.control = server.control

    def _streams(self) -> list[dict]:
        return self.control.all("stream")

    def _sources_of(self, stream_id: str) -> list[dict]:
        s = self.server
        return [src for sid, src in s.sources.items() if (home := s.streams_for(sid)) and home[0]["id"] == stream_id]

    def services_of(self, group_id: str) -> list[str]:
        return sorted({src["service"] for st in self._streams() if st.get("group") == group_id
                       for src in self._sources_of(st["id"]) if src["service"]})

    def list(self) -> list[dict]:
        streams = self._streams()
        out = []
        for g in sorted(self.control.all(KIND), key=lambda g: g["name"]):
            members = [st for st in streams if st.get("group") == g["id"]]
            out.append(g | {"streams": [{"id": st["id"], "name": st["name"]} for st in members],
                            "services": self.services_of(g["id"])})
        return out

    def services(self) -> dict[str, list[str]]:
        """Group name -> its services (for Ask's catalog)."""
        return {g["name"]: g["services"] for g in self.list()}

    def _id(self, name: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "group"
        gid, n = base, 2
        while self.control.get(KIND, gid) is not None:
            gid, n = f"{base}-{n}", n + 1
        return gid

    def save(self, body: dict, group_id: str | None = None) -> dict:
        """Create or rename a group; ``streams`` (ids), when given, is the full list of its streams:
        listed ones move into it (from any other group), the others leave it."""
        name = str(body.get("name", "")).strip()
        if not name or len(name) > 60:
            raise ValueError("a group needs a name (at most 60 characters)")
        if any(g["name"].lower() == name.lower() and g["id"] != group_id for g in self.control.all(KIND)):
            raise ValueError(f"there is already a group called {name!r}")
        if group_id is not None and self.control.get(KIND, group_id) is None:
            raise KeyError(group_id)
        wanted = body.get("streams")
        if wanted is not None:
            if not isinstance(wanted, list) or not all(isinstance(s, str) for s in wanted):
                raise ValueError("streams is a list of stream ids")
            known = {st["id"] for st in self._streams()}
            if unknown := [s for s in wanted if s not in known]:
                raise ValueError(f"unknown stream {unknown[0]!r}")
        group_id = group_id or self._id(name)
        existing = self.control.get(KIND, group_id) or {"id": group_id}
        self.control.put(KIND, group_id, existing | {"name": name, "description": str(body.get("description") or "")[:300]})
        if wanted is not None:
            for st in self._streams():
                inside = st["id"] in wanted
                if inside and st.get("group") != group_id:
                    self.control.update("stream", st["id"], group=group_id)
                elif not inside and st.get("group") == group_id:
                    self.control.update("stream", st["id"], group=None)
        self.server.events.notify()
        return next(g for g in self.list() if g["id"] == group_id)

    def delete(self, group_id: str) -> None:
        if self.control.get(KIND, group_id) is None:
            raise KeyError(group_id)
        for st in self._streams():
            if st.get("group") == group_id:
                self.control.update("stream", st["id"], group=None)
        self.control.delete(KIND, group_id)
        self.server.events.notify()

    def set_stream(self, stream_id: str, group_id: str | None) -> None:
        """Move one stream into a group, or out of any (None)."""
        if group_id is not None and self.control.get(KIND, group_id) is None:
            raise ValueError(f"unknown group {group_id!r}")
        self.control.update("stream", stream_id, group=group_id)

"""Adaptation Guard Types (FASE 8)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GuardCheck:
    """Resultado de un chequeo individual (auditable)."""

    name: str
    ok: bool
    detail: str

    def render(self) -> str:
        mark = "✓" if self.ok else "✗"
        return f"  {mark} {self.name}: {self.detail}"


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Veredicto completo del guard para una propuesta."""

    passed: bool
    checks: tuple[GuardCheck, ...]

    @property
    def failed_checks(self) -> tuple[GuardCheck, ...]:
        return tuple(c for c in self.checks if not c.ok)

    def render(self) -> str:
        lines = [c.render() for c in self.checks]
        if self.passed:
            lines.append("  ACCEPT")
        else:
            lines.append(f"  REJECT ({len(self.failed_checks)} chequeo(s) fallido(s))")
        return "\n".join(lines)
"""Prompt helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    template: str = "<image>\n{instruction}"

    def format(self, instruction: str) -> str:
        return self.template.format(instruction=instruction)


DEFAULT_TEMPLATE = PromptTemplate()


__all__ = ["PromptTemplate", "DEFAULT_TEMPLATE"]

from __future__ import annotations
from typing import Any

from ovos_plugin_manager.templates.transformers import DialogTransformer
from ovos_solver_plugin_gemini_chat.engines import GeminiChatCompletionsSolver


class GeminiDialogTransformer(DialogTransformer):
    def __init__(
        self: GeminiDialogTransformer,
        name: str = "ovos-dialog-transformer-hugchat-plugin",
        priority: int = 10,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, priority, config)
        self.solver = GeminiChatCompletionsSolver(
            {
                "api_key": self.config.get("key"),
                "initial_prompt": (
                    "your task is to rewrite text as "
                    "if it was spoken by a different character"
                ),
            }
        )

    def transform(
        self: GeminiDialogTransformer,
        dialog: str,
        context: dict | None = None,
    ) -> tuple[str, dict]:
        """
        Optionally transform passed dialog and/or return additional context
        :param dialog: str utterance to mutate before TTS
        :returns: str mutated dialog
        """
        if self.solver is None:
            return dialog, context or {}
        context = context or {}
        prompt = (
            context.get("prompt")
            or self.config.get("rewrite_prompt")
        )
        if not prompt:
            return dialog, context
        return (
            self.solver.get_spoken_answer(
                f"{prompt}: {dialog}",
                lang=context.get("lang")
            ) or "",
            context,
        )

from __future__ import annotations
from typing import Any
from ovos_solver_gemini_chat.engines import GeminiChatCompletionsSolver


class GeminiChatSolver(GeminiChatCompletionsSolver):
    """Default engine"""

    def __init__(
        self: GeminiChatSolver,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)
        self.default_persona = (
            config.get("persona")
            or "helpful, creative, clever, and very friendly"
        )
        self.qa_pairs = []

    def get_prompt(
        self: GeminiChatSolver,
        utt: str,
        persona: str | None = None,
    ) -> str:
        persona = persona or self.config.get("persona") or self.default_persona
        initial_prompt = (
            "You are a helpful assistant. "
            "You understand all languages. "
            "You give short and factual answers. "
            "You answer in the same language the question was asked. "
            f"You are {persona}."
        )
        prompt = f"{initial_prompt}\n\n{utt}\n"
        return prompt

    # Officially exported method
    def get_spoken_answer(
        self: GeminiChatSolver,
        query: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        context = context or {}
        persona = context.get("persona") or self.default_persona
        prompt = self.get_prompt(query, persona)
        response = self._do_api_request(prompt)
        answer = response.strip()
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        return answer


if __name__ == "__main__":
    bot = GeminiChatSolver({"api_key": "your-gemini-api-key"})
    for utt in bot.stream_utterances("describe quantum mechanics very briefly"):
        print(utt)
    # Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level,
    # where classical physics breaks down.

    print(bot.get_spoken_answer("Quem encontrou o caminho maritimo para o Brasil?"))
    # Pedro Álvares Cabral.

from __future__ import annotations
import warnings
from ovos_solver_gemini_chat.engines import GeminiChatCompletionsSolver


class GeminiChatSolver(GeminiChatCompletionsSolver):
    def __init__(self, *args, **kwargs):
        """
        Initializes the solver and issues a deprecation warning.
        
        A DeprecationWarning is raised advising to use GeminiChatCompletionsSolver instead.
        """
        warnings.warn(
            "use GeminiChatCompletionsSolver instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# for ovos-persona
GEMINI_DEMO = {
  "name": "Gemini Chat",
  "solvers": [
    "ovos-solver-plugin-gemini-chat",
  ],
  "ovos-solver-plugin-gemini-chat": {
    "api_key": "your-gemini-api-key",
    "model": "gemini-2.5-flash"
  }
}


if __name__ == "__main__":
    bot = GeminiChatCompletionsSolver(GEMINI_DEMO["ovos-solver-plugin-gemini-chat"])
    for utt in bot.stream_utterances("describe quantum mechanics very briefly"):
        print(utt)
    # Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level,
    # where classical physics breaks down.

    print(bot.get_spoken_answer("Quem encontrou o caminho maritimo para o Brasil?"))
    # Pedro Álvares Cabral.

from __future__ import annotations
from typing import (
    Any,
    Generator,
)

# import requests
import re
from google.genai import types, Client
from google.genai.chats import Chat

from ovos_plugin_manager.templates.solvers import QuestionSolver
from ovos_utils.log import LOG


NON_TEXT_CHARS = re.compile(r'[^\w\s\d\.\,\?\!\:\;\-\&\'\"]')
ENDING_CHARS = [".", "!", "?", ":"]
ENDING_EXPRS = [
    re.compile(rf"(?<=[^\s][a-zA-Z]){re.escape(c)}\"?(?=[\s\n])")
    for c in ENDING_CHARS
]


class GeminiChatCompletionsSolver(QuestionSolver):
    def __init__(
        self: GeminiChatCompletionsSolver,
        config: dict[str, Any] | None = None,
    ) -> None:
        config = config or {}
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        self.reasoning = self.config.get("enable_reasoning", False)
        if not self.api_key:
            LOG.error("'api_key' not set in config")
            raise ValueError("api key must be set")

        self.client = self._setup_client()
        LOG.info("Gemini client initialized successfully.")

        try:
            self.available_models = list(
                filter(
                    lambda m: m.startswith("gemini"),
                    map(
                        lambda m: m.name.split("/")[-1],  # type: ignore
                        self.client.models.list()
                    ),
                )
            )
            LOG.info(f"Available Gemini models: {self.available_models}")
            self.model = self.config.get("model", "gemini-2.5-flash")
        except Exception as e:
            self.available_models = []
            self.model = "undefined"
            LOG.warning(f"unable to select model: {e}")
            LOG.warning(f"trying to use whichever model was selected previously")

        self.initial_prompt = config.get("initial_prompt", "You are a helpful assistant.")
        self.chatbot = self._setup_chat()
        LOG.info(f"Using Gemini model: {self.model}")

    def _setup_client(self: GeminiChatCompletionsSolver) -> Client:
        """Authenticate user for a Hugging Chat session and retrieve cookie jar."""
        return Client(api_key=self.api_key)
 
    def _setup_chat(self: GeminiChatCompletionsSolver) -> Chat:
        """Initialize chat session."""
        return self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1 if self.reasoning else 0
                ),
                system_instruction=self.initial_prompt,
            ),
        )

    # Gemini integration
    def _do_api_request(
        self: GeminiChatCompletionsSolver,
        prompt: str,
    ) -> str:
        """Send query to Gemini"""
        response = self.chatbot.send_message(prompt)
        return response.text or ""

    def _do_streaming_api_request(
        self: GeminiChatCompletionsSolver,
        prompt: str,
    ) -> Generator[str, None, None]:
        """Send query to Gemini"""
        for chunk in self.chatbot.send_message_stream(prompt):
            if chunk.text is not None:
                yield NON_TEXT_CHARS.sub('', chunk.text)

    def get_spoken_answer(
        self: GeminiChatCompletionsSolver,
        query: str,
        **kwargs: Any,
    ) -> str | None:
        response = self._do_api_request(query)
        answer = response.strip()
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        return answer

    # Officially exported method
    def stream_utterances(
        self: GeminiChatCompletionsSolver,
        query: str,
    ) -> Generator[str, None, None]:
        answer = ""
        for chunk in self._do_streaming_api_request(query):
            end_detection = [r.search(chunk) is not None for r in ENDING_EXPRS]
            if any(end_detection):
                ending_char = ENDING_CHARS[end_detection.index(True)]
                ending_expr = ENDING_EXPRS[end_detection.index(True)]
                elements = ending_expr.split(chunk)
                next_chunk = elements.pop(-1)
                ending_chunk = ending_char.join(elements + [''])
                answer += ending_chunk
                answer = ' '.join(answer.split())
                LOG.debug(f"Gemini phrase: {answer}")
                if answer.strip():
                    yield answer
                answer = next_chunk
            else:
                answer += chunk

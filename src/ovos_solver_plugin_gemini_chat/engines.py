from __future__ import annotations
from typing import (
    Any,
    Generator,
)

# import requests
import re
from google.genai import types, Client
from google.genai.chats import Chat

from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_plugin_manager.templates.solvers import QuestionSolver, ChatMessageSolver
from ovos_utils.log import LOG


NON_TEXT_CHARS = re.compile(r'[^\w\s\d\.\,\?\!\:\;\-\&\'\"]')
ENDING_CHARS = [".", "!", "?", ":"]
ENDING_EXPRS = [
    re.compile(rf"(?<=[^\s][a-zA-Z]){re.escape(c)}\"?(?=[\s\n])")
    for c in ENDING_CHARS
]


def post_process_sentence(text: str) -> str:
    return NON_TEXT_CHARS.sub('', text)


class GeminiCompletionsSolver(QuestionSolver):
    def __init__(
        self: GeminiCompletionsSolver,
        config: dict[str, Any] | None = None,
        translator: LanguageTranslator | None = None,
        detector: LanguageDetector | None = None,
        priority: int = 50,
        enable_tx: bool = False,
        enable_cache: bool = False,
        internal_lang: str | None = None,
    ) -> None:
        config = config or {}
        super().__init__(
            config=config,
            translator=translator,
            detector=detector, priority=priority,
            enable_tx=enable_tx, enable_cache=enable_cache,
            internal_lang=internal_lang,
        )
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
        self.content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1 if self.reasoning else 0
            ),
            system_instruction=self.initial_prompt,
        )
        LOG.info(f"Using Gemini model: {self.model}")

    def _setup_client(self: GeminiCompletionsSolver) -> Client:
        """Authenticate user for a Hugging Chat session and retrieve cookie jar."""
        return Client(api_key=self.api_key)

    # Gemini integration
    def _do_api_request(
        self: GeminiCompletionsSolver,
        prompt: str,
    ) -> str | None:
        """Send query to Gemini"""
        response = self.client.models.generate_content(
            model=self.model,
            config=self.content_config,
            contents=prompt,
        )
        return response.text

    def _do_streaming_api_request(
        self: GeminiCompletionsSolver,
        prompt: str,
    ) -> Generator[str, None, None]:
        """Send query to Gemini"""
        content_iterator = self.client.models.generate_content_stream(
            model=self.model,
            config=self.content_config,
            contents=prompt,
        )
        for chunk in content_iterator:
            if chunk.text is not None:
                yield post_process_sentence(chunk.text)

    def get_spoken_answer(
        self: GeminiCompletionsSolver,
        query: str,
        lang: str | None = None,
        units: str | None = None
    ) -> str | None:
        response = self._do_api_request(query)
        answer = response.strip() if response else None
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        return answer

    # Officially exported method
    def stream_utterances(
        self: GeminiCompletionsSolver,
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


class GeminiChatCompletionsSolver(ChatMessageSolver):
    def __init__(
        self: GeminiChatCompletionsSolver,
        config: dict[str, Any] | None = None,
        translator: LanguageTranslator | None = None,
        detector: LanguageDetector | None = None,
        priority: int = 50,
        enable_tx: bool = False,
        enable_cache: bool = False,
        internal_lang: str | None = None,
    ) -> None:
        config = config or {}
        super().__init__(
            config=config,
            translator=translator,
            detector=detector, priority=priority,
            enable_tx=enable_tx, enable_cache=enable_cache,
            internal_lang=internal_lang,
        )
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
        self.content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1 if self.reasoning else 0
            ),
            system_instruction=self.initial_prompt,
        )
        self.chat = self._setup_chat()
        LOG.info(f"Using Gemini model: {self.model}")

        self.memory = config.get("enable_memory", True)
        self.max_utts = config.get("memory_size", 3)
        self.qa_pairs = []  # tuple of q+a
        if "persona" in config:
            LOG.warning("'persona' config option is deprecated, use 'system_prompt' instead")
        if "initial_prompt" in config:
            LOG.warning("'initial_prompt' config option is deprecated, use 'system_prompt' instead")
        self.system_prompt = config.get("system_prompt") or config.get("initial_prompt")
        if not self.system_prompt:
            self.system_prompt =  "You are a helpful assistant."
            LOG.error(f"system prompt not set in config! defaulting to '{self.system_prompt}'")

    def _setup_client(self: GeminiChatCompletionsSolver) -> Client:
        """Authenticate user for a Hugging Chat session and retrieve cookie jar."""
        return Client(api_key=self.api_key)

    def _setup_chat(self: GeminiChatCompletionsSolver) -> Chat:
        """Initialize chat session."""
        return self.client.chats.create(
            model=self.model,
            config=self.content_config,
        )

    def _do_api_request(
        self: GeminiChatCompletionsSolver,
        prompt: str,
    ) -> str | None:
        """Send query to Gemini"""
        response = self.chat.send_message(prompt)
        return response.text

    def _do_streaming_api_request(
        self: GeminiChatCompletionsSolver,
        prompt: str,
    ) -> Generator[str, None, None]:
        """Send query to Gemini"""
        for chunk in self.chat.send_message_stream(prompt):
            if chunk.text is not None:
                yield post_process_sentence(chunk.text)

    def get_chat_history(
        self: GeminiChatCompletionsSolver,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """
        Builds the chat history as a list of messages, starting with a system prompt.
        """
        messages = []
        for message in self.chat.get_history():
            messages.append(
                {
                    "role": message.role,
                    "content": message.parts[0].text if message.parts else "",
                }
            )
        return messages

    def get_messages(
        self: GeminiChatCompletionsSolver,
        utt: str,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """
        Builds a list of chat messages including the system prompt, recent conversation history, and the current user utterance.
        """
        messages = self.get_chat_history(system_prompt)
        messages.append({"role": "user", "content": utt})
        return messages

    # abstract Solver methods
    def continue_chat(
        self: GeminiChatCompletionsSolver,
        messages: list[dict],
        lang: str | None = None,
        units: str | None = None,
    ) -> str | None:
        """
        Generates a chat response using the provided message history and updates memory if enabled.
        """
        query = messages[-1]["content"]
        response = self._do_api_request(query)
        answer = response.strip() if response else None
        if not answer or not answer.strip("?") or not answer.strip("_"):
            return None
        return answer

    def stream_chat_utterances(
        self: GeminiChatCompletionsSolver,
        messages: list[dict],
        lang: str | None = None,
        units: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream utterances for the given chat history as they become available.
        """
        query = messages[-1]["content"]
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

    def stream_utterances(
        self: GeminiChatCompletionsSolver,
        query: str,
        lang: str | None = None,
        units: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream utterances for the given query as they become available.
        """
        messages = [{"role": "user", "content": query}]
        yield from self.stream_chat_utterances(messages, lang, units)

    def get_spoken_answer(
        self: GeminiChatCompletionsSolver,
        query: str,
        lang: str | None = None,
        units: str | None = None,
    ) -> str | None:
        """
        Obtain the spoken answer for a given query.
        """
        messages = [{"role": "user", "content": query}]
        # just for api compat since it's a subclass, shouldn't be directly used
        return self.continue_chat(messages=messages, lang=lang, units=units)

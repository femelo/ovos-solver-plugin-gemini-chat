# <img src='https://raw.githack.com/FortAwesome/Font-Awesome/master/svgs/solid/robot.svg' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> HuggingChat Persona

Talk to Gemini through OpenVoiceOS.

Uses [Gemini](https://gemini.google.com) via [google-genai](https://github.com/googleapis/python-genai) to create some fun interactions. Phrases not explicitly handled by other skills will be run by a LLM, so nearly every interaction will have _some_ response.

## Usage

```python
from ovos_solver_gemini_chat import GeminiChatSolver

bot = GeminiChatSolver(
    {
        "api_key": "your-gemini-api-key",
        "persona": "helpful, creative, clever, and very friendly"
    }
)
print(bot.get_spoken_answer("describe quantum mechanics very briefly"))
# Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level,
# where classical physics breaks down.
print(bot.get_spoken_answer("Quem encontrou o caminho maritimo para o Brasil"))
# Pedro Álvares Cabral.

```

This plugin will work with [ovos-persona-server](https://github.com/OpenVoiceOS/ovos-persona-server)

## Configuration

This plugin can be configured as follows

```json
{
    "api_key": "your-gemini-api-key",
    "initial_prompt": "You are a helpful assistant."
}
```

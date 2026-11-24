"""Eval service factories: judge and persona LLM on OpenAI, persona voice on Cartesia.

Referenced from scenario yaml via dotted paths, e.g.
``judge.eval.factory: evals.factories.openai_judge``. Keys come from
``server/.env``.
"""

import os

from dotenv import load_dotenv
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.evals.services import DEFAULT_OPENAI_MODEL, ollama_service
from pipecat.services.cartesia.tts import CartesiaHttpTTSService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tts_service import TTSService

# The eval harness doesn't read .env itself; factories run inside it.
load_dotenv(override=True)


def _openai(config: dict) -> OpenAILLMService:
    return OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            model=config.get("model", DEFAULT_OPENAI_MODEL),
            extra=config.get("extra") or {},
        ),
    )


def openai_judge(config: dict) -> LLMClassifier:
    """Judge on OpenAI; handles concurrent per-turn calls without queueing."""
    return LLMClassifier(llm=_openai(config), timeout=float(config.get("timeout", 30)))


def openai_persona(config: dict) -> LLMService:
    """Persona LLM on OpenAI."""
    return _openai(config)


def cartesia_speech(config: dict) -> TTSService:
    """Persona voice on Cartesia HTTP TTS; expressive delivery for voice perception.

    Config keys: ``voice`` (required), optional ``model`` (default sonic-2)
    and ``api_key`` (default ``$CARTESIA_API_KEY``).
    """
    api_key = config.get("api_key") or os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise RuntimeError("Cartesia API key not found: set $CARTESIA_API_KEY or speech.api_key")
    return CartesiaHttpTTSService(
        api_key=api_key,
        settings=CartesiaHttpTTSService.Settings(
            voice=str(config.get("voice", "")),
            model=config.get("model") or "sonic-2",
        ),
    )


def local_judge(config: dict) -> LLMClassifier:
    """Ollama judge with a timeout sized for serial local inference.

    Concurrent per-turn judge calls queue behind one local GPU; the
    classifier's 10s default drops queued calls, so allow 60s (override
    with ``timeout:`` in the ``judge.eval`` block).
    """
    return LLMClassifier(llm=ollama_service(config), timeout=float(config.get("timeout", 60)))

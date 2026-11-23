"""Judge classifier factory for the local Ollama judge."""

from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.evals.services import ollama_service


def local_judge(config: dict) -> LLMClassifier:
    """Default Ollama judge with a timeout sized for serial local inference.

    Concurrent per-turn judge calls queue behind one local GPU; the
    classifier's 10s default drops queued calls, so allow 60s (override
    with ``timeout:`` in the ``judge.eval`` block).
    """
    return LLMClassifier(llm=ollama_service(config), timeout=float(config.get("timeout", 60)))

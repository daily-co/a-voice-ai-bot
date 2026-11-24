#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""pipecat-quickstart - Pipecat Voice Agent

Healthcare front-desk bot: verifies a caller's identity and updates contact
details on file. Cascade pipeline: Speech-to-Text → LLM → Text-to-Speech.

Required AI services:
- Deepgram (Speech-to-Text)
- Openai_Responses (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.expressive.processors.perception_reader import PerceptionReader
from pipecat.expressive.processors.tone_reader import PipecatToneSource
from pipecat.expressive.processors.voice_reader import PipecatVoiceSource
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


async def verify_identity(params: FunctionCallParams, full_name: str, date_of_birth: str):
    """Verify the caller's identity before any change to their file.

    Args:
        full_name: The patient's full name as given on the call.
        date_of_birth: The patient's date of birth, e.g. "March 12, 1985".
    """
    # Stub verification; swap for real patient lookup.
    await params.result_callback({"verified": True, "patient_id": "PT-48213"})


async def update_address(params: FunctionCallParams, street: str, city: str, state: str, zip_code: str):
    """Update the patient's mailing address on file.

    Args:
        street: Street address including any unit, e.g. "123 Maple St, Apt 4B".
        city: City name.
        state: State name or abbreviation, e.g. "Illinois" or "IL".
        zip_code: ZIP code, e.g. "60201".
    """
    # Stub update; swap for real EHR/CRM write.
    await params.result_callback(
        {
            "status": "updated",
            "street": street,
            "city": city,
            "state": state,
            "zip_code": zip_code,
        }
    )


async def run_bot(transport: BaseTransport):
    """Main bot logic."""
    logger.info("Starting bot")

    # Speech-to-Text service
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Text-to-Speech service
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"), # Skylar (en-US)
        ),
    )

    # LLM service
    llm = OpenAIResponsesLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAIResponsesLLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            system_instruction=(
                "You are the front-desk assistant for Lakeshore Health, a healthcare "
                "clinic, in a voice conversation. Your responses will be spoken aloud, "
                "so avoid emojis, bullet points, or other formatting that can't be "
                "spoken. Keep replies brief, warm, and professional. You help callers "
                "update the contact details on their patient file. Before making any "
                "change, verify the caller's identity: collect their full name and "
                "date of birth and call verify_identity. To change a mailing address, "
                "collect the complete new address (street with any unit, city, state, "
                "and ZIP code), read it back to the caller to confirm, and only then "
                "call update_address. Tell the caller clearly once the update is done."
            ),
        ),
    )

    # Direct functions: schema auto-derived from signature and docstring.
    context = LLMContext(tools=[verify_identity, update_address])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # LLM gates end-of-turn: incomplete user turns suppress bot response.
            user_turn_strategies=FilterIncompleteUserTurnStrategies(),
        ),
    )

    # Pipeline - assembled from reusable components
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            # Reads each user turn's emotion and sarcasm (RCS Perception).
            # Reads each turn's words (tone) and audio (voice) into its perception.
            PerceptionReader(sources=[PipecatToneSource(), PipecatVoiceSource()]),
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            # RCS Expression: LLM writes delivery/sound tags the TTS performs.
            enable_expressive_conversation=True,
            expression_policy=ExpressionPolicy(
                guidance=[
                    ExpressionGuidance.MATCH_STYLE,
                    ExpressionGuidance.SLOW_FOR_DETAILS,
                    "Each time the user sounds frustrated or impatient, in "
                    "their words or in the reading, acknowledge it in one "
                    "short clause (like 'I know, sorry about that') before "
                    "continuing, even if you already apologized earlier in "
                    "the call. Keep the rest of the reply short.",
                ],
            ),
        ),
        observers=[],
    )

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Kick off the conversation
        context.add_message(
            {"role": "developer", "content": "Start by concisely introducing yourself."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        # Serves `pipecat eval run` over the dev runner's /ws route.
        "eval": lambda: EvalTransportParams(audio_in_enabled=True),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

"""
VibeVoice Inference Server (Mac Edition)

FastAPI server for VibeVoice-Large TTS model.
Optimized for Apple Silicon (MPS) - standalone, no Redis queue.

Endpoints:
- POST /synthesize - Generate speech from text with voice cloning (single speaker)
- POST /synthesize/stream - Generate speech with real-time streaming (returns PCM chunks)
- POST /dialogue - Generate multi-speaker dialogue (native multi-speaker, 1-pass)
- POST /dialogue/stream - Generate multi-speaker dialogue with real-time streaming
- GET /health - Health check
- GET /ping - Health check (alias)
- GET /voices - List available voice samples
- POST /reload-voices - Reload voice samples from disk
"""

import os
import io
import time
import logging
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("vibevoice")

# Suppress noisy progress bars from transformers/tqdm
import transformers
transformers.logging.set_verbosity_error()

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
model = None
processor = None
model_loaded = False
device = "cpu"
voice_samples: Dict[str, torch.Tensor] = {}
gen_lock = asyncio.Lock()
AsyncAudioStreamer = None  # Loaded dynamically with model

# Batch queue infrastructure
dialogue_batch_queue: Optional[asyncio.Queue] = None
dialogue_batch_processor_task: Optional[asyncio.Task] = None


@dataclass
class DialogueBatchRequest:
    """A single /dialogue request waiting in the batch queue."""
    formatted_script: str
    voice_sample_list: List
    cfg_scale: float
    future: asyncio.Future
    # Metadata for response
    segment_count: int
    speaker_count: int
    allow_voice_sharing: bool


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
MODEL_PATH = os.getenv("VIBEVOICE_MODEL_LOCAL_PATH") or os.getenv("VIBEVOICE_MODEL_PATH", "rsxdalv/VibeVoice-Large")
OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", "24000"))

# Default guidance scale range
CFG_MIN = float(os.getenv("CFG_MIN", "1.0"))
CFG_MAX = float(os.getenv("CFG_MAX", "2.0"))
CFG_DEFAULT = float(os.getenv("CFG_DEFAULT", "1.3"))

# Behavior: reject >4 speakers unless explicitly allowed
ALLOW_VOICE_SHARING_DEFAULT = os.getenv("ALLOW_VOICE_SHARING_DEFAULT", "false").lower() in ("1", "true", "yes")

# Batching configuration (lower defaults for Mac)
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1"))
BATCH_TIMEOUT_SECONDS = float(os.getenv("BATCH_TIMEOUT_SECONDS", "10.0"))


# -----------------------------------------------------------------------------
# Request/Response models
# -----------------------------------------------------------------------------
class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", max_length=10000)
    speaker: str = Field(..., description="Speaker/voice name (must match a .wav file in voices dir)")
    cfg_scale: float = Field(default=CFG_DEFAULT, ge=CFG_MIN, le=CFG_MAX, description="Classifier-free guidance scale")


class DialogueSegment(BaseModel):
    text: str = Field(..., description="Text for this segment")
    speaker: str = Field(..., description="Speaker name for this segment")


class DialogueRequest(BaseModel):
    segments: List[DialogueSegment] = Field(..., description="Dialogue segments (ordered turns)")
    cfg_scale: float = Field(default=CFG_DEFAULT, ge=CFG_MIN, le=CFG_MAX, description="Classifier-free guidance scale")
    allow_voice_sharing: bool = Field(
        default=ALLOW_VOICE_SHARING_DEFAULT,
        description="If >4 unique speakers, allow extra speakers to share the first 4 voice slots.",
    )


class VoiceInfo(BaseModel):
    name: str
    file_path: str


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _handle_memory_error():
    """Clear memory after error - works for any device."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # MPS doesn't have explicit cache clearing - handled by OS
    logger.warning("Memory error recovery: ran garbage collection")


def _to_mono_24k(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample to OUTPUT_SAMPLE_RATE and convert to mono."""
    if sample_rate != OUTPUT_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, OUTPUT_SAMPLE_RATE)
        waveform = resampler(waveform)
    if waveform.dim() == 1:
        return waveform.cpu()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    else:
        waveform = waveform.squeeze(0)
    return waveform.cpu()


def load_voice_samples() -> None:
    """Load voice sample files from voices directory."""
    global voice_samples
    voice_samples = {}

    if not VOICES_DIR.exists():
        logger.warning("Voices directory does not exist: %s", VOICES_DIR)
        return

    for wav_file in sorted(VOICES_DIR.glob("*.wav")):
        voice_name = wav_file.stem
        try:
            waveform, sample_rate = torchaudio.load(wav_file)
            waveform = _to_mono_24k(waveform, sample_rate)
            voice_samples[voice_name] = waveform
            logger.info(
                "Loaded voice: %s (%.1fs)",
                voice_name,
                waveform.shape[-1] / OUTPUT_SAMPLE_RATE,
            )
        except Exception as e:
            logger.exception("Failed to load voice sample %s: %s", wav_file, e)

    logger.info("Loaded %d total voice samples", len(voice_samples))


def load_model() -> None:
    """Load the VibeVoice model and processor."""
    global model, processor, model_loaded, device

    if model_loaded:
        logger.warning("Model already loaded, skipping reload")
        return

    logger.info("Loading VibeVoice model from: %s", MODEL_PATH)

    # Download model from HuggingFace if not cached locally
    model_path = MODEL_PATH
    local_model_path = os.getenv("VIBEVOICE_MODEL_LOCAL_PATH")
    if local_model_path:
        model_dir = Path(local_model_path)
        if not (model_dir / "config.json").exists():
            logger.info("Model not found locally, downloading from HuggingFace...")
            from huggingface_hub import snapshot_download
            hf_model_id = os.getenv("VIBEVOICE_MODEL_PATH", "rsxdalv/VibeVoice-Large")
            model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(hf_model_id, local_dir=str(model_dir))
            logger.info("Model downloaded to: %s", model_dir)
        model_path = str(model_dir)

    try:
        global AsyncAudioStreamer
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        from vibevoice.modular.streamer import AsyncAudioStreamer as _AsyncAudioStreamer
        AsyncAudioStreamer = _AsyncAudioStreamer

        # Device detection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using device: CUDA")
            logger.info("GPU: %s", torch.cuda.get_device_name(0))
            logger.info("GPU Memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using device: MPS (Apple Silicon)")
        else:
            device = "cpu"
            logger.info("Using device: CPU")

        processor = VibeVoiceProcessor.from_pretrained(model_path)
        logger.info("Processor loaded")

        # CUDA uses bfloat16, MPS uses float16, CPU uses float32
        if device == "cuda":
            dtype = torch.bfloat16
        elif device == "mps":
            dtype = torch.float16  # MPS supports float16, saves ~50% memory vs float32
        else:
            dtype = torch.float32
        logger.info("Using dtype: %s", dtype)

        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=os.getenv("VIBEVOICE_ATTN_IMPL", "sdpa"),
        ).to(device)

        model.eval()
        model_loaded = True
        logger.info("VibeVoice model loaded successfully on %s", device)

        load_voice_samples()

    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise


def _extract_waveform_from_outputs(outputs) -> torch.Tensor:
    """Extract a single mixed waveform tensor from VibeVoice outputs."""
    if hasattr(outputs, "speech_outputs") and outputs.speech_outputs is not None:
        speech_out = outputs.speech_outputs
        if isinstance(speech_out, list) and len(speech_out) > 0:
            item = speech_out[0]
            if isinstance(item, (list, tuple)):
                tensors: List[torch.Tensor] = []
                for t in item:
                    if isinstance(t, torch.Tensor):
                        tensors.append(t)
                    else:
                        tensors.append(torch.tensor(t))
                wav = torch.stack(tensors, dim=0).sum(dim=0)
                return wav
            if isinstance(item, torch.Tensor):
                return item
            return torch.tensor(item)

        if isinstance(speech_out, torch.Tensor):
            return speech_out
        return torch.tensor(speech_out)

    for name in ("waveform", "audio"):
        if hasattr(outputs, name) and getattr(outputs, name) is not None:
            val = getattr(outputs, name)
            if isinstance(val, torch.Tensor):
                return val
            return torch.tensor(val)

    if hasattr(outputs, "sequences") and outputs.sequences is not None:
        seq = outputs.sequences
        if isinstance(seq, torch.Tensor):
            if hasattr(model, "decode_audio"):
                return model.decode_audio(seq)
            if hasattr(model, "audio_encoder") and hasattr(model.audio_encoder, "decode"):
                return model.audio_encoder.decode(seq)
            if seq.dtype in (torch.float16, torch.float32, torch.bfloat16):
                return seq
        raise ValueError("Found sequences but cannot decode to audio.")

    raise ValueError(f"Could not extract audio from output type: {type(outputs)}")


def _normalize_wav(wav: torch.Tensor) -> torch.Tensor:
    """Normalize output to 1D float32 CPU tensor."""
    if not isinstance(wav, torch.Tensor):
        wav = torch.tensor(wav)

    wav = wav.detach().cpu().float()

    if wav.dim() == 2 and wav.shape[0] == 1:
        wav = wav.squeeze(0)
    elif wav.dim() > 2:
        wav = wav.reshape(-1)

    return wav.contiguous()


def _extract_single_waveform(item) -> torch.Tensor:
    """Extract waveform from a single batch item's output."""
    if isinstance(item, torch.Tensor):
        return item.squeeze()
    elif isinstance(item, (list, tuple)):
        tensors = []
        for t in item:
            wav = torch.tensor(t) if not isinstance(t, torch.Tensor) else t
            tensors.append(wav.squeeze())
        return torch.stack(tensors, dim=0).sum(dim=0)
    else:
        import numpy as np
        if isinstance(item, np.ndarray):
            return torch.from_numpy(item).squeeze()
    raise ValueError(f"Unknown output format: {type(item)}")


def _extract_batch_waveforms(outputs, batch_size: int) -> List[torch.Tensor]:
    """Extract audio for each item in batch."""
    if hasattr(outputs, "speech_outputs") and outputs.speech_outputs is not None:
        speech_out = outputs.speech_outputs

        if isinstance(speech_out, list) and len(speech_out) == batch_size:
            results = []
            for item in speech_out:
                wav = _extract_single_waveform(item)
                results.append(_normalize_wav(wav))
            return results

        if batch_size == 1:
            wav = _extract_waveform_from_outputs(outputs)
            return [_normalize_wav(wav)]

    raise ValueError(f"Cannot extract {batch_size} waveforms from outputs")


def _build_script_and_voices(
    segments: List[Tuple[str, str]],
    allow_voice_sharing: bool,
) -> Tuple[str, List[str], List]:
    """Build formatted script and voice sample list."""
    ordered_speakers: List[str] = []
    for text, spk in segments:
        if spk not in ordered_speakers:
            ordered_speakers.append(spk)

    unique_count = len(ordered_speakers)
    if unique_count == 0:
        raise ValueError("No speakers provided")

    if unique_count > 4 and not allow_voice_sharing:
        raise ValueError(f"VibeVoice supports up to 4 unique speakers per call; got {unique_count}")

    speaker_to_slot: Dict[str, int] = {}
    for i, spk in enumerate(ordered_speakers):
        if i < 4:
            speaker_to_slot[spk] = i + 1
        else:
            speaker_to_slot[spk] = ((i) % 4) + 1

    for spk in ordered_speakers:
        if spk not in voice_samples:
            raise ValueError(f"Voice '{spk}' not found. Available: {list(voice_samples.keys())}")

    lines: List[str] = []
    for text, spk in segments:
        t = (text or "").strip()
        if not t:
            continue
        slot = speaker_to_slot[spk]
        lines.append(f"Speaker {slot}: {t}")

    if not lines:
        raise ValueError("No non-empty text segments to synthesize")

    formatted_script = "\n".join(lines)
    base_speakers = ordered_speakers[:4]
    voice_sample_list = [voice_samples[spk].contiguous().cpu().numpy() for spk in base_speakers]

    return formatted_script, ordered_speakers, voice_sample_list


def generate_multi_speaker_audio(
    segments: List[Tuple[str, str]],
    cfg_scale: float,
    allow_voice_sharing: bool,
) -> torch.Tensor:
    """Generate audio from multiple speakers in a single inference pass."""
    formatted_script, ordered_speakers, voice_sample_list = _build_script_and_voices(
        segments, allow_voice_sharing=allow_voice_sharing
    )

    logger.info(
        "Generating: %d segments, %d unique speakers (allow_voice_sharing=%s)",
        len(segments),
        len(set([s for _, s in segments])),
        allow_voice_sharing,
    )

    inputs = processor(
        text=[formatted_script],
        voice_samples=[voice_sample_list],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
        )

    wav = _extract_waveform_from_outputs(outputs)
    wav = _normalize_wav(wav)

    del inputs
    del outputs

    return wav


class MemoryError(Exception):
    """Raised when device runs out of memory but recovery was attempted."""
    pass


def generate_batched_audio(batch: List[DialogueBatchRequest]) -> List[torch.Tensor]:
    """Generate audio for multiple dialogue requests in one forward pass."""
    texts = [req.formatted_script for req in batch]
    voice_samples_list = [req.voice_sample_list for req in batch]
    cfg_scale = batch[0].cfg_scale

    logger.info("Batched generation: %d requests", len(batch))

    inputs = None
    outputs = None
    try:
        t0 = time.perf_counter()
        inputs = processor(
            text=texts,
            voice_samples=voice_samples_list,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        t1 = time.perf_counter()
        logger.info("TIMING: processor() took %.2fs for %d items", t1 - t0, len(batch))

        t2 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )
        t3 = time.perf_counter()
        logger.info("TIMING: model.generate() took %.2fs for %d items", t3 - t2, len(batch))

        t4 = time.perf_counter()
        results = _extract_batch_waveforms(outputs, len(batch))
        t5 = time.perf_counter()
        logger.info("TIMING: _extract_batch_waveforms() took %.2fs for %d items", t5 - t4, len(batch))

        logger.info("TIMING: total generate_batched_audio() took %.2fs", t5 - t0)
        return results

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Handle both CUDA OOM and MPS/CPU memory errors
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            logger.error("Memory error during batched generation (batch_size=%d): %s", len(batch), e)
            _handle_memory_error()
            raise MemoryError(
                f"Out of memory with batch size {len(batch)}. Memory cleared, please retry with smaller batch."
            ) from e
        raise

    finally:
        # Always clean up tensors
        if inputs is not None:
            del inputs
        if outputs is not None:
            del outputs


async def dialogue_batch_processor():
    """Background task that batches /dialogue requests."""
    global dialogue_batch_queue

    while True:
        batch: List[DialogueBatchRequest] = []

        try:
            first = await dialogue_batch_queue.get()
            batch.append(first)

            deadline = asyncio.get_event_loop().time() + BATCH_TIMEOUT_SECONDS

            while len(batch) < MAX_BATCH_SIZE:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(
                        dialogue_batch_queue.get(),
                        timeout=remaining
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            logger.info("Processing dialogue batch: %d requests", len(batch))

            try:
                async with gen_lock:
                    loop = asyncio.get_event_loop()
                    audios = await loop.run_in_executor(None, generate_batched_audio, batch)

                for req, audio in zip(batch, audios):
                    req.future.set_result(audio)

            except MemoryError as e:
                logger.warning("Memory error in batch processor, cleared. Clients should retry.")
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

            except Exception as e:
                logger.exception("Batch generation error: %s", e)
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

        except Exception as e:
            logger.exception("Batch processor error: %s", e)
            await asyncio.sleep(0.1)


# -----------------------------------------------------------------------------
# FastAPI lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global dialogue_batch_queue, dialogue_batch_processor_task

    load_model()

    # Start HTTP batch processor
    dialogue_batch_queue = asyncio.Queue()
    dialogue_batch_processor_task = asyncio.create_task(dialogue_batch_processor())

    yield

    # Shutdown batch processor
    if dialogue_batch_processor_task:
        dialogue_batch_processor_task.cancel()
        try:
            await dialogue_batch_processor_task
        except asyncio.CancelledError:
            pass

    # Clean up model
    global model, processor, model_loaded
    if model is not None:
        del model
        model = None
    if processor is not None:
        del processor
        processor = None
    model_loaded = False

    # Clean up device memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("CUDA memory cleared")

    logger.info("Shutting down VibeVoice server")


app = FastAPI(
    title="VibeVoice Inference Server (Mac)",
    description="TTS inference server using VibeVoice-Large with voice cloning (Apple Silicon optimized)",
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Device info
    device_info = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    if device == "cuda":
        device_info["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1e9
        device_info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        device_info["gpu_name"] = torch.cuda.get_device_name(0)

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "voices_loaded": len(voice_samples),
        "model_path": MODEL_PATH,
        "sample_rate": OUTPUT_SAMPLE_RATE,
        **device_info,
    }


@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/voices", response_model=List[VoiceInfo])
async def list_voices():
    return [VoiceInfo(name=name, file_path=f"{name}.wav") for name in sorted(voice_samples.keys())]


@app.post("/reload-voices")
async def reload_voices():
    load_voice_samples()
    return {"status": "ok", "voices_loaded": len(voice_samples)}


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech and return WAV audio directly."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not voice_samples:
        raise HTTPException(status_code=503, detail="No voice samples loaded")
    if request.speaker not in voice_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{request.speaker}' not found. Available: {list(voice_samples.keys())}",
        )
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        start_time = time.time()

        formatted_script, _, voice_sample_list = _build_script_and_voices(
            [(request.text, request.speaker)], allow_voice_sharing=False
        )

        future = asyncio.get_event_loop().create_future()
        batch_req = DialogueBatchRequest(
            formatted_script=formatted_script,
            voice_sample_list=voice_sample_list,
            cfg_scale=request.cfg_scale,
            future=future,
            segment_count=1,
            speaker_count=1,
            allow_voice_sharing=False,
        )
        await dialogue_batch_queue.put(batch_req)

        audio = await future

        processing_time = time.time() - start_time
        duration_seconds = len(audio) / OUTPUT_SAMPLE_RATE

        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, audio.numpy(), OUTPUT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_bytes = buffer.getvalue()

        rtf = processing_time / duration_seconds if duration_seconds > 0 else 0.0

        logger.info(
            "Synthesized %d chars in %.2fs (dur=%.2fs, RTF=%.2f, speaker=%s)",
            len(request.text),
            processing_time,
            duration_seconds,
            rtf,
            request.speaker,
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Duration-Seconds": str(round(duration_seconds, 3)),
                "X-Processing-Time-Seconds": str(round(processing_time, 3)),
                "X-RTF": str(round(rtf, 3)),
                "X-Speaker": request.speaker,
                "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
                "X-Device": device,
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MemoryError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "1"}
        )
    except Exception as e:
        logger.exception("Synthesis error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dialogue")
async def dialogue(request: DialogueRequest):
    """Generate multi-speaker dialogue and return WAV audio directly."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not voice_samples:
        raise HTTPException(status_code=503, detail="No voice samples loaded")
    if not request.segments:
        raise HTTPException(status_code=400, detail="Segments cannot be empty")

    segments_for_gen: List[Tuple[str, str]] = []
    unique_speakers = []
    seen = set()

    for i, seg in enumerate(request.segments):
        spk = seg.speaker
        txt = (seg.text or "").strip()

        if spk not in voice_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{spk}' not found in segment {i}. Available: {list(voice_samples.keys())}",
            )

        if txt:
            segments_for_gen.append((txt, spk))
            if spk not in seen:
                seen.add(spk)
                unique_speakers.append(spk)

    if not segments_for_gen:
        raise HTTPException(status_code=400, detail="No valid (non-empty) segments to process")

    if len(unique_speakers) > 4 and not request.allow_voice_sharing:
        raise HTTPException(
            status_code=400,
            detail=f"VibeVoice supports up to 4 unique speakers per dialogue call; got {len(unique_speakers)}. "
                   f"Either reduce speakers or set allow_voice_sharing=true.",
        )

    try:
        start_time = time.time()

        formatted_script, ordered_speakers, voice_sample_list = _build_script_and_voices(
            segments_for_gen, allow_voice_sharing=request.allow_voice_sharing
        )

        future = asyncio.get_event_loop().create_future()
        batch_req = DialogueBatchRequest(
            formatted_script=formatted_script,
            voice_sample_list=voice_sample_list,
            cfg_scale=request.cfg_scale,
            future=future,
            segment_count=len(segments_for_gen),
            speaker_count=len(unique_speakers),
            allow_voice_sharing=request.allow_voice_sharing,
        )
        await dialogue_batch_queue.put(batch_req)

        audio = await future

        processing_time = time.time() - start_time
        duration_seconds = len(audio) / OUTPUT_SAMPLE_RATE
        rtf = processing_time / duration_seconds if duration_seconds > 0 else 0.0

        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, audio.numpy(), OUTPUT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_bytes = buffer.getvalue()

        logger.info(
            "Generated dialogue: %d segments, %d unique speakers in %.2fs (dur=%.2fs, RTF=%.2f)",
            len(segments_for_gen),
            len(unique_speakers),
            processing_time,
            duration_seconds,
            rtf,
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Duration-Seconds": str(round(duration_seconds, 3)),
                "X-Processing-Time-Seconds": str(round(processing_time, 3)),
                "X-RTF": str(round(rtf, 3)),
                "X-Segment-Count": str(len(segments_for_gen)),
                "X-Speaker-Count": str(len(unique_speakers)),
                "X-Speakers": ",".join(unique_speakers),
                "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
                "X-Device": device,
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MemoryError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "1"}
        )
    except Exception as e:
        logger.exception("Dialogue error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Streaming Endpoint
# -----------------------------------------------------------------------------
async def generate_streaming(
    text: str,
    speaker: str,
    cfg_scale: float,
) -> AsyncGenerator[bytes, None]:
    """Generate audio with real-time streaming.

    Yields raw 16-bit PCM audio chunks as they are decoded by the model.
    """
    formatted_script, _, voice_sample_list = _build_script_and_voices(
        [(text, speaker)], allow_voice_sharing=False
    )

    inputs = processor(
        text=[formatted_script],
        voice_samples=[voice_sample_list],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Create async streamer for single sample
    streamer = AsyncAudioStreamer(batch_size=1)

    # Create executor for blocking generation
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    memory_error = None

    def blocking_generate():
        """Run model.generate() in a thread."""
        nonlocal memory_error
        try:
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=streamer,
                    verbose=False,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.error("Memory error during streaming synthesis: %s", e)
                _handle_memory_error()
                memory_error = MemoryError("Out of memory during streaming. Memory cleared, please retry.")
        finally:
            # Signal completion
            streamer.end()

    # Start generation in background thread
    loop = asyncio.get_event_loop()
    gen_future = loop.run_in_executor(executor, blocking_generate)

    # Stream audio chunks as they arrive
    try:
        async for audio_chunk in streamer.get_stream(0):  # Single sample, index 0
            if audio_chunk is None:
                break
            # Normalize and convert to 16-bit PCM
            chunk = audio_chunk.detach().cpu().float()
            if chunk.dim() > 1:
                chunk = chunk.squeeze()
            # Clamp and convert to int16
            chunk = torch.clamp(chunk, -1.0, 1.0)
            audio_int16 = (chunk * 32767).to(torch.int16)
            yield audio_int16.numpy().tobytes()
    finally:
        # Ensure generation completes and cleanup
        await gen_future
        executor.shutdown(wait=False)
        if memory_error:
            raise memory_error


@app.post("/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest):
    """Synthesize speech with real-time audio streaming.

    Returns raw 16-bit PCM audio (audio/L16) as chunks are generated.
    Client should concatenate chunks and optionally wrap in WAV header.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if AsyncAudioStreamer is None:
        raise HTTPException(status_code=503, detail="Streaming not available - AsyncAudioStreamer not loaded")
    if not voice_samples:
        raise HTTPException(status_code=503, detail="No voice samples loaded")
    if request.speaker not in voice_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{request.speaker}' not found. Available: {list(voice_samples.keys())}",
        )
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    logger.info(
        "Streaming synthesis: %d chars, speaker=%s, cfg=%.2f",
        len(request.text),
        request.speaker,
        request.cfg_scale,
    )

    return StreamingResponse(
        generate_streaming(request.text, request.speaker, request.cfg_scale),
        media_type="audio/L16;rate=24000;channels=1",
        headers={
            "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
            "X-Speaker": request.speaker,
            "X-Streaming": "true",
            "X-Device": device,
        }
    )


async def generate_dialogue_streaming(
    segments: List[Tuple[str, str]],
    cfg_scale: float,
    allow_voice_sharing: bool,
) -> AsyncGenerator[bytes, None]:
    """Generate multi-speaker dialogue with real-time streaming.

    Yields raw 16-bit PCM audio chunks as they are decoded by the model.
    """
    formatted_script, _, voice_sample_list = _build_script_and_voices(
        segments, allow_voice_sharing=allow_voice_sharing
    )

    inputs = processor(
        text=[formatted_script],
        voice_samples=[voice_sample_list],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Create async streamer for single sample
    streamer = AsyncAudioStreamer(batch_size=1)

    # Create executor for blocking generation
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    memory_error = None

    def blocking_generate():
        """Run model.generate() in a thread."""
        nonlocal memory_error
        try:
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=streamer,
                    verbose=False,
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.error("Memory error during streaming dialogue: %s", e)
                _handle_memory_error()
                memory_error = MemoryError("Out of memory during streaming. Memory cleared, please retry.")
        finally:
            # Signal completion
            streamer.end()

    # Start generation in background thread
    loop = asyncio.get_event_loop()
    gen_future = loop.run_in_executor(executor, blocking_generate)

    # Stream audio chunks as they arrive
    try:
        async for audio_chunk in streamer.get_stream(0):  # Single sample, index 0
            if audio_chunk is None:
                break
            # Normalize and convert to 16-bit PCM
            chunk = audio_chunk.detach().cpu().float()
            if chunk.dim() > 1:
                chunk = chunk.squeeze()
            # Clamp and convert to int16
            chunk = torch.clamp(chunk, -1.0, 1.0)
            audio_int16 = (chunk * 32767).to(torch.int16)
            yield audio_int16.numpy().tobytes()
    finally:
        # Ensure generation completes and cleanup
        await gen_future
        executor.shutdown(wait=False)
        if memory_error:
            raise memory_error


@app.post("/dialogue/stream")
async def dialogue_stream(request: DialogueRequest):
    """Generate multi-speaker dialogue with real-time audio streaming.

    Returns raw 16-bit PCM audio (audio/L16) as chunks are generated.
    Client should concatenate chunks and optionally wrap in WAV header.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if AsyncAudioStreamer is None:
        raise HTTPException(status_code=503, detail="Streaming not available - AsyncAudioStreamer not loaded")
    if not voice_samples:
        raise HTTPException(status_code=503, detail="No voice samples loaded")
    if not request.segments:
        raise HTTPException(status_code=400, detail="Segments cannot be empty")

    segments_for_gen: List[Tuple[str, str]] = []
    unique_speakers = []
    seen = set()

    for i, seg in enumerate(request.segments):
        spk = seg.speaker
        txt = (seg.text or "").strip()

        if spk not in voice_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{spk}' not found in segment {i}. Available: {list(voice_samples.keys())}",
            )

        if txt:
            segments_for_gen.append((txt, spk))
            if spk not in seen:
                seen.add(spk)
                unique_speakers.append(spk)

    if not segments_for_gen:
        raise HTTPException(status_code=400, detail="No valid (non-empty) segments to process")

    if len(unique_speakers) > 4 and not request.allow_voice_sharing:
        raise HTTPException(
            status_code=400,
            detail=f"VibeVoice supports up to 4 unique speakers per dialogue call; got {len(unique_speakers)}. "
                   f"Either reduce speakers or set allow_voice_sharing=true.",
        )

    logger.info(
        "Streaming dialogue: %d segments, %d speakers, cfg=%.2f",
        len(segments_for_gen),
        len(unique_speakers),
        request.cfg_scale,
    )

    return StreamingResponse(
        generate_dialogue_streaming(segments_for_gen, request.cfg_scale, request.allow_voice_sharing),
        media_type="audio/L16;rate=24000;channels=1",
        headers={
            "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
            "X-Segment-Count": str(len(segments_for_gen)),
            "X-Speaker-Count": str(len(unique_speakers)),
            "X-Speakers": ",".join(unique_speakers),
            "X-Streaming": "true",
            "X-Device": device,
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

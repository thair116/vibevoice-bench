# VibeVoice Mac Benchmark

A one-command benchmark for the [VibeVoice-Large](https://huggingface.co/rsxdalv/VibeVoice-Large) text-to-speech model on Apple Silicon. You run it, it produces a JSON results file, and you email that file back.

## What you'll need

- **Apple Silicon Mac** (M1, M2, M3, or M4 — any model)
- **15 GB of free disk space** (mostly for the one-time model download)
- **Internet connection** (the first run downloads ~10 GB)
- **Python 3.10 or newer** — check by running `python3 --version` in Terminal. If you don't have it, install from [python.org/downloads](https://www.python.org/downloads/).
- **About 30 minutes** the first time (most of it is the model download). Subsequent runs take ~5–10 minutes.

## How to run it

Open Terminal and run these three lines:

```bash
git clone https://github.com/<owner>/vibevoice-mac-bench.git
cd vibevoice-mac-bench
./run.sh
```

That's it. The script will:

1. Check your Mac is supported.
2. Set up a Python environment.
3. Download the model (first time only, ~10 GB).
4. Start the TTS server.
5. Run the benchmark.
6. Save a results file like `results_<your-mac-name>_<timestamp>.json`.

When it's done, **email the `results_*.json` file to travis@gethero.com**. That's all I need.

## What if something goes wrong?

**"command not found: python3"**
You don't have Python installed. Install it from [python.org/downloads](https://www.python.org/downloads/) and try again.

**"Port 8888 is already in use"**
Something else on your Mac is using port 8888. Either close that program, or run with a different port:
```bash
PORT=9000 ./run.sh
```

**Stuck on "Waiting for server to become healthy" for a long time**
On the first run, the model download can take 10–15 minutes (longer on slow connections). If it's been more than 20 minutes, open another Terminal window and check progress:
```bash
tail -f ~/Code/vibevoice-mac-bench/server.log
```
You should see HuggingFace download lines. If it's truly stuck, Ctrl-C and try again.

**Anything else**
Send me whatever error you see along with `server.log` and I'll figure it out: travis@gethero.com.

## What the script actually does (for the curious)

- Spins up a [FastAPI](https://fastapi.tiangolo.com/) server in `server/main.py` that wraps VibeVoice and exposes a `/dialogue` endpoint.
- Runs `benchmark/benchmark.py` against it with batch sizes 1, 2, 4 and three runs each, generating ~60 seconds of multi-speaker audio per request.
- Captures Real-Time Factor (RTF), throughput, wall-clock time, and your Mac's specs into a single JSON file.
- Cleans up the server process when finished (or if you Ctrl-C mid-run).

Voice samples in `voices/` are short reference clips that VibeVoice uses for speaker conditioning. The benchmark dialogue itself is in `benchmark/sample_dialogue.json`.

## Re-running

You can run `./run.sh` as many times as you want. After the first run the model is cached in `model/`, so subsequent runs skip the download and finish in ~5–10 minutes. Each run produces a separately-timestamped results file.

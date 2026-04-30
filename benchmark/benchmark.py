#!/usr/bin/env python3
"""
Vibevoice Benchmarking Harness

Tests server performance with concurrent requests simulating different batch sizes.
Collects extended metrics including GPU utilization and VRAM usage.

Usage:
    python benchmark.py --batch-sizes 1,2,4,6,8 --runs 3
    python benchmark.py --server http://192.168.1.100:8080 --output results.json

    # Test with fewer diffusion steps (faster, lower quality):
    python benchmark.py --diffusion-steps 8 --batch-sizes 1,2,4
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip install aiohttp")
    sys.exit(1)


@dataclass
class GPUMetrics:
    utilization: float  # GPU utilization %
    memory_used: float  # VRAM used in MB
    memory_total: float  # Total VRAM in MB


@dataclass
class RequestResult:
    success: bool
    duration_seconds: float
    processing_time_seconds: float
    rtf: float
    segment_count: int
    speaker_count: int
    error: Optional[str] = None


@dataclass
class BatchResult:
    batch_size: int
    run_number: int
    wall_time_seconds: float
    requests: list[RequestResult]
    gpu_util_peak: float
    vram_mb_peak: float
    throughput: float  # Audio seconds generated per wall-clock second


def get_gpu_metrics() -> Optional[GPUMetrics]:
    """Query nvidia-smi for GPU metrics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]  # First GPU
        parts = [float(x.strip()) for x in line.split(",")]
        return GPUMetrics(
            utilization=parts[0], memory_used=parts[1], memory_total=parts[2]
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        return None


async def monitor_gpu_during_request(
    stop_event: asyncio.Event, metrics_list: list[GPUMetrics]
):
    """Background task to sample GPU metrics during request processing."""
    while not stop_event.is_set():
        metrics = get_gpu_metrics()
        if metrics:
            metrics_list.append(metrics)
        await asyncio.sleep(0.5)  # Sample every 500ms


async def heartbeat_during_request(stop_event: asyncio.Event, interval: int = 10):
    """Print elapsed-time dots so the user knows we're still alive."""
    elapsed = 0
    try:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                elapsed += interval
                print(f" [{elapsed}s]", end="", flush=True)
    except asyncio.CancelledError:
        return


async def send_dialogue_request(
    session: aiohttp.ClientSession, url: str, dialogue: dict
) -> RequestResult:
    """Send a single dialogue request and parse response headers."""
    try:
        async with session.post(
            f"{url}/dialogue",
            json=dialogue,
            timeout=aiohttp.ClientTimeout(total=1800),  # 30 min timeout for long audio
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return RequestResult(
                    success=False,
                    duration_seconds=0,
                    processing_time_seconds=0,
                    rtf=0,
                    segment_count=0,
                    speaker_count=0,
                    error=f"HTTP {response.status}: {error_text[:200]}",
                )

            # Read the audio data (discard it, we just need headers)
            await response.read()

            return RequestResult(
                success=True,
                duration_seconds=float(
                    response.headers.get("X-Duration-Seconds", 0)
                ),
                processing_time_seconds=float(
                    response.headers.get("X-Processing-Time-Seconds", 0)
                ),
                rtf=float(response.headers.get("X-RTF", 0)),
                segment_count=int(response.headers.get("X-Segment-Count", 0)),
                speaker_count=int(response.headers.get("X-Speaker-Count", 0)),
            )
    except asyncio.TimeoutError:
        return RequestResult(
            success=False,
            duration_seconds=0,
            processing_time_seconds=0,
            rtf=0,
            segment_count=0,
            speaker_count=0,
            error="Request timed out",
        )
    except Exception as e:
        return RequestResult(
            success=False,
            duration_seconds=0,
            processing_time_seconds=0,
            rtf=0,
            segment_count=0,
            speaker_count=0,
            error=str(e),
        )


async def run_batch(
    session: aiohttp.ClientSession,
    url: str,
    dialogue: dict,
    batch_size: int,
    run_number: int,
) -> BatchResult:
    """Run a batch of concurrent requests and collect metrics."""
    # Start GPU monitoring + heartbeat
    gpu_samples: list[GPUMetrics] = []
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(monitor_gpu_during_request(stop_event, gpu_samples))
    heartbeat_task = asyncio.create_task(heartbeat_during_request(stop_event))

    # Send concurrent requests
    start_time = time.time()
    tasks = [send_dialogue_request(session, url, dialogue) for _ in range(batch_size)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    # Stop monitors
    stop_event.set()
    await monitor_task
    await heartbeat_task

    wall_time = end_time - start_time

    # Calculate peak GPU metrics
    gpu_util_peak = max((s.utilization for s in gpu_samples), default=0)
    vram_mb_peak = max((s.memory_used for s in gpu_samples), default=0)

    # Calculate throughput (total audio seconds / wall clock time)
    total_audio_seconds = sum(r.duration_seconds for r in results if r.success)
    throughput = total_audio_seconds / wall_time if wall_time > 0 else 0

    return BatchResult(
        batch_size=batch_size,
        run_number=run_number,
        wall_time_seconds=wall_time,
        requests=list(results),
        gpu_util_peak=gpu_util_peak,
        vram_mb_peak=vram_mb_peak,
        throughput=throughput,
    )


def print_results_table(all_results: dict[int, list[BatchResult]], dialogue_info: str):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("Vibevoice Benchmark Results")
    print("=" * 80)
    print(f"Script: {dialogue_info}")
    print()

    # Header
    print(
        f"{'Batch':<7} {'Runs':<5} {'Avg RTF':<9} {'Avg Wall':<10} "
        f"{'Throughput':<12} {'GPU %':<8} {'VRAM MB':<10} {'Success':<8}"
    )
    print("-" * 80)

    for batch_size in sorted(all_results.keys()):
        runs = all_results[batch_size]
        successful_runs = [r for r in runs if all(req.success for req in r.requests)]

        if not successful_runs:
            print(f"{batch_size:<7} {len(runs):<5} {'FAILED':<9}")
            continue

        avg_rtf = sum(
            sum(req.rtf for req in r.requests) / len(r.requests) for r in successful_runs
        ) / len(successful_runs)
        avg_wall = sum(r.wall_time_seconds for r in successful_runs) / len(successful_runs)
        avg_throughput = sum(r.throughput for r in successful_runs) / len(successful_runs)
        avg_gpu = sum(r.gpu_util_peak for r in successful_runs) / len(successful_runs)
        avg_vram = sum(r.vram_mb_peak for r in successful_runs) / len(successful_runs)
        success_rate = len(successful_runs) / len(runs) * 100

        print(
            f"{batch_size:<7} {len(runs):<5} {avg_rtf:<9.3f} {avg_wall:<10.1f}s "
            f"{avg_throughput:<12.2f}x {avg_gpu:<8.1f} {avg_vram:<10.0f} {success_rate:<8.0f}%"
        )

    print("=" * 80)


async def check_server_health(session: aiohttp.ClientSession, url: str) -> bool:
    """Check if the server is healthy."""
    try:
        async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
            return response.status == 200
    except Exception:
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vibevoice server with concurrent requests"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8080",
        help="Server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,6,8",
        help="Comma-separated batch sizes to test (default: 1,2,4,6,8)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per batch size (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--dialogue",
        type=str,
        help="Path to custom dialogue JSON file",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        help="Max target duration in seconds (limits segments used)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help="Number of diffusion steps (default: model default ~20). Try 6-8 for low latency.",
    )
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Load dialogue
    script_dir = Path(__file__).parent
    dialogue_path = Path(args.dialogue) if args.dialogue else script_dir / "sample_dialogue.json"

    if not dialogue_path.exists():
        print(f"Error: Dialogue file not found: {dialogue_path}")
        sys.exit(1)

    with open(dialogue_path) as f:
        dialogue = json.load(f)

    # Limit segments if max duration specified
    # Estimate ~11 seconds per segment based on full dialogue being ~10 min / 54 segments
    if args.max_duration:
        segments = dialogue.get("segments", [])
        estimated_seconds_per_segment = 11
        max_segments = max(1, args.max_duration // estimated_seconds_per_segment)
        if max_segments < len(segments):
            dialogue["segments"] = segments[:max_segments]

    # Add diffusion_steps if specified
    if args.diffusion_steps is not None:
        dialogue["diffusion_steps"] = args.diffusion_steps

    segment_count = len(dialogue.get("segments", []))
    estimated_duration = segment_count * 11
    if estimated_duration >= 60:
        duration_str = f"~{estimated_duration // 60} min audio"
    else:
        duration_str = f"~{estimated_duration}s audio"
    steps_str = f", {args.diffusion_steps} steps" if args.diffusion_steps else ""
    dialogue_info = f"{segment_count} segments, {duration_str}{steps_str}"

    print(f"Vibevoice Benchmark")
    print(f"Server: {args.server}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Runs per batch: {args.runs}")
    print(f"Diffusion steps: {args.diffusion_steps if args.diffusion_steps else 'default'}")
    print(f"Dialogue: {dialogue_path.name} ({dialogue_info})")
    print()

    async with aiohttp.ClientSession() as session:
        # Health check
        print("Checking server health...", end=" ", flush=True)
        if not await check_server_health(session, args.server):
            print("FAILED")
            print(f"Error: Server at {args.server} is not responding")
            sys.exit(1)
        print("OK")

        all_results: dict[int, list[BatchResult]] = {}

        for batch_size in batch_sizes:
            print(f"\nTesting batch size {batch_size}...")
            all_results[batch_size] = []

            for run in range(1, args.runs + 1):
                print(f"  Run {run}/{args.runs}...", end=" ", flush=True)
                result = await run_batch(session, args.server, dialogue, batch_size, run)
                all_results[batch_size].append(result)

                success_count = sum(1 for r in result.requests if r.success)
                print(
                    f"done ({success_count}/{batch_size} success, "
                    f"{result.wall_time_seconds:.1f}s wall time, "
                    f"{result.throughput:.2f}x throughput)"
                )

                # Brief pause between runs to let GPU cool
                if run < args.runs:
                    await asyncio.sleep(2)

    # Print results table
    print_results_table(all_results, dialogue_info)

    # Save JSON output if requested
    if args.output:
        output_data = {
            "server": args.server,
            "batch_sizes": batch_sizes,
            "runs_per_batch": args.runs,
            "diffusion_steps": args.diffusion_steps,
            "dialogue_file": str(dialogue_path),
            "results": {
                batch_size: [asdict(r) for r in runs]
                for batch_size, runs in all_results.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

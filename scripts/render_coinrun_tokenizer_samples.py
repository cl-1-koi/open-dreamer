#!/usr/bin/env python3
"""Render held-out CoinRun tokenizer reconstructions as a static gallery."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from coinrun_preflight import (
    compute_tokenizer_probe_metrics,
    iter_array_records,
    select_tokenizer_probe_records,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _decode_clip(record: dict[str, Any], sequence_length: int) -> np.ndarray:
    available = int(record["sequence_length"])
    frame_shape = tuple(
        int(value) for value in record.get("frame_shape", (64, 64, 3))
    )
    return np.frombuffer(record["raw_video"], dtype=np.uint8).reshape(
        available, *frame_shape
    )[:sequence_length]


def _uint8(frames: Any) -> np.ndarray:
    return np.clip(np.asarray(frames), 0, 255).astype(np.uint8)


def _error_heatmap(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    error = np.mean(
        np.abs(target.astype(np.float32) - prediction.astype(np.float32)),
        axis=-1,
    )
    intensity = np.clip(error * 8.0, 0, 255).astype(np.uint8)
    return np.stack(
        (
            intensity,
            (intensity.astype(np.float32) * 0.35).astype(np.uint8),
            np.zeros_like(intensity),
        ),
        axis=-1,
    )


def _model_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    delta = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(delta * delta))
    mae = float(np.mean(np.abs(delta)))
    psnr = 120.0 if mse == 0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)
    return {"pixel_mse": mse, "pixel_mae": mae, "psnr_db": psnr}


def _contact_sheet(
    target: np.ndarray,
    online: np.ndarray,
    ema: np.ndarray,
    heatmap: np.ndarray,
) -> np.ndarray:
    frame_indices = np.linspace(0, len(target) - 1, num=5, dtype=int)
    return np.concatenate(
        [
            np.concatenate(
                (target[index], online[index], ema[index], heatmap[index]),
                axis=1,
            )
            for index in frame_indices
        ],
        axis=0,
    )


def _render_html(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    cards = []
    for sample in samples:
        cards.append(
            f"""
            <article class="sample">
              <header>
                <h2>Sample {sample["index"]:02d}</h2>
                <dl>
                  <div><dt>Collector</dt><dd>{html.escape(sample["collector"])}</dd></div>
                  <div><dt>Level seed</dt><dd>{sample["level_seed"]}</dd></div>
                  <div><dt>Online PSNR</dt><dd>{sample["online"]["psnr_db"]:.2f} dB</dd></div>
                  <div><dt>EMA PSNR</dt><dd>{sample["ema"]["psnr_db"]:.2f} dB</dd></div>
                </dl>
              </header>
              <div class="column-labels" aria-hidden="true">
                <span>Original</span><span>Online</span><span>EMA</span><span>EMA error ×8</span>
              </div>
              <video controls preload="metadata" playsinline>
                <source src="{html.escape(sample["video"])}" type="video/mp4">
              </video>
              <details>
                <summary>Five-frame contact sheet</summary>
                <img src="{html.escape(sample["contact_sheet"])}"
                     alt="Original, online, EMA, and error frames at five times">
              </details>
            </article>
            """
        )

    reconstruction = summary["metrics"]["ema"]["reconstruction"]
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CoinRun tokenizer checkpoint 6000</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: #111315;
      color: #edf0f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #111315; }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 24px auto 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    .subtitle {{ margin: 0 0 22px; color: #aeb7bd; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 1px;
      margin-bottom: 24px;
      background: #343a3f;
      border: 1px solid #343a3f;
      border-radius: 6px;
      overflow: hidden;
    }}
    .summary div {{ padding: 14px; background: #1b1f22; }}
    .summary strong {{ display: block; margin-top: 4px; font-size: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 16px; }}
    .sample {{ border: 1px solid #343a3f; border-radius: 6px; overflow: hidden; background: #181b1e; }}
    .sample header {{ padding: 14px 16px 12px; }}
    h2 {{ font-size: 17px; margin: 0 0 10px; }}
    dl {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 0; }}
    dt {{ color: #8e9aa1; font-size: 11px; text-transform: uppercase; }}
    dd {{ margin: 3px 0 0; font-variant-numeric: tabular-nums; }}
    .column-labels {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      padding: 7px 0;
      background: #24292d;
      color: #d9dee1;
      font-size: 12px;
      text-align: center;
    }}
    video {{ display: block; width: 100%; aspect-ratio: 4 / 1; background: #000; image-rendering: pixelated; }}
    details {{ padding: 10px 16px 14px; color: #bdc5ca; }}
    summary {{ cursor: pointer; }}
    img {{ display: block; width: min(100%, 512px); margin: 12px auto 0; image-rendering: pixelated; }}
    @media (max-width: 560px) {{
      main {{ width: min(100% - 16px, 1180px); margin-top: 14px; }}
      .grid {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: repeat(2, 1fr); }}
      .column-labels {{ font-size: 10px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>CoinRun Tokenizer, Checkpoint 6000</h1>
  <p class="subtitle">Held-out validation clips. These are reconstructions, not dynamics rollouts.</p>
  <section class="summary">
    <div>Clips<strong>{summary["record_count"]}</strong></div>
    <div>Frames<strong>{summary["frame_count"]}</strong></div>
    <div>EMA PSNR<strong>{reconstruction["model_psnr_db"]:.2f} dB</strong></div>
    <div>Copy-frame baseline<strong>{reconstruction["copy_previous_baseline_psnr_db"]:.2f} dB</strong></div>
  </section>
  <section class="grid">
    {"".join(cards)}
  </section>
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    import jax
    import jax.numpy as jnp
    from flax import nnx

    from dreamer.checkpointing import TokenizerCheckpointBundle
    from dreamer.parallel import build_parallel

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, skipped, seen, selected_counts = select_tokenizer_probe_records(
        iter_array_records(args.dataset_dir.resolve()),
        sequence_length=args.sequence_length,
        max_records=args.max_records,
    )
    if not selected:
        raise RuntimeError("No held-out validation records were long enough")

    videos = np.stack(
        [_decode_clip(record, args.sequence_length) for record in selected]
    )
    mesh, _, mesh_rules = build_parallel("data")
    with jax.set_mesh(mesh):
        bundle = TokenizerCheckpointBundle.from_pretrained(
            str(args.checkpoint_dir.resolve()),
            mesh_rules=mesh_rules,
            model_names={"tokenizer", "tokenizer_ema"},
        )

        @nnx.jit
        def reconstruct(model, batch):
            latents, _, _ = model.encode(batch, deterministic=True)
            decoded, _ = model.decode(latents, deterministic=True)
            return latents, decoded

        online_batches = []
        online_latent_batches = []
        ema_batches = []
        ema_latent_batches = []
        for start in range(0, len(videos), args.batch_size):
            batch = jnp.asarray(videos[start : start + args.batch_size])
            online_latent_batch, online_batch = reconstruct(bundle.tokenizer, batch)
            ema_latent_batch, ema_batch = reconstruct(bundle.tokenizer_ema, batch)
            (
                online_latent_batch,
                online_batch,
                ema_latent_batch,
                ema_batch,
            ) = jax.device_get(
                (
                    online_latent_batch,
                    online_batch,
                    ema_latent_batch,
                    ema_batch,
                )
            )
            online_latent_batches.append(online_latent_batch)
            online_batches.append(online_batch)
            ema_latent_batches.append(ema_latent_batch)
            ema_batches.append(ema_batch)

    online_latents = np.concatenate(online_latent_batches)
    online_recon_raw = np.concatenate(online_batches)
    ema_latents = np.concatenate(ema_latent_batches)
    ema_recon_raw = np.concatenate(ema_batches)
    online_recon = _uint8(online_recon_raw)
    ema_recon = _uint8(ema_recon_raw)
    sample_payloads = []
    for index, (record, target, online, ema) in enumerate(
        zip(selected, videos, online_recon, ema_recon, strict=True),
        start=1,
    ):
        collector = str(
            record.get("collector", record.get("collector_policy", "unknown"))
        )
        heatmap = _error_heatmap(target, ema)
        combined = np.concatenate((target, online, ema, heatmap), axis=2)
        stem = f"sample_{index:02d}_{_slug(collector)}"
        video_name = f"{stem}.mp4"
        sheet_name = f"{stem}_contact.png"
        iio.imwrite(
            output_dir / video_name,
            combined,
            fps=args.fps,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=["-crf", "16", "-movflags", "+faststart"],
        )
        iio.imwrite(
            output_dir / sheet_name,
            _contact_sheet(target, online, ema, heatmap),
        )
        sample_payloads.append(
            {
                "index": index,
                "collector": collector,
                "level_seed": record.get("level_seed", "unknown"),
                "video": video_name,
                "contact_sheet": sheet_name,
                "online": _model_metrics(online, target),
                "ema": _model_metrics(ema, target),
            }
        )

    summary = {
        "schema_version": 1,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint_step": max(
            int(path.name)
            for path in args.checkpoint_dir.resolve().iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "sequence_length": args.sequence_length,
        "record_count": len(selected),
        "frame_count": int(videos.shape[0] * videos.shape[1]),
        "skipped_short_records": skipped,
        "collector_records_seen": seen,
        "collector_records_selected": selected_counts,
        "metrics": {
            "online": compute_tokenizer_probe_metrics(
                latents=online_latents,
                reconstructions=online_recon_raw,
                targets=videos,
                dataset_mean=tuple(bundle.tokenizer.cfg.encoder.dataset_mean),
            ),
            "ema": compute_tokenizer_probe_metrics(
                latents=ema_latents,
                reconstructions=ema_recon_raw,
                targets=videos,
                dataset_mean=tuple(bundle.tokenizer_ema.cfg.encoder.dataset_mean),
            ),
        },
        "samples": sample_payloads,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _render_html(output_dir=output_dir, samples=sample_payloads, summary=summary)
    print(
        f"Wrote {len(selected)} held-out samples to {output_dir}; "
        f"EMA PSNR={summary['metrics']['ema']['reconstruction']['model_psnr_db']:.2f} dB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

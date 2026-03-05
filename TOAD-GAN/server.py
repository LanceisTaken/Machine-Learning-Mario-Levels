"""TOAD-GAN Flask API server.

Wraps the TOAD-GAN inference pipeline in a tiny HTTP server so that Unity
(or any other client) can request freshly generated Mario levels at runtime.

Endpoints
---------
GET /generate
    Generate one level and return its JSON payload.
    Query parameters (all optional):
        temperature : float  – noise temperature (default 1.0, >1 = more variety)
        scale_w     : float  – width multiplier  (default 1.0)
        scale_h     : float  – height multiplier (default 1.0)

GET /health
    Returns {"status": "ok"} – useful for checking if the server is alive.

Usage
-----
    python server.py --model_dir output/models --vocab vocab.json

Then in Unity hit Play; ApiClient.cs will call http://localhost:5000/generate.

Requirements
------------
    pip install flask
    (torch, etc. assumed already installed from requirements.txt)
"""

import argparse
import logging
import sys
import os

# ── Make sure generate.py / models.py / level_utils.py are importable ────────
# server.py lives in the TOAD-GAN directory alongside these modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request as flask_request

from generate import (
    load_checkpoint,
    generate_tensor,
    tensor_to_tile_ids,
    fix_pipes,
    fix_lucky_blocks,
    build_unity_json,
)
from level_utils import load_vocab

# ── Globals (populated after argument parsing) ────────────────────────────────

app = Flask(__name__)

_generators     = None
_noise_amps     = None
_pyramid_shapes = None
_stoi           = None
_itos           = None
_device         = "cpu"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/generate")
def generate():
    """Generate one Mario level and return a JSON payload ready for Unity."""
    # --- Parse optional query parameters ---
    try:
        temperature = float(flask_request.args.get("temperature", 1.0))
        scale_w     = float(flask_request.args.get("scale_w",     1.0))
        scale_h     = float(flask_request.args.get("scale_h",     1.0))
    except ValueError as exc:
        return jsonify({"error": f"Invalid query parameter: {exc}"}), 400

    # Clamp to sane ranges
    temperature = max(0.1, min(temperature, 5.0))
    scale_w     = max(0.5, min(scale_w,     4.0))
    scale_h     = max(0.5, min(scale_h,     4.0))

    app.logger.info(
        "Generating level: temp=%.2f scale_w=%.2f scale_h=%.2f",
        temperature, scale_w, scale_h,
    )

    # --- Run the TOAD-GAN pipeline ---
    try:
        output_tensor = generate_tensor(
            _generators, _noise_amps, _pyramid_shapes, _device,
            temperature=temperature,
            scale_h=scale_h,
            scale_w=scale_w,
        )

        tile_ids = tensor_to_tile_ids(output_tensor)
        tile_ids = fix_pipes(tile_ids, _stoi)
        tile_ids = fix_lucky_blocks(tile_ids, _stoi)

        payload = build_unity_json(tile_ids, _itos, _stoi)

    except Exception as exc:
        app.logger.exception("Generation failed")
        return jsonify({"error": str(exc)}), 500

    app.logger.info(
        "Level generated: %dh x %dw",
        payload["height"], payload["width"],
    )

    # --- Return JSON ---
    # tile_map is a plain {"0": "-", "1": "#", ...} object.
    # Newtonsoft.Json on the Unity side deserializes this into a
    # Dictionary<string, string> without any extra conversion needed.
    return jsonify({
        "height":   payload["height"],
        "width":    payload["width"],
        "tile_ids": payload["tile_ids"],
        "tile_map": payload["tile_map"],
    })


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TOAD-GAN Flask API server")
    p.add_argument("--model_dir",   required=True,
                   help="Directory containing toadgan_checkpoint.pt")
    p.add_argument("--vocab",       required=True,
                   help="Path to vocab.json")
    p.add_argument("--checkpoint",  default="toadgan_checkpoint.pt",
                   help="Checkpoint filename inside model_dir")
    p.add_argument("--host",        default="127.0.0.1",
                   help="Interface to listen on (use 0.0.0.0 for LAN access)")
    p.add_argument("--port",        type=int, default=5000,
                   help="Port to listen on")
    p.add_argument("--device",      default="",
                   help="'cuda' or 'cpu'. Defaults to CUDA if available.")
    return p.parse_args()


def main():
    global _generators, _noise_amps, _pyramid_shapes, _stoi, _itos, _device

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Pick device ──────────────────────────────────────────────────────
    import torch
    if args.device:
        _device = args.device
    else:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    app.logger.info("Using device: %s", _device)

    # ── Load vocab & model ───────────────────────────────────────────────
    app.logger.info("Loading vocab from %s …", args.vocab)
    _stoi, _itos = load_vocab(args.vocab)

    app.logger.info("Loading checkpoint from %s/%s …", args.model_dir, args.checkpoint)
    _generators, _noise_amps, _pyramid_shapes, _ = load_checkpoint(
        args.model_dir, _device, checkpoint_name=args.checkpoint
    )
    app.logger.info("Model loaded (%d scale(s)).", len(_generators))

    # ── Start server ─────────────────────────────────────────────────────
    app.logger.info(
        "Server ready. Listening on http://%s:%d", args.host, args.port
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

"""Export the trained TOAD-GAN multi-scale pipeline to a single ONNX file.

The wrapper ``TOADGANPipeline`` bakes all generators, noise amplitudes, and
inter-scale upsampling into one traceable ``nn.Module`` so that Unity (via
Sentis / Barracuda) can run the full generation pipeline natively.

Usage
-----
    python export_onnx.py --model_dir output --vocab vocab.json

Outputs
-------
- ``toadgan.onnx`` in the script directory (and copied to Unity StreamingAssets)
- Verification pass comparing PyTorch vs ONNX Runtime outputs
"""

import argparse
import json
import os
import shutil
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import GeneratorBlock
from generate import load_checkpoint


# ── Wrapper module ──────────────────────────────────────────────────────────

class TOADGANPipeline(nn.Module):
    """Wraps the full multi-scale TOAD-GAN generation pipeline into a single
    ``nn.Module`` suitable for ``torch.onnx.export``.

    The module takes one noise tensor per scale and returns a softmax
    probability map over token channels at the finest scale.

    Inputs  (during forward):
        noise_0 : (1, C, H0, W0)  — noise for the coarsest scale
        noise_1 : (1, C, H1, W1)  — noise for scale 1
        noise_2 : (1, C, H2, W2)  — noise for scale 2
        ...and so on for however many scales exist.

    Output:
        (1, C, H_final, W_final)  — softmax probabilities (argmax → tile IDs)
    """

    def __init__(
        self,
        generators: List[GeneratorBlock],
        noise_amps: List[float],
        pyramid_shapes: List[Tuple[int, ...]],
    ) -> None:
        super().__init__()
        # Store generators as a ModuleList so their parameters are registered
        self.generators = nn.ModuleList(generators)

        # Register noise_amps and target spatial sizes as buffers so they
        # travel with the model but are not trainable parameters.
        self.register_buffer(
            "noise_amps",
            torch.tensor(noise_amps, dtype=torch.float32),
        )

        # Store (H, W) for each scale so forward() knows the target sizes
        # Shape: (num_scales, 2)
        hw = torch.tensor(
            [[s[2], s[3]] for s in pyramid_shapes], dtype=torch.long,
        )
        self.register_buffer("scale_hw", hw)

    def forward(self, *noises: torch.Tensor) -> torch.Tensor:
        """Run the multi-scale generation pipeline.

        Args:
            *noises: One noise tensor per scale, each ``(1, C, H_s, W_s)``.
                     The tensors should already be sampled from N(0,1);
                     noise-amp scaling is applied internally.

        Returns:
            Softmax probability map ``(1, C, H_final, W_final)``.
        """
        prev_output: torch.Tensor | None = None

        for s, gen in enumerate(self.generators):
            noise = noises[s] * self.noise_amps[s]

            if prev_output is not None:
                h = int(self.scale_hw[s, 0].item())
                w = int(self.scale_hw[s, 1].item())
                prev_up = F.interpolate(
                    prev_output, size=(h, w),
                    mode="bilinear", align_corners=False,
                )
            else:
                prev_up = None

            prev_output = F.softmax(gen(noise, prev_up), dim=1)

        return prev_output  # type: ignore[return-value]


# ── Export logic ────────────────────────────────────────────────────────────

def export(
    model_dir: str,
    vocab_path: str,
    checkpoint_name: str = "toadgan_checkpoint.pt",
    opset: int = 17,
    unity_assets: str | None = None,
) -> str:
    """Export the TOAD-GAN checkpoint to ONNX and optionally copy to Unity.

    Returns the path to the exported ``.onnx`` file.
    """
    device = "cpu"  # export always on CPU for maximum compatibility

    # ── Load checkpoint ──────────────────────────────────────────────
    generators, noise_amps, pyramid_shapes, num_tokens = load_checkpoint(
        model_dir, device, checkpoint_name=checkpoint_name,
    )
    num_scales = len(generators)

    print(f"Loaded {num_scales} scale(s), {num_tokens} tokens")
    for s, shape in enumerate(pyramid_shapes):
        print(f"  scale {s}: shape {shape}  noise_amp={noise_amps[s]:.6f}")
    print()

    # ── Build wrapper ────────────────────────────────────────────────
    pipeline = TOADGANPipeline(generators, noise_amps, pyramid_shapes)
    pipeline.eval()

    # ── Dummy inputs (one per scale) ─────────────────────────────────
    dummy_noises = tuple(
        torch.randn(pyramid_shapes[s], device=device)
        for s in range(num_scales)
    )

    input_names = [f"noise_{s}" for s in range(num_scales)]
    output_names = ["token_probs"]

    # ── Export ────────────────────────────────────────────────────────
    onnx_path = os.path.join(os.path.dirname(__file__) or ".", "toadgan.onnx")

    print(f"Exporting to {onnx_path}  (opset {opset}) ...")
    torch.onnx.export(
        pipeline,
        dummy_noises,
        onnx_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
    )
    onnx_size = os.path.getsize(onnx_path)
    print(f"[OK] Exported  {onnx_path}  ({onnx_size / 1024:.1f} KB)")
    print()

    # ── Verification with ONNX Runtime ───────────────────────────────
    try:
        import onnxruntime as ort
        import numpy as np

        print("Verifying ONNX output against PyTorch ...")

        # PyTorch reference
        with torch.inference_mode():
            pt_out = pipeline(*dummy_noises).numpy()

        # ONNX Runtime
        sess = ort.InferenceSession(onnx_path)
        ort_inputs = {
            f"noise_{s}": dummy_noises[s].numpy() for s in range(num_scales)
        }
        ort_out = sess.run(["token_probs"], ort_inputs)[0]

        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        close = np.allclose(pt_out, ort_out, atol=1e-5)

        print(f"  PyTorch output shape : {pt_out.shape}")
        print(f"  ONNX    output shape : {ort_out.shape}")
        print(f"  Max absolute diff    : {max_diff:.2e}")
        if close:
            print(f"  [PASS] ONNX output matches PyTorch (atol=1e-5)")
        else:
            print(f"  [FAIL] WARNING: outputs differ beyond tolerance!")
        print()

    except ImportError:
        print("  [WARN] onnxruntime not installed -- skipping verification.")
        print("    Install with: pip install onnxruntime")
        print()

    # ── Copy to Unity StreamingAssets ─────────────────────────────────
    if unity_assets is None:
        # Default: relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        unity_assets = os.path.normpath(os.path.join(
            script_dir, "..", "TOAD-GAN Mario", "Assets", "StreamingAssets",
        ))

    toadgan_dir = os.path.join(unity_assets, "TOADGAN")
    os.makedirs(toadgan_dir, exist_ok=True)

    dest_onnx = os.path.join(toadgan_dir, "toadgan.onnx")
    shutil.copy2(onnx_path, dest_onnx)
    print(f"Copied ONNX  -> {dest_onnx}")

    dest_vocab = os.path.join(toadgan_dir, "vocab.json")
    shutil.copy2(vocab_path, dest_vocab)
    print(f"Copied vocab -> {dest_vocab}")

    # Also save a small metadata JSON with shapes/amps for Unity to read
    meta = {
        "num_scales": num_scales,
        "num_tokens": num_tokens,
        "noise_amps": noise_amps,
        "pyramid_shapes": [list(s) for s in pyramid_shapes],
        "input_names": input_names,
        "output_name": "token_probs",
    }
    meta_path = os.path.join(toadgan_dir, "toadgan_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta   -> {meta_path}")
    print()
    print("Done! Unity can now load the model from StreamingAssets/TOADGAN/")

    return onnx_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TOAD-GAN to ONNX for Unity",
    )
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing toadgan_checkpoint.pt")
    parser.add_argument("--vocab", required=True,
                        help="Path to vocab.json")
    parser.add_argument("--checkpoint", default="toadgan_checkpoint.pt",
                        help="Checkpoint filename inside model_dir")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--unity_assets", default=None,
                        help="Path to Unity Assets/StreamingAssets/ "
                             "(auto-detected by default)")
    args = parser.parse_args()

    export(
        model_dir=args.model_dir,
        vocab_path=args.vocab,
        checkpoint_name=args.checkpoint,
        opset=args.opset,
        unity_assets=args.unity_assets,
    )


if __name__ == "__main__":
    main()

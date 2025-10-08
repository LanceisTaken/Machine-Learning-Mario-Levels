import argparse
import os
import torch
from MarioTrainer import CharLSTM


def export_onnx(
    model_path: str = "mario_lstm.pt",
    onnx_path: str = "mario_lstm.onnx",
    opset: int = 14,
    device: str = "cpu",
) -> None:
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]

    model = CharLSTM(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Dummy inputs: single token and zero hidden state
    x = torch.ones(1, 1, dtype=torch.long, device=device)
    h0 = torch.zeros(cfg["num_layers"], 1, cfg["hidden_dim"], device=device)
    c0 = torch.zeros(cfg["num_layers"], 1, cfg["hidden_dim"], device=device)

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    torch.onnx.export(
        model,
        (x, (h0, c0)),
        onnx_path,
        input_names=["x", "h0", "c0"],
        output_names=["logits", "hn", "cn"],
        dynamic_axes={
            "x": {0: "batch", 1: "time"},
            "logits": {0: "batch", 1: "time"},
        },
        opset_version=opset,
    )
    print(f"Exported ONNX to {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained LSTM to ONNX")
    parser.add_argument("--model_path", type=str, default="mario_lstm.pt")
    parser.add_argument("--out", type=str, default="mario_lstm.onnx")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    export_onnx(model_path=args.model_path, onnx_path=args.out, opset=args.opset)


if __name__ == "__main__":
    main()



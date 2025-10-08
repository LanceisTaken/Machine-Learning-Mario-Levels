import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Create itos.txt (id->char) list for Unity from vocab.json")
    parser.add_argument("--vocab", type=str, default="vocab.json")
    parser.add_argument("--out", type=str, default=os.path.join("Assets", "Resources", "itos.txt"))
    args = parser.parse_args()

    with open(args.vocab, "r", encoding="utf-8") as f:
        v = json.load(f)
    itos = {int(k): ch for k, ch in v["itos"].items()}
    n = max(itos.keys()) + 1
    arr = [""] * n
    for i, ch in itos.items():
        arr[i] = "\\n" if ch == "\n" else ch

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ch in arr:
            f.write(ch + "\n")
    print(f"Wrote {args.out} with {n} tokens")


if __name__ == "__main__":
    main()



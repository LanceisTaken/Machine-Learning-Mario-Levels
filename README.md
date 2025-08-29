# Machine Learning - Mario Levels
Training a model to generate Super Mario Levels

## Quickstart

1) Install dependencies (ideally in a virtual environment):

```bash
pip install -r requirements.txt
```

2) Train an LSTM on the text levels in `Levels/`:

```bash
python MarioTrainer.py train --levels_dir Levels --output_dir . --max_epochs 20
```

This will save `mario_lstm.pt` and `vocab.json` to the output directory.

3) Generate a new level:

```bash
python MarioTrainer.py generate --model_path mario_lstm.pt --vocab_path vocab.json --length 2000 --temperature 0.9 --out generated_level.txt
```

You can optionally seed generation with some characters:

```bash
python MarioTrainer.py generate --seed_text "################\n" --length 1000
```

Notes:
- Characters are modeled at the character level. Your annotations like `#`, `?`, `e`, `p`, `P`, `y`, `o`, `c` are treated as tokens alongside whitespace and newlines.
- The script concatenates all files in `Levels/` and splits 95/5 into train/val.
- Tune `--temperature` for more or less randomness.

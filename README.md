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

4) Render a PNG of the generated level for easy viewing:

```bash
python MarioTrainer.py generate --model_path mario_lstm.pt --vocab_path vocab.json --length 2000 --render_png --png_path generated_level.png
```

The renderer colors common symbols (`#`, `?`, `B`, `P/p`, `c/C`, `o/O`, enemies like `e,g,k,K,t,l,V,h`) and leaves sky as blue. Adjust colors in `render_level_text_to_png` if your tiles differ.

To stabilize output shape and ensure a visible ground line, use postprocessing:

```bash
python MarioTrainer.py generate --length 800 --render_png --wrap_width 120 --target_height 14 --enforce_ground --png_path generated_level.png
```

If levels contain too little ground, try sampling controls and a ground bias:

```bash
python MarioTrainer.py generate --length 1200 --wrap_width 120 --target_height 14 --top_k 20 --top_p 0.9 --hash_bias 0.6 --render_png --png_path generated_level.png
```

Heuristic rules (pipes stack and require ground):

```bash
python MarioTrainer.py generate --length 1200 --wrap_width 120 --target_height 14 --enforce_ground --enforce_rules --render_png --png_path generated_level.png
```

Current rules when `--enforce_rules` is set:
- Pipes (`p/P`) stack vertically and get ground support if floating.
- Question blocks (`?`) must have sky beneath; we move them up or clear below if safe.
- Cannons (`c`) are attached to ground; if floating we add ground beneath.

Ban unwanted tokens (default bans the goal pole `|`):

```bash
python MarioTrainer.py generate --length 1200 --ban "|" --render_png --png_path generated_level.png
```

Add bottom gaps to the ground (danger pits):

```bash
python MarioTrainer.py generate --length 1680 --wrap_width 120 --target_height 14 --enforce_ground --enforce_rules --gap_rate 0.08 --gap_min 2 --gap_max 6 --render_png --png_path generated_level.png
```

Notes:
- Characters are modeled at the character level with one exception: `pP` is treated as a single compound token internally and expanded back on output. This helps keep pipes coherent.
- The script concatenates all files in `Levels/` and splits 95/5 into train/val.
- Tune `--temperature` for more or less randomness.

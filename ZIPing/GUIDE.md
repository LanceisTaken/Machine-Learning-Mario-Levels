# FYP Submission — File & Folder Guide

This document explains what every folder and file in this submission is, what it was used for, and how the pieces connect together.

---

## Top-Level Structure

```
ZIPing/
├── GUIDE.md                  ← this file
├── Annoations.txt            ← tile annotation reference
├── compute_tpkl.py           ← evaluation metric script
├── tpkl_results.csv          ← saved TPKL evaluation results
├── vocab.json                ← tile character → integer ID mapping
├── Levels/                   ← training level dataset
├── TOAD-GAN/                 ← Python ML pipeline (training + inference + API)
├── TOAD-GAN Mario/           ← Unity game project (source)
└── MarioBuild/               ← compiled Windows game build (runnable)
```

---

## Root Files

| File | Purpose |
|------|---------|
| `Annoations.txt` | Human-readable reference mapping tile characters (e.g. `X`, `?`, `E`) to their in-game meaning. Used as a legend when reading level `.txt` files. |
| `compute_tpkl.py` | Standalone script that computes the **Tile Pattern KL-Divergence (TPKL)** metric — measures how statistically similar generated levels are to the training set. Run independently of the game. |
| `tpkl_results.csv` | Saved output from `compute_tpkl.py`. Contains TPKL scores across generated level samples. |
| `vocab.json` | Shared vocabulary file mapping each tile character to a unique integer ID. Used by both the Python pipeline (training/inference) and Unity (level parsing). |

---

## `Levels/` — Training Dataset

60 hand-curated Mario level files sourced from:
- **Super Mario Bros. 1** (`SMB1_*.txt`) — 16 levels
- **Super Mario Bros. 2 (JP / Lost Levels)** (`SMB2_*.txt` and `SuperMarioBros2(J)-*.txt`) — 31 levels
- **Super Mario Land** (`super_mario_land_*.txt`) — 13 levels

Each file is a plain-text grid where each character represents one tile (e.g. `X` = solid ground, `?` = question block, `E` = enemy). These are the levels TOAD-GAN was trained on.

---

## `TOAD-GAN/` — Python Machine Learning Pipeline

The core ML component. A re-implementation of [TOAD-GAN](https://github.com/Mawiszus/TOAD-GAN) adapted and extended for this project.

### Python Scripts

| File | Purpose |
|------|---------|
| `train.py` | **Main training script.** Trains the multi-scale WGAN-GP model on a single input level. Saves generator checkpoints (`.pt` files) to `output/`. |
| `train_timed.py` | Variant of `train.py` that also records per-scale training times. Outputs results to `output_timed/` and a `training_time_report.json`. Used to benchmark training performance. |
| `generate.py` | **Inference script.** Loads trained checkpoints and generates new Mario levels. Outputs `.txt` (tile grid), `.json` (integer tile array for Unity), and `.png` (colour visualisation). |
| `server.py` | **Flask API server.** Wraps the generation pipeline in an HTTP server so Unity can request new levels at runtime via `GET /generate`. This is the bridge between the ML model and the game. |
| `export_onnx.py` | Exports the trained multi-scale pipeline to a single `.onnx` file. Bakes all generators and noise amplitudes into one traceable model for Unity Sentis/Barracuda. Produces `toadgan.onnx`. |
| `models.py` | Defines the Generator and Discriminator neural network architectures (convolutional, multi-scale). |
| `config.py` | Central configuration: hyperparameters (learning rate, number of scales, noise weight, etc.) used by training and inference. |
| `level_utils.py` | Utility functions for reading/writing level `.txt` files, converting between character grids and integer arrays, and post-processing fixes (e.g. pipe integrity, lucky block repair). |
| `constraint_report.py` | Wraps the post-processing fix functions with instrumentation. Logs which constraint violations were detected and repaired in each generated level. Produces `constraint_log.csv`. |
| `requirements.txt` | Python package dependencies. Run `pip install -r requirements.txt` to recreate the environment. |
| `vocab.json` | Local copy of the tile vocabulary (same as the root-level copy). |
| `README_USAGE.md` | Quick-start instructions for running training, inference, and the API server. |

### Output Folders

| Folder | Contents |
|--------|---------|
| `output/` | Trained model checkpoints from the main training runs. Contains per-scale generator weights (`toadgan_scale_N.pt`), a combined checkpoint (`toadgan_checkpoint.pt`), and sample generated levels (`.txt`, `.png`, `.json`). Subfolders (`SMB1_1/`, `smoke_test/`, `smoke_test2/`) correspond to different training runs. |
| `output_timed/` | Checkpoints and outputs from the timed training run (`train_timed.py`). Includes `training_time_report.json` with per-scale timing data. |
| `Sample Level Output/` | 10 sample generated levels (`generated_level_0` through `generated_level_9`) each stored as `.txt`, `.png`, and `.json`. These are the showcase outputs used in the thesis evaluation. |

### Other

| File/Folder | Contents |
|-------------|---------|
| `toadgan.onnx` | The exported ONNX model — the trained generator pipeline as a single portable file, loadable by Unity Sentis. |
| `constraint_log.csv` | Log of constraint violations and repairs recorded during generation runs. |
| `.pytest_cache/` | Cached data from pytest runs. Not needed to run anything. |
| `__pycache__/` | Python bytecode cache. Auto-generated, can be ignored. |

---

## `TOAD-GAN Mario/` — Unity Game Project (Source)

The Unity 2D game built around the TOAD-GAN generator. Open this folder in **Unity 2022.3+** to view and edit the project. The `Library/` and `Temp/` folders are excluded from this submission — Unity regenerates them automatically on first open.

### `Assets/Scripts/` — Game Scripts

| Script | Purpose |
|--------|---------|
| `ApiClient.cs` | Sends HTTP requests to `server.py` (`GET /generate`) to fetch new generated level data at runtime. |
| `ToadGanGenerator.cs` | Manages on-device ONNX inference via Unity Sentis as an alternative to the Flask API — runs `toadgan.onnx` directly inside the game. |
| `LevelInstantiator.cs` | Parses the JSON level data (integer tile array) and spawns the corresponding Unity prefabs (ground, pipes, enemies, etc.) to build the level in-scene. |
| `ChunkScheduler.cs` | Schedules ahead-of-time generation of level chunks as the player moves right, creating a seamless infinite-level experience. |
| `SchedulerConfig.cs` | ScriptableObject config for the chunk scheduler (chunk width, generation look-ahead distance, etc.). |
| `PlayerController.cs` | Handles player movement, jumping, dashing, and collision. Reads from the Unity Input System. |
| `CameraFollow.cs` | Smoothly follows the player horizontally; locks the camera at level bounds vertically. |
| `EnemyPatrol.cs` | Makes enemies walk back and forth on platforms and reverses direction on edges or walls. |
| `EntityUnstuck.cs` | Detects and nudges entities that have clipped into geometry, preventing soft-locks. |
| `GameManager.cs` | Central game state manager: tracks score, lives, win/lose conditions, and coordinates scene transitions. |
| `UIManager.cs` | Updates all on-screen HUD elements (score, lives, timer). |
| `UIPopup.cs` | Shows/hides popup panels (pause menu, game over screen, etc.). |
| `RestartManager.cs` | Handles restarting the current level or returning to the main menu. |
| `QuestionBlock.cs` | Logic for question/lucky blocks — spawns a power-up or coin when hit from below. |
| `PowerUpItem.cs` | Base class for power-up behaviour (sliding out of block, moving until collected). |
| `SuperMushroom.cs` | Power-up that grows the player (increases size and grants an extra hit). |
| `OneUpMushroom.cs` | Power-up that awards an extra life. |
| `StarMan.cs` | Star power-up granting temporary invincibility and enemy-defeat on contact. |
| `JumpPowerUpCollectible.cs` | Collectible that temporarily boosts the player's jump height. |
| `WindDashCollectible.cs` | Collectible that enables the wind-dash ability (horizontal air dash). |
| `DashTrail.cs` | Renders a fading sprite trail behind the player during a dash. |
| `ScrollingBackground.cs` | Parallax-scrolls background layers at different speeds relative to the camera. |
| `PerformanceMonitor.cs` | Tracks and displays runtime FPS and frame time; used during development profiling. |

### Other Asset Folders

| Folder | Contents |
|--------|---------|
| `Assets/Scenes/` | The main Unity scene (`SampleScene.unity`) — the full game level scene with all objects configured. |
| `Assets/Prefabs/` | Reusable GameObjects: Ground, Brick, Pipe Head/Body, Coin, Lucky Block, Enemy, Player, and power-up items. |
| `Assets/Sprites/` | Sprite sheets for tiles, enemies, and player animations; background images. |
| `Assets/Animations/` | Player animation controller and clips (idle, walk, jump, death). |
| `Assets/Models/` | The `toadgan.onnx` model file embedded in the Unity project for on-device Sentis inference. |
| `Assets/StreamingAssets/TOADGAN/` | Runtime-loaded JSON files: `toadgan_meta.json` (model metadata) and `vocab.json` (tile vocabulary), accessed by the game at runtime. |
| `Assets/Settings/` | Universal Render Pipeline (URP) 2D renderer and pipeline asset configuration. |
| `Assets/TextMesh Pro/` | TextMesh Pro package assets (fonts, shaders) used for all in-game UI text. |
| `Packages/` | Unity package manifest (`manifest.json`, `packages-lock.json`) — defines all Unity package dependencies. |
| `ProjectSettings/` | Unity editor project settings (physics, audio, input, graphics, tags/layers). |
| `UserSettings/` | Local editor layout and preferences (not required to run the project). |
| `OPTIMISATION_LOG.md` | Development log recording performance optimisation steps taken during the project. |

---

## `MarioBuild/` — Compiled Windows Game Build

> **Note:** The `.exe` and `UnityPlayer.dll` files must be present here for the game to be runnable. If they are missing, rebuild from the Unity project via `File → Build Settings → Build`.

A compiled Windows standalone build of the game. To play:
1. Ensure `server.py` is running (`python server.py` in the `TOAD-GAN/` folder).
2. Double-click `TOAD-GAN Mario.exe`.

| Item | Purpose |
|------|---------|
| `TOAD-GAN Mario.exe` | The game executable. |
| `UnityPlayer.dll` | Unity engine runtime DLL — required to launch the `.exe`. |
| `DirectML.dll` | DirectML library for GPU-accelerated ML inference (Unity Sentis). |
| `UnityCrashHandler64.exe` | Unity crash reporter — launched automatically on crash. |
| `TOAD-GAN Mario_Data/` | Game data: compiled assets, level resources, managed DLLs, ONNX model, streaming assets. |
| `MonoBleedingEdge/` | Mono runtime for executing C# game scripts. |
| `D3D12/` | DirectX 12 shader cache used by the URP renderer. |

---

## How It All Connects

```
Training levels (Levels/)
        │
        ▼
  train.py  ──►  output/toadgan_scale_N.pt  (trained weights)
        │
        ▼
  export_onnx.py  ──►  toadgan.onnx
        │                    │
        ▼                    ▼
  server.py            Unity Sentis
  (Flask API)       (on-device inference)
        │                    │
        └──────────┬──────────┘
                   ▼
           LevelInstantiator.cs
           (spawns level tiles)
                   │
                   ▼
           ChunkScheduler.cs
        (streams chunks as player moves)
                   │
                   ▼
           Playable Mario game
```

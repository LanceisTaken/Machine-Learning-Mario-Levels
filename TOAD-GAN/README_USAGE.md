# TOAD-GAN Usage Guide

Here are the commands you need to run the Python Flask server and the generator script using your trained model.

### Prerequisites
Make sure you are in the `TOAD-GAN` directory and your virtual environment (if any) is activated. All commands assume that your trained model checkpoint `toadgan_checkpoint.pt` is inside the `output` directory.

---

### 1. Running the Python Flask Server

The server provides a local API (by default at `http://127.0.0.1:5000/generate`) that Unity or other clients can use to request levels on the fly.

To start the server, run:
```bash
python server.py --model_dir output --vocab vocab.json
```

***Optional Arguments:***
* `--port 5000`: Change the port to listen on.
* `--host 0.0.0.0`: Allow access from other devices on your LAN.
* `--device cpu`: Force the server to use CPU instead of CUDA.

---

### 2. Running `generate.py` with your model

The standalone generator script allows you to create raw `.txt`, `.png`, and `.json` level representations without using the server API.

To generate a single level locally, run:
```bash
python generate.py --model_dir output --vocab vocab.json --out generated_level.txt
```

***Optional Arguments:***
* `--temperature 1.5`: Increase the noise temperature for more level variation (values > 1.0 produce wilder results, values < 1.0 stick closer to training data).
* `--scale_w 1.5`: Make the generated level wider (multiplier).
* `--scale_h 1.2`: Make the generated level taller (multiplier).
* `--num_samples 5`: Generate multiple levels at once.
* `--no_png` or `--no_json`: Skip saving those extra visual/JSON outputs.

*(The generated outputs will be stored in your `TOAD-GAN` directory alongside the generated `.txt` file).*

# Running Thesis Builder on a local model

This runs the real pipeline on your own machine with open-weight models served
by [Ollama](https://ollama.com). No API keys and no per-token cost. Web search
still needs an internet connection (it uses DuckDuckGo, which is free).

The commands below work in **PowerShell** on Windows (open the Start menu and
type "PowerShell") and in a **Terminal** on Linux or macOS. Type each line,
press Enter, and wait for it to finish before the next one.

## 1. Install the tools (once)

1. **Ollama:** download and install it from https://ollama.com/download.
2. **Git and uv** (uv runs the Python project for you):

   Windows (PowerShell):
   ```powershell
   winget install --id Git.Git -e
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   Linux or macOS:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Close the window and open a new one so the new commands are found.

## 2. Get the code

```bash
git clone https://github.com/ankittalwar07/thesis-builder
cd thesis-builder
uv sync
```

## 3. Download the models (once, about 45 GB, can take an hour)

```bash
ollama pull llama3.3:70b
ollama pull llama3.1:8b
ollama create thesis-70b -f config/ollama/thesis-70b.Modelfile
ollama create thesis-8b -f config/ollama/thesis-8b.Modelfile
```

The last two lines make copies of the models with a larger working memory
(context window), which the pipeline needs.

## 4. Check the setup

```bash
uv run python scripts/check_models.py --config config/models.local.yaml
```

You should see three `[ok]` lines with a speed in tokens per second. Then, while
a model is still loaded, run:

```bash
ollama ps
```

The `PROCESSOR` column should say `100% GPU`. If it says CPU, the model is not
using the graphics chip and runs will be very slow; stop there and ask.

## 5. Run one theme

```bash
uv run thesis "Robotics" --config config/models.local.yaml
```

Expect this to take a long time with a 70B model (an hour or more is normal).
The result lands in `outputs/robotics/`, including `rebuttals.json`, which
shows every critique beside its answer and whether the rebuttal check passed.

## 6. The comparison for the paper (leave it running, for example overnight)

First stop the machine from sleeping (Windows: Settings, System, Power, set
sleep to Never while plugged in). Then:

```bash
uv run python scripts/ablation.py --config config/models.local.yaml
```

It runs all three themes with and without the Skeptic. The numbers end up in
`outputs/ablation/summary.json`.

## 7. Record runs for the demo booth

```bash
uv run thesis-serve --live --config config/models.local.yaml --record runs/
```

Open http://127.0.0.1:8000/ in a browser and run each theme once. Each finished
run is saved in `runs/`. Later, `uv run thesis-serve --replay runs/` plays them
back with no network at all.

## Using the box from another computer (later)

Ollama only listens to its own machine by default. To let a laptop use the
box, set `OLLAMA_HOST` to `0.0.0.0` on the box (see Ollama's FAQ), then on the
laptop set `OLLAMA_API_BASE` to `http://<box address>:11434`. Do this only on a
network you trust: anyone who can reach that address can use the models.

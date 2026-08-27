# Adversarial ML Security Playground

An interactive web app for attacking, defending, and scoring the robustness of ML models.
Two playgrounds are included out of the box:

- **Computer Vision** — 8×8 handwritten digits (an MNIST-style dataset bundled with
  scikit-learn, so there's nothing to download)
- **Cybersecurity ML** — a synthetic network-intrusion dataset (`BENIGN` vs `ATTACK`) with
  realistic behavioural features

You train a model, attack it (FGSM / PGD / DeepFool / random noise), defend it
(preprocessing or adversarial training), and get a 0–100 robustness score with a
downloadable report.

```
adversarial-ml-playground/
├── backend/     FastAPI + PyTorch + scikit-learn API
└── frontend/    Next.js + Tailwind + Recharts dashboard
```

Both the backend and frontend were built and smoke-tested (backend: unit-tested piece by
piece; frontend: `npm run build` completed with no errors) — the only thing not tested
end-to-end in a live session is the PyTorch training/attack code itself, since PyTorch's
install is too large for the sandbox this was built in. It's ordinary PyTorch code and
should run fine on a normal machine.

---

## 1. Which IDE

**VS Code** is a good fit here — one editor, two extensions, and it comfortably runs the
Python backend and the Next.js frontend side by side in separate integrated terminals.

Install these extensions:

- **Python** (`ms-python.python`) — backend editing, linting, the built-in test runner
- **Pylance** (`ms-python.vscode-pylance`) — usually installed automatically with the
  Python extension; gives you type checking and autocomplete
- **ES7+ React/Redux/React-Native snippets** (`dsznajder.es7-react-js-snippets`) — optional,
  handy for the frontend
- **Tailwind CSS IntelliSense** (`bradlc.vscode-tailwindcss`) — autocomplete for the utility
  classes used throughout `frontend/`

Open the **project root** (`adversarial-ml-playground/`) as your VS Code workspace, not
just one subfolder — that way both `backend/` and `frontend/` show up in the sidebar and
you can run one terminal per side.

If you use PyCharm or another IDE instead, everything below still applies — it's just
terminal commands.

---

## 2. Prerequisites

- **Python 3.10–3.12** (PyTorch doesn't yet support every 3.13 build — 3.10–3.12 is safest)
- **Node.js 18.18+** (Next.js 14 requirement) — Node 20 LTS is a safe choice
- pip and npm on your PATH

Check what you have:

```bash
python3 --version
node --version
npm --version
```

---

## 3. Backend setup (FastAPI)

From the project root:

```bash
cd backend
python3 -m venv venv

# activate it
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate              # Windows (Command Prompt / PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

This installs FastAPI, PyTorch (CPU build from PyPI), scikit-learn, and the rest.
PyTorch is a large download (a few hundred MB) — it just takes a minute, that's normal.

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

You should see `Uvicorn running on http://127.0.0.1:8000`. Leave this terminal running.
Visit `http://localhost:8000/api/health` — you should get `{"status":"ok"}`. FastAPI's
interactive API docs are also available at `http://localhost:8000/docs` if you want to
poke at the endpoints directly.

A local `experiments.db` SQLite file will be created automatically the first time you
start the server — that's where the experiment history / activity log lives.

---

## 4. Frontend setup (Next.js)

Open a **second terminal**, from the project root:

```bash
cd frontend
npm install
```

Copy the example env file (it just points the frontend at your local backend):

```bash
cp .env.local.example .env.local        # macOS/Linux
copy .env.local.example .env.local       # Windows
```

Run the dev server:

```bash
npm run dev
```

Visit `http://localhost:3000`. You should see the dashboard. If it shows a "can't reach
the backend" message, double-check the FastAPI server from step 3 is still running on
port 8000.

---

## 5. Using it

1. **Models** — pick a dataset (digits or network intrusion) and an architecture
   (Logistic Regression, Random Forest, Small NN, or CNN — CNN is image-only), then
   train it. Training runs live and usually takes a few seconds.
2. **Attack Lab** — pick a trained model, pick an attack (FGSM, PGD, DeepFool, or a
   random-noise baseline), set the strength (epsilon), and run it. You'll see the
   original input, the perturbation, the adversarial input, and how the prediction
   changed — as images for the digits dataset, or a feature-by-feature delta table for
   the network-intrusion dataset.
3. **Defense Lab** — two independent tools:
   - **Preprocessing**: run an attack, then apply a defense (Gaussian smoothing, feature
     clipping, or normalization) to the adversarial input and see how much accuracy comes
     back.
   - **Adversarial training**: trains a hardened copy of a model on a mix of clean and
     adversarial examples, then compares clean/attacked accuracy for the original vs. the
     hardened version.
4. **Robustness Report** — pick a model and one or more attacks, run a batch evaluation
   over the held-out test set, and get a 0–100 robustness score plus a per-attack
   breakdown. Export the full report as JSON from the button on that page.

A note on Random Forest: it's not differentiable, so gradient-based attacks (FGSM/PGD/
DeepFool) can't be computed against it directly. The app handles this automatically by
training a small differentiable "surrogate" network that mimics the forest's predictions,
crafting the attack against the surrogate, then evaluating the real forest on the result —
a standard technique called a transfer / substitute-model attack. You'll see a "surrogate"
badge on Random Forest models to make this visible rather than silent.

---

## 6. Project structure

```
backend/
  app/
    main.py                    FastAPI app + all API routes
    schemas.py                 Pydantic request/response models
    storage.py                 SQLite experiment history
    data/
      mnist_loader.py          8x8 digit dataset (sklearn, no download)
      cyber_loader.py          synthetic network-intrusion dataset
    models/
      torch_models.py          LogisticRegression / SmallNN / CNN (PyTorch)
      registry.py               training, in-memory model store, RF surrogate logic
    attacks/
      fgsm.py, pgd.py, deepfool.py, random_noise.py
      runner.py                 dispatches by attack name, handles RF transfer attacks
    defenses/
      preprocessing.py          gaussian smoothing / clipping / normalization
      adversarial_training.py   retrains a hardened model
    evaluation/
      robustness.py             batch evaluation + 0-100 robustness score
      explainability.py         perturbation heatmaps, tabular feature deltas
  requirements.txt

frontend/
  app/                          Next.js App Router pages (dashboard, models, attack-lab,
                                 defense-lab, robustness-report)
  components/                   shared UI (Sidebar, Panel, ModelPicker, charts, etc.)
  lib/api.js                    fetch wrapper for the backend
  package.json
```

---

## 7. Extending it (matches the original project brief)

- **Real MNIST**: swap `mnist_loader.load_image_dataset` for a
  `torchvision.datasets.MNIST` loader if you want real 28×28 images instead of the 8×8
  stand-in — the rest of the pipeline (attacks, defenses, evaluation) doesn't assume a
  specific resolution.
- **Real intrusion-detection data**: swap `cyber_loader.load_cyber_dataset` for a
  `pandas.read_csv` loader over NSL-KDD, CIC-IDS, or a Kaggle phishing/malware dataset —
  everything downstream only assumes a 2D float feature matrix and integer labels.
- **More attacks**: add a new file under `attacks/`, register it in
  `attacks/runner.ATTACK_NAMES` and the `run_attack` dispatcher.
- **More defenses**: add a function to `defenses/preprocessing.PREPROCESSING_DEFENSES`.
- **Persist trained models to disk**: right now trained models live in memory
  (`registry.MODEL_STORE`) and reset when the server restarts — swap in `torch.save` /
  `joblib.dump` plus a lookup table if you want them to survive a restart.
- **Resume line**: *"Developed an interactive adversarial machine-learning security
  platform to evaluate model robustness against FGSM, PGD, and DeepFool attacks,
  visualizing adversarial perturbations, attack success rates, confidence degradation, and
  defense effectiveness through adversarial training and input preprocessing."*

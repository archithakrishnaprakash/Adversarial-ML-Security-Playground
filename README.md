# Adversarial ML Security Playground

An interactive web app for attacking, defending, and scoring the robustness of ML models —
and, since the second build pass, for running the kind of assessment a security team would
actually want: threat-model-aware attack selection, training-time poisoning, pure black-box
attacks, cross-model transferability, a risk-rated security assessment with findings, an
automated experiment grid, a robustness leaderboard, and an MLSecOps CI gate.

Two playgrounds are included out of the box:

- **Computer Vision** — 8×8 handwritten digits (an MNIST-style dataset bundled with
  scikit-learn, so there's nothing to download)
- **Cybersecurity ML** — a synthetic network-intrusion dataset (`BENIGN` vs `ATTACK`) with
  realistic behavioural features

```
adversarial-ml-playground/
├── .github/workflows/   ML security gate CI workflow
├── backend/     FastAPI + PyTorch + scikit-learn API
└── frontend/    Next.js + Tailwind + Recharts dashboard
```

Both the backend and frontend were built and tested (backend: `pytest` suite, 38 passing
+ a torch-gated suite for training/attack code; frontend: `npm run build` completed with no
errors across all 10 pages). The only thing not tested end-to-end in a live session is the
PyTorch training/attack code itself, since PyTorch's install is too large for the sandbox
this was built in — see [Testing](#8-testing) below for exactly what ran and what didn't.

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

This installs FastAPI, PyTorch (CPU build from PyPI), scikit-learn, pytest, and the rest.
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

**Core workflow:**

1. **Models** — pick a dataset (digits or network intrusion) and an architecture
   (Logistic Regression, Random Forest, Small NN, or CNN — CNN is image-only), then
   train it. Training runs live and usually takes a few seconds.
2. **Attack Lab** — pick a threat model (white-box / gray-box / black-box), pick an
   attack, set the strength (epsilon), and run it. Attacks that don't apply under the
   selected threat model (e.g. FGSM under black-box) are flagged and disabled rather than
   silently failing. The same page also has a **Black-Box Query Lab** panel for attacks
   that never touch a gradient at all — `transfer` (train a query-limited substitute
   model, attack that) and `query` (zeroth-order random search directly against the
   prediction API), both reporting the number of queries spent.
3. **Poisoning Lab** — training-time attacks instead of inference-time ones: **label
   flipping** (relabel a fraction of training data) and a **backdoor trigger** (stamp a
   fixed pattern onto a fraction of training data, relabel only those to a target class).
   Both train a clean baseline alongside the poisoned model so the gap is directly
   comparable, and register both as normal models you can open elsewhere in the app.
4. **Defense Lab** — preprocessing defenses (Gaussian smoothing, feature clipping,
   normalization) and adversarial training, with before/after accuracy comparisons.
5. **Robustness Report** — batch evaluation over the held-out test set: a 0–100
   robustness score plus, per attack, accuracy, attack success rate, L0/L2/L∞
   perturbation norms, and average confidence shift. Exportable as JSON.
6. **Security Assessment** — turns a robustness benchmark (optionally combined with a
   transferability check against other trained models) into a risk rating (HIGH / MEDIUM
   / LOW), a sorted list of specific findings ("PGD reduces accuracy from 98% to 61%..."),
   and a deploy / don't-deploy recommendation with concrete next steps.
7. **Leaderboard** — ranks every model that's had a robustness evaluation run against it,
   sortable by robustness score, clean accuracy, or attack success rate.
8. **Experiment Runner** — sweeps a whole grid (models × attacks × epsilons × defenses)
   in one request instead of clicking through Attack Lab one configuration at a time.
   Capped at 60 configurations per run since it executes synchronously; every
   configuration is logged individually via the experiment history, so results are
   reproducible later, not just summarized away.

A note on Random Forest: it's not differentiable, so gradient-based attacks (FGSM/PGD/
DeepFool) can't be computed against it directly. The app handles this automatically by
training a small differentiable "surrogate" network that mimics the forest's predictions,
crafting the attack against the surrogate, then evaluating the real forest on the result —
a standard technique called a transfer / substitute-model attack. You'll see a "surrogate"
badge on Random Forest models to make this visible rather than silent. The same surrogate
mechanism powers the Black-Box Lab's `transfer` method for *any* model, not just Random
Forest — the difference is the black-box surrogate is trained by *querying* the target
(no true labels), while the Random Forest surrogate is trained at registration time on the
forest's own training data.

---

## 6. MLSecOps: the CI security gate

`backend/scripts/security_gate.py` trains a model, runs a robustness benchmark against it,
and exits non-zero if robust accuracy falls below a threshold — the same idea as a
test-coverage gate, applied to adversarial robustness. Run it directly:

```bash
cd backend
python scripts/security_gate.py --dataset cyber --model logistic_regression \
    --attack pgd --epsilon 0.15 --threshold 0.70
```

`.github/workflows/ml-security-gate.yml` runs this automatically in GitHub Actions on any
push/PR that touches `backend/`, after running the pytest suite. It can also be triggered
manually from the Actions tab with a different dataset/model/threshold via
`workflow_dispatch`. If you don't use GitHub, the script itself has no GitHub-specific
dependencies — call it from any CI system the same way.

---

## 7. API reference (new endpoints)

All endpoints from the original build are unchanged and **fully backward compatible** —
see [Migration notes](#9-migration-notes-backward-compatibility) below. New surface area:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/threat-model/matrix` | GET | Full capability × attack applicability table |
| `/api/threat-model/check` | POST | Check whether one attack applies under one capability |
| `/api/attack/blackbox` | POST | Run `transfer` or `query` black-box attacks |
| `/api/poisoning/label-flip` | POST | Label-flip poisoning experiment |
| `/api/poisoning/backdoor` | POST | Backdoor-trigger poisoning experiment |
| `/api/robustness/transferability` | POST | Cross-model adversarial transferability matrix |
| `/api/security/assessment` | POST | Risk rating + findings + recommendation |
| `/api/leaderboard` | GET | Ranked models by latest robustness evaluation |
| `/api/experiments/grid` | POST | Run a capped grid sweep |
| `/api/experiments/grid/limits` | GET | The current grid size cap |

Full request/response shapes are in `backend/app/schemas.py` and viewable interactively at
`http://localhost:8000/docs` once the server is running.

---

## 8. Testing

```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

The suite is split by dependency:

- **Torch-free tests** (threat model, poisoning primitives, perturbation norms, risk
  engine, data loaders, preprocessing defenses) — **38 tests, all passing**, verified in
  this build.
- **`tests/test_torch_dependent.py`** (model training, FGSM/PGD/DeepFool, black-box
  attacks, poisoning-lab training, transferability, experiment grid, robustness Lp norms)
  — written and syntax-checked, uses `pytest.importorskip("torch")` so it **skips cleanly**
  in an environment without PyTorch rather than failing the whole suite. Runs for real
  wherever PyTorch is installed (i.e. after step 3 above, on any normal machine).

If you only see 38 passed / 1 skipped, that's expected in a torch-free environment — it's
not a sign anything is broken.

---

## 9. Migration notes (backward compatibility)

Every change in this pass was additive. If you have code or scripts calling the API from
before this update:

- **`POST /api/attack/run`** gained one new optional field, `capability`, defaulting to
  `"white_box"` — the exact behavior every prior caller already got. It's only enforced if
  you explicitly send a different capability and pick an attack that isn't valid under it
  (e.g. `fgsm` under `black_box`), in which case you now get a `400` explaining why instead
  of the attack silently running as if it had gradient access it wouldn't really have.
- **`POST /api/robustness/evaluate`**'s per-attack response objects gained four new keys
  (`l0_norm`, `l2_norm`, `linf_norm`, `confidence_shift`) alongside the existing ones —
  nothing existing was removed or renamed.
- **`app.models.registry.train_model()`** has the exact same signature and behavior as
  before; internally it's now a thin wrapper around `fit_and_register()`, which the
  poisoning lab also calls with tampered data. No caller-visible change.
- Every other addition is a new file, new function, or new endpoint — nothing else was
  removed, renamed, or had its default behavior changed.

---

## 10. Project structure

```
backend/
  app/
    main.py                    FastAPI app + all API routes
    schemas.py                 Pydantic request/response models
    storage.py                 SQLite experiment history
    threat_model.py            capability x attack applicability engine
    data/
      mnist_loader.py          8x8 digit dataset (sklearn, no download)
      cyber_loader.py          synthetic network-intrusion dataset
    models/
      torch_models.py          LogisticRegression / SmallNN / CNN (PyTorch)
      registry.py               training, in-memory model store, RF surrogate logic
    attacks/
      fgsm.py, pgd.py, deepfool.py, random_noise.py
      poisoning.py               label-flip and backdoor-trigger primitives
      blackbox.py                surrogate-transfer and zeroth-order query attacks
      runner.py                  dispatches by attack name, handles RF transfer attacks
    defenses/
      preprocessing.py          gaussian smoothing / clipping / normalization
      adversarial_training.py   retrains a hardened model
    evaluation/
      robustness.py             batch evaluation + 0-100 robustness score + Lp norms
      explainability.py         perturbation heatmaps, tabular feature deltas
      norms.py                  L0/L2/Linf perturbation norms, confidence shift
      poisoning_eval.py         orchestrates clean-vs-poisoned training experiments
      transferability.py        cross-model adversarial transferability matrix
      risk_engine.py            risk rating, findings generator, recommendations
      leaderboard.py            ranks models by latest robustness evaluation
    experiments/
      runner.py                 capped grid-sweep engine
  scripts/
    security_gate.py           MLSecOps CI gate CLI
  tests/                       pytest suite (see section 8)
  requirements.txt

frontend/
  app/                          Next.js App Router pages: dashboard, models, attack-lab
                                 (incl. threat model + black-box lab), poisoning-lab,
                                 defense-lab, robustness-report, security-assessment,
                                 leaderboard, experiments
  components/                   shared UI (Sidebar, Panel, ModelPicker, charts, etc.)
  lib/api.js                    fetch wrapper for the backend
  package.json

.github/workflows/
  ml-security-gate.yml          pytest + security_gate.py on every backend change
```

---

## 11. Extending it further

- **Real MNIST**: swap `mnist_loader.load_image_dataset` for a
  `torchvision.datasets.MNIST` loader if you want real 28×28 images instead of the 8×8
  stand-in — the rest of the pipeline doesn't assume a specific resolution.
- **Real intrusion-detection data**: swap `cyber_loader.load_cyber_dataset` for a
  `pandas.read_csv` loader over NSL-KDD, CIC-IDS2017, or UNSW-NB15 — everything
  downstream only assumes a 2D float feature matrix and integer labels. This wasn't
  wired up automatically because those datasets are multi-gigabyte downloads from
  external hosts; grab one yourself and point a loader at the CSV.
- **Async experiment grid**: the grid runner is currently synchronous and capped at 60
  configurations per call so a single HTTP request doesn't time out. A background job
  queue (Celery, RQ, or even a simple thread + polling endpoint) would remove that cap.
- **Membership inference / model extraction**: not implemented — the threat-model and
  evaluation architecture (see `app/evaluation/`) is set up so a new attack category is
  "add a module, register it, add an endpoint," following the same pattern as poisoning
  or black-box attacks.
- **PDF report export**: `/api/experiments/export` currently returns JSON; a PDF
  renderer (e.g. `reportlab` or `weasyprint` over an HTML template) could sit in front of
  the same data.
- **Resume line**: *"Engineered an adversarial ML security assessment framework
  evaluating multiple model architectures across evasion, poisoning, and black-box attack
  surfaces; implemented threat-model-aware attack gating, cross-model transferability
  analysis, and an automated robustness regression gate integrated into CI."*


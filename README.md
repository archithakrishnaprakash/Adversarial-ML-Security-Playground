# AegisML — Adversarial ML Security Assessment Framework

*(formerly "Adversarial ML Security Playground" — renamed to reflect what it's grown into:
a systematic robustness benchmark, not just an attack demo. See §9 for what actually changed
under the hood; the rename itself is cosmetic.)*

An interactive web app for attacking, defending, and scoring the robustness of ML models —
and, as of this build pass, for running the kind of assessment a security team would
actually want: threat-model-aware attack selection, training-time poisoning, pure black-box
attacks, cross-model transferability, a systematic robustness benchmark matrix, a risk-rated
security assessment with findings, an automated experiment grid, a robustness leaderboard,
and an MLSecOps CI gate.

Three datasets are included:

- **Computer Vision** — 8×8 handwritten digits (an MNIST-style dataset bundled with
  scikit-learn, so there's nothing to download)
- **Cybersecurity ML — synthetic (quick-start)** — a synthetic network-intrusion dataset
  (`BENIGN` vs `ATTACK`) with realistic behavioural features, zero setup
- **Cybersecurity ML — UNSW-NB15 (real)** — a real, published intrusion-detection dataset.
  Not bundled (it's a genuine external download), but the exact same attack/defense/
  evaluation pipeline runs against it once you've downloaded it — see §5b. This exists
  because a synthetic-only cybersecurity dataset is a fair thing for a reviewer to push
  back on; "the attacks work on my generated data" and "the attacks work on real traffic"
  are different claims.

```
adversarial-ml-playground/
├── .github/workflows/   ML security gate CI workflow
├── backend/     FastAPI + PyTorch + scikit-learn API
└── frontend/    Next.js + Tailwind + Recharts dashboard
```

Both the backend and frontend were built and tested (backend: `pytest` suite, 46 passing
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

1. **Models** — pick a dataset (digits, synthetic network intrusion, or real UNSW-NB15
   intrusion data — see §5b for that last one) and an architecture (Logistic Regression,
   Random Forest, Small NN, or CNN — CNN is image-only), then train it. Training runs live
   and usually takes a few seconds.
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
5. **Robustness Report** — batch evaluation over the held-out test set for *one* model.
   Standard metrics come first (robust accuracy, accuracy degradation, attack success
   rate, L1/L2/L∞ perturbation norms, confidence shift, all per attack) — the Aegis
   Robustness Index (a single 0–100 composite score) is shown separately, explicitly
   labeled as a project metric rather than a standardized benchmark. Exportable as JSON.
6. **Security Assessment** — turns a robustness benchmark (optionally combined with a
   transferability check against other trained models) into a risk rating (HIGH / MEDIUM
   / LOW), a sorted list of specific findings ("PGD reduces accuracy from 98% to 61%..."),
   and a deploy / don't-deploy recommendation with concrete next steps.
7. **Leaderboard** — ranks every model that's had a robustness evaluation run against it,
   sortable by Aegis Robustness Index, clean accuracy, or attack success rate.
8. **Benchmark & Experiments** — two systematic-evaluation tools instead of running Attack
   Lab one configuration at a time: a **Robustness Matrix** (every selected model ×
   every selected attack, at one epsilon — the classic side-by-side benchmark table) for
   an at-a-glance comparison, and a full **grid sweep** (models × attacks × epsilons ×
   defenses, capped at 60 configurations per run since it executes synchronously) for
   deeper exploration. Every configuration from either tool is logged individually via the
   experiment history, so results are reproducible later, not just summarized away.

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

## 5b. Getting real intrusion-detection data (UNSW-NB15)

The synthetic cybersecurity dataset is deliberately zero-setup — generated on the fly, no
download, fast to train on. That's good for a quick demo, but it can't demonstrate that
these attacks work against realistic network traffic; that's a fair thing for a reviewer to
push back on. UNSW-NB15 closes that gap: same pipeline, real data.

1. **Download it.** UNSW-NB15 is published by the Australian Centre for Cyber Security
   (research.unsw.edu.au/projects/unsw-nb15-dataset) and commonly mirrored on Kaggle. Get
   the "clean CSV" release — usually distributed as two files with headers already
   included: `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`. This project
   can't download it for you (it's a genuine external, multi-hundred-MB file on a host
   outside this sandbox's network access).
2. **Place the files.** Put both CSVs in a directory, e.g. `backend/data/unsw_nb15/`.
3. **Point the backend at it.** Set the `UNSW_NB15_DATA_DIR` environment variable to that
   directory before starting the backend:
   ```bash
   export UNSW_NB15_DATA_DIR=./data/unsw_nb15
   uvicorn app.main:app --reload --port 8000
   ```
   (On Docker: add `UNSW_NB15_DATA_DIR=/app/data/unsw_nb15` to the backend service's
   `environment:` in `docker-compose.yml` and mount the directory as a volume.)
4. **Select "Network Intrusion — UNSW-NB15 (real)"** on the Models page. If the files
   aren't found, the dataset shows as "needs setup" instead of the app just failing.

What the loader does: drops the row `id` and free-text `attack_cat` columns (label alone
encodes attack-vs-not), label-encodes the three categorical fields (`proto`, `service`,
`state` — one-hot would blow up the feature space, since `proto` alone has 100+ distinct
values in the full dataset), min-max scales every feature to `[0, 1]` using train-set
statistics only, and by default subsamples to 20,000 training / 5,000 test rows for
training speed (raise `max_samples` in `unsw_nb15_loader.load_unsw_nb15` if you want the
full ~175k/82k rows and don't mind slower training). Only a single CSV works too, if
that's what you have — it'll be internally split 80/20.

**CIC-IDS2017 wasn't wired up** alongside UNSW-NB15 — not because it wouldn't work, but
because implementing two real-dataset loaders for one project pass is exactly the kind of
feature-hoarding this app is trying to avoid ("4 attacks + rigorous evaluation" beats "12
attacks + no depth", and the same principle applies to datasets). If you want it: it's
messier (multiple CSVs, 78 features, known encoding quirks) but the same shape of problem —
a new loader module returning `(X_train, X_test, y_train, y_test, feature_names)` plugs
into `registry.py` the same way `unsw_nb15_loader.py` does.

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
| `/api/robustness/matrix` | POST | Models × attacks robustness benchmark matrix |
| `/api/security/assessment` | POST | Risk rating + findings + recommendation |
| `/api/leaderboard` | GET | Ranked models by latest robustness evaluation |
| `/api/experiments/grid` | POST | Run a capped grid sweep |
| `/api/experiments/grid/limits` | GET | The current grid size cap |

`/api/datasets` now also reports an `available` boolean per dataset (`real_ids` is `false`
until UNSW-NB15 is set up per §5b) and `/api/robustness/evaluate`'s response gained
`aegis_robustness_index` / `mean_robust_accuracy` / `mean_accuracy_degradation` fields and
an `l1_norm` per attack — see [Migration notes](#9-migration-notes-backward-compatibility).

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
  engine, data loaders, preprocessing defenses, UNSW-NB15 loader) — **46 tests, all
  passing**, verified in this build.
- **`tests/test_torch_dependent.py`** (model training, FGSM/PGD/DeepFool, black-box
  attacks, poisoning-lab training, transferability, robustness matrix, experiment grid,
  robustness Lp norms) — written and syntax-checked, uses `pytest.importorskip("torch")`
  so it **skips cleanly** in an environment without PyTorch rather than failing the whole
  suite. Runs for real wherever PyTorch is installed (i.e. after step 3 above, on any
  normal machine).

If you only see 46 passed / 1 skipped, that's expected in a torch-free environment — it's
not a sign anything is broken.

---

## 9. Migration notes (backward compatibility)

Every change across both build passes has been additive. If you have code or scripts
calling the API from before this update:

- **`POST /api/attack/run`** gained one new optional field, `capability`, defaulting to
  `"white_box"` — the exact behavior every prior caller already got. It's only enforced if
  you explicitly send a different capability and pick an attack that isn't valid under it
  (e.g. `fgsm` under `black_box`), in which case you now get a `400` explaining why instead
  of the attack silently running as if it had gradient access it wouldn't really have.
- **`POST /api/robustness/evaluate`**'s per-attack response objects gained five new keys
  (`accuracy_degradation`, `l0_norm`, `l1_norm`, `l2_norm`, `linf_norm`, `confidence_shift`)
  alongside the existing ones. The top-level response also gained `mean_robust_accuracy`,
  `mean_accuracy_degradation`, `aegis_robustness_index`, and
  `aegis_robustness_index_note` — **`robustness_score` is unchanged and still present**,
  holding the exact same value as `aegis_robustness_index` (same composite formula as
  before; it's now just explicitly labeled and disclaimed as a project metric rather than
  a standardized benchmark, per the framing change described in the intro). Nothing
  existing was removed or renamed.
- **`GET /api/leaderboard`** rows gained `aegis_robustness_index` and
  `mean_robust_accuracy` fields; `robustness_score` is unchanged and still present with
  the same value.
- **`GET /api/datasets`** gained an `available` boolean and an optional `note` string per
  dataset (used for the new `real_ids` UNSW-NB15 dataset, which is unavailable until you've
  downloaded it — see §5b). Existing datasets (`image`, `cyber`) always report
  `available: true`, matching their previous unconditional-availability behavior.
- **`app.models.registry.get_data()`** has the exact same signature and behavior for
  `"image"` and `"cyber"` as before. It now also accepts `"real_ids"`, which raises a
  `ValueError` with setup instructions (surfaced as a `400`) rather than a dataset error
  if the UNSW-NB15 CSVs aren't present — this is new behavior for a new dataset key, not a
  change to the two existing ones.
- **`app.models.registry.train_model()`** has the exact same signature and behavior as
  before; internally it's now a thin wrapper around `fit_and_register()`, which the
  poisoning lab and the robustness matrix both also call directly. No caller-visible change.
- Every other addition is a new file, new function, or new endpoint — nothing else was
  removed, renamed, or had its default behavior changed.

---

## 10. Project structure

```
adversarial-ml-playground/
├── .env.example                docker-compose environment template
├── docker-compose.yml          brings up both services together
├── .github/workflows/
│   └── ml-security-gate.yml    pytest + security_gate.py on every backend change
├── backend/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt
│   ├── scripts/
│   │   └── security_gate.py    MLSecOps CI gate CLI
│   ├── tests/                  pytest suite (see section 8)
│   └── app/
│       ├── main.py                    FastAPI app + all API routes
│       ├── schemas.py                 Pydantic request/response models
│       ├── storage.py                 SQLite experiment history
│       ├── threat_model.py            capability x attack applicability engine
│       ├── data/
│       │   ├── mnist_loader.py        8x8 digit dataset (sklearn, no download)
│       │   ├── cyber_loader.py        synthetic network-intrusion dataset
│       │   └── unsw_nb15_loader.py    real UNSW-NB15 network-intrusion dataset (needs setup, §5b)
│       ├── models/
│       │   ├── torch_models.py        LogisticRegression / SmallNN / CNN (PyTorch)
│       │   └── registry.py            training, in-memory model store, RF surrogate logic
│       ├── attacks/
│       │   ├── fgsm.py, pgd.py, deepfool.py, random_noise.py
│       │   ├── poisoning.py           label-flip and backdoor-trigger primitives
│       │   ├── blackbox.py            surrogate-transfer and zeroth-order query attacks
│       │   └── runner.py              dispatches by attack name, handles RF transfer attacks
│       ├── defenses/
│       │   ├── preprocessing.py       gaussian smoothing / clipping / normalization
│       │   └── adversarial_training.py    retrains a hardened model
│       ├── evaluation/
│       │   ├── robustness.py          batch evaluation + Aegis Robustness Index + Lp norms
│       │   ├── robustness_matrix.py   models x attacks systematic benchmark table
│       │   ├── explainability.py      perturbation heatmaps, tabular feature deltas
│       │   ├── norms.py               L0/L1/L2/Linf perturbation norms, confidence shift
│       │   ├── poisoning_eval.py      orchestrates clean-vs-poisoned training experiments
│       │   ├── transferability.py     cross-model adversarial transferability matrix
│       │   ├── risk_engine.py         risk rating, findings generator, recommendations
│       │   └── leaderboard.py         ranks models by latest robustness evaluation
│       └── experiments/
│           └── runner.py              capped grid-sweep engine
└── frontend/
    ├── Dockerfile
    ├── .dockerignore
    ├── package.json
    ├── lib/api.js               fetch wrapper for the backend
    ├── components/              shared UI (Sidebar, Panel, ModelPicker, charts, etc.)
    └── app/                     Next.js App Router pages: dashboard, models, attack-lab
                                  (incl. threat model + black-box lab), poisoning-lab,
                                  defense-lab, robustness-report, security-assessment,
                                  leaderboard, experiments
```

---

## 11. Extending it further

- **Real MNIST**: swap `mnist_loader.load_image_dataset` for a
  `torchvision.datasets.MNIST` loader if you want real 28×28 images instead of the 8×8
  stand-in — the rest of the pipeline doesn't assume a specific resolution.
- **CIC-IDS2017 (a second real dataset)**: UNSW-NB15 is wired up (§5b); CIC-IDS2017 wasn't
  added alongside it in this pass — deliberately, see the "one real dataset, done properly"
  note at the end of §5b. The same loader contract (`unsw_nb15_loader.py`'s function
  signature) is what a second loader would need to match.
- **Async experiment grid**: the grid runner and robustness matrix are currently
  synchronous and capped (60 configurations, 6 models x 6 attacks respectively) so a
  single HTTP request doesn't time out. A background job queue (Celery, RQ, or even a
  simple thread + polling endpoint) would remove those caps.
- **Membership inference / model extraction**: not implemented — the threat-model and
  evaluation architecture (see `app/evaluation/`) is set up so a new attack category is
  "add a module, register it, add an endpoint," following the same pattern as poisoning
  or black-box attacks.
- **PDF report export**: `/api/experiments/export` currently returns JSON; a PDF
  renderer (e.g. `reportlab` or `weasyprint` over an HTML template) could sit in front of
  the same data.
- **Resume line**: *"Engineered AegisML, an adversarial ML security assessment framework
  evaluating multiple model architectures — including against a real published
  intrusion-detection dataset, not only synthetic data — across evasion, poisoning, and
  black-box attack surfaces; implemented threat-model-aware attack gating, cross-model
  transferability analysis, a systematic robustness benchmark matrix, and an automated
  robustness regression gate integrated into CI."*

---

## 12. Deploying with Docker

The whole app (backend + frontend) can be brought up with Docker Compose. This is the
recommended path — it's portable to any VPS or cloud container platform, and sidesteps
free-tier PaaS limits that sometimes choke on PyTorch's install size.

```bash
cp .env.example .env
# edit .env — see the two variables below
docker compose up --build -d
docker compose logs -f
```

**Two things you must set correctly, or nothing will work once this leaves your laptop:**

1. **`NEXT_PUBLIC_API_URL`** (in `.env`, used at frontend *build* time) — this has to be a
   URL the **end user's browser** can reach, not a Docker-internal address. The frontend
   pages call the API directly from client-side JavaScript (`lib/api.js`), so
   `http://backend:8000` (the Docker Compose service name) will not work even though it
   resolves fine container-to-container. For local testing, `http://localhost:8000` is
   correct. For a real deployment, use your server's public IP/domain, e.g.
   `http://203.0.113.10:8000` or `https://api.yourdomain.com` behind a reverse proxy.
   **Changing this requires rebuilding the frontend image** (`docker compose up --build`)
   — `NEXT_PUBLIC_*` variables are inlined into the JS bundle at build time, not read at
   container start.
2. **`ALLOWED_ORIGINS`** (in `.env`, read by the backend at runtime) — comma-separated list
   of origins allowed to call the API from a browser. Defaults to `*` if unset, which is
   fine for a quick local test but should be locked down to your actual frontend URL
   (e.g. `https://yourdomain.com`) before this is exposed publicly.

**On a single VPS** (DigitalOcean, Hetzner, a plain EC2 box, etc.): install Docker, clone
the repo, set `.env` to the server's public IP or domain, `docker compose up --build -d`.
Put a reverse proxy (Caddy or nginx) in front for a real domain + free TLS if you want one
— Caddy in particular needs almost no config for a straightforward two-service setup like
this.

**On a container platform** (Railway, Fly.io, Render, etc.): each service can generally be
deployed by pointing the platform at `backend/Dockerfile` and `frontend/Dockerfile`
separately (most of these platforms deploy one Dockerfile per service, not a whole compose
file) — set `NEXT_PUBLIC_API_URL` to whatever public URL the platform assigns your backend
service, and set `ALLOWED_ORIGINS` to whatever public URL it assigns your frontend service.

**Remember:** trained models and the experiment history live in the backend process's
memory / a local SQLite file (see the "Migration notes" section above) — a container
restart or redeploy resets both. That's an intentional simplicity trade-off for a
demo/portfolio project, not a bug; swap in persistent storage (see "Extending it further")
if you need state to survive restarts.



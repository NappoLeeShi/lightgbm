# LightGBM Case Study

LightGBM experiments: a NumPy reimplementation of LightGBM plus a case-study notebook trained on a real-world dataset.

## Dataset

The case study uses the New York City taxi-trip duration dataset:

- Source: <https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration/data>
- Where to put it: `data/NYC.csv` (project root `data/` directory)
- File name: `NYC.csv`

Download it from the link above, save the CSV as `data/NYC.csv`, and it will be
picked up by `main.ipynb`. The file has been added to `.gitignore` because it
is very heavy and should not be committed or pushed.

## Directory layout

```
.
├── data/                        # Datasets
│   ├── NYC.csv                  # New York City taxi-trip dataset — THE current case study
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Old Telco churn dataset — IGNORED
├── main.ipynb                   # NYC case-study notebook (formerly main_nyc.ipynb)
├── IGNOREME_old_case_study.ipynb  # Old Telco churn case study — IGNORED
├── LightGBMLibrary/             # NumPy-only reimplementation of LightGBM
│   ├── lgbm_numpy/              # The library package
│   │   ├── api/                 # Public training/inference API (train, predict, ...)
│   │   ├── boosting/            # GBM boosting engine
│   │   ├── dataset/             # Dataset handling / construction
│   │   ├── engine/              # Callbacks and training loop
│   │   ├── histogram/           # Histogram binning used for splits
│   │   ├── metrics/             # Evaluation metrics
│   │   ├── objective/           # Loss functions (regression, classification)
│   │   ├── tree/                # Tree / node structures
│   │   └── model.py            # Model wrapper
│   ├── training/main.ipynb     # Notebook used to build/test the library
│   ├── tests/test_smoke.py     # Smoke tests
│   ├── dataset.csv              # Copy of the (old) Telco dataset
│   └── churn_model.json         # Serialized trained model
├── prompts_and_sessions/        # Exported opencode sessions (see below)
└── .venv/                       # Python virtual environment
```

## main.ipynb (the NYC case study)

`main.ipynb` in the project root is the active case study. Inside the exported
session `prompts_and_sessions/session-case-study.json` this same notebook was
still called `main_nyc.ipynb`; it was renamed to `main.ipynb`, and the old
Telco notebook (`IGNOREME_old_case_study.ipynb`) was kept as a duplicate copy
so it could be fixed and aligned with the new `data/NYC.csv` dataset. Use
`data/NYC.csv` — ignore the old Telco churn files.

## Sessions

The session files in `prompts_and_sessions/` are JSON exports of opencode
sessions. There are two:

- `session-case-study.json` — the NYC model-training session (worked on
  `main_nyc.ipynb`).
- `session-lightgbm-lib.json` — the session that built the NumPy LightGBM
  framework.

Export a session to JSON:

```bash
opencode export <sessionID>
```

See `opencode session list` to find session IDs.

### Importing a session back into opencode

```bash
opencode import prompts_and_sessions/session-case-study.json
opencode import prompts_and_sessions/session-lightgbm-lib.json
```

This restores the full conversation (messages, commands, tool output) so you
can browse or continue it. You can also feed the share URL directly to
`opencode import` (the LightGBM-lib session has one at
https://opncd.ai/share/bSWhBGpR).
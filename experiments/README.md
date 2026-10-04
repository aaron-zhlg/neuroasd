# Experiments

Each experiment gets a **folder** here with frozen config and results (safe to cite in papers).

| Folder | Description | Git tag |
|--------|-------------|---------|
| `baseline_gcn_v1/` | 2-layer GCN, random 80/20 split | `baseline-gcn-v1` |
| `loso_cv_gcn_v1/` | Leave-one-site-out CV (20 sites) | — |
| `linear_baseline_dcan_v1/` | Tangent / Fisher-z ridge on DCAN ABIDE I + II (1,443 subjects); leave-one-site-out and leave-one-institution-out. **Reference for the ABIDE I + II dataset.** | `linear-baseline-dcan-v1` |

## Convention

```text
experiments/<experiment_name>/
├── README.md        # methods, metrics, interpretation
├── run_config.json  # hyperparameters & data settings
├── results.json     # metrics, confusion matrix
└── figures/         # GNN attribution plots from ./scripts/run_explain.sh

outputs/<experiment_name>/   # local only (gitignored)
├── final_model.pt   # the reported model
├── best_model.pt    # diagnostic only; selected on the evaluation set
├── train.log
└── eval.log
```

## Reporting rule

Report the **final epoch**. Selecting the epoch that scores best on the evaluation set
uses that set twice, once to choose and once to report, which inflates the result. On
this dataset the inflation was worth about 0.07 AUC under LOSO.

## Branches vs folders

| Use | For |
|-----|-----|
| **Git branch** | Code changes (e.g. `experiment/loso-cv`) |
| **`experiments/` folder** | Frozen results & config for paper |
| **`outputs/` folder** | Model checkpoints & logs (local) |

When an experiment is done: commit results to its folder on `main`, tag if it's a reference baseline.

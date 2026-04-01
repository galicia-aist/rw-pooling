# Graph Classification Experiment

Run experiments with configurable datasets, training settings, and pooling methods.

---

## 🚀 Usage

```bash
python train.py [OPTIONS]
```

---

## ⚙️ Arguments

| Argument         | Type  | Default                | Description                                |
| ---------------- | ----- | ---------------------- | ------------------------------------------ |
| `--dataset`      | str   | `PROTEINS`             | Dataset to use for training and evaluation |
| `--root`         | str   | `data`                 | Root directory for datasets                |
| `--methods`      | str   | `all`                  | Pooling methods to evaluate                |
| `--runs`         | int   | `10`                   | Number of independent runs                 |
| `--epochs`       | int   | `200`                  | Number of training epochs                  |
| `--batch-size`   | int   | `32`                   | Batch size                                 |
| `--hidden`       | int   | `64`                   | Hidden dimension size                      |
| `--lr`           | float | `1e-3`                 | Learning rate                              |
| `--weight-decay` | float | `0.0`                  | Weight decay (L2 regularization)           |
| `--pool-ratio`   | float | `0.5`                  | Fraction of nodes kept during pooling      |
| `--max-nodes`    | int   | `None`                 | Maximum nodes per graph                    |
| `--quantile`     | float | `0.95`                 | Quantile threshold for pooling             |
| `--seed`         | int   | `0`                    | Random seed                                |
| `--log-every`    | int   | `20`                   | Logging frequency (epochs)                 |
| `--out-csv`      | str   | `results_proteins.csv` | Output CSV file                            |

---

## 📊 Example

```bash
python train.py --dataset PROTEINS --methods mean --runs 5 --epochs 100
```

---

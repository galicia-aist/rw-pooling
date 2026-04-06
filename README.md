# Abstract

Graph pooling plays a crucial role in graph neural networks (GNNs) by enabling hierarchical representation learning through graph coarsening. Existing pooling methods, such as Top-K, SAGPool, and DiffPool, can be interpreted within the Select–Reduce–Connect (SRC) framework, where their primary differences lie in the implementation of the selection function. In this work, we investigate whether the learned selection mechanisms commonly used in these methods are necessary for effective pooling. Inspired by recent advances in random-weight GNNs, we introduce randomized pooling strategies that replace learned selection functions with stochastic counterparts. Specifically, we explore both selection-based and clustering-based pooling, along with their random variants based on uniform node sampling and CountSketch-inspired clustering. Through experiments on standard graph classification benchmarks, we analyze the impact of randomness on pooling performance and assess its viability as a lightweight alternative to learned pooling operators.

---

## 🚀 Usage

```bash
python main.py [OPTIONS]
```

---

## ⚙️ Arguments

| Argument         | Type  | Default        | Description                                                                                                                      |
|------------------|-------|----------------|----------------------------------------------------------------------------------------------------------------------------------|
| `--dataset`      | str   | `PROTEINS`     | Dataset (`ENZYMES`, `IMDB-BINARY`, `MUTAG`, `PROTEINS`, `REDDIT-BINARY`, `NCI1`, `NCI109`, `PTC_MR`, `FRANKENSTEIN`, `ogbg-ppa`) |
| `--pmethod`      | str   | `asap`         | Pooling method (`mean`, `uniform`, `topk`, `sag`, `diffpool`, `countsketch`, `sum`, `asap`, `max`, `edge`)                       |
| `--root`         | str   | `data`         | Root directory for datasets                                                                                                      |
| `--runs`         | int   | `1`            | Number of independent runs                                                                                                       |
| `--epochs`       | int   | `200`          | Number of training epochs                                                                                                        |
| `--batch-size`   | int   | `32`           | Batch size                                                                                                                       |
| `--hidden`       | int   | `64`           | Hidden dimension size                                                                                                            |
| `--lr`           | float | `1e-3`         | Learning rate                                                                                                                    |
| `--weight-decay` | float | `0.0`          | Weight decay (L2 regularization)                                                                                                 |
| `--pool-ratio`   | float | `0.5`          | Fraction of nodes kept during pooling                                                                                            |
| `--max-nodes`    | int   | `None`         | Maximum nodes per graph                                                                                                          |
| `--quantile`     | float | `0.95`         | Quantile threshold for pooling                                                                                                   |
| `--seed`         | int   | `0`            | Random seed                                                                                                                      |
| `--log-every`    | int   | `20`           | Logging frequency (epochs)                                                                                                       |
| `--exp_name`     | str   | `lorem`        | Experiment name (used for logs/results directories)                                                                              |
| `--log_path`     | str   | `local`        | Path to store logs                                                                                                               |
| `--log_level`    | str   | `info`         | Logging level (`debug`, `info`, `warning`, `error`, `critical`)                                                                  |
---

## 📊 Example

```bash
python main.py --dataset PROTEINS --pmethod mean --runs 5 --epochs 100 --exp_name baseline_exp
```

---

## 🧠 Notes

- Supported pooling methods:
  - `mean` → global mean pooling
  - `topk` → Top-K pooling
  - `sag` → SAGPool
  - `diffpool` → differentiable clustering pooling
  - `uniform` → random node selection (under development)
  - `countsketch` → random clustering pooling (under development)

These methods allow studying both **learned vs random** and **selection vs clustering** pooling strategies under a unified framework.

---

## 📚 References

- <a href="https://arxiv.org/pdf/2502.00190" target="_blank">
  <strong>On the Effectiveness of Random Weights in Graph Neural Networks</strong>
  </a>

- <a href="https://arxiv.org/pdf/2110.05292" target="_blank">
  <strong>Understanding Pooling in Graph Neural Networks</strong>
  </a>

- <a href="https://arxiv.org/pdf/2204.07321" target="_blank">
  <strong>Graph Pooling for Graph Neural Networks: Progress, Challenges, and Opportunities</strong>
  </a>
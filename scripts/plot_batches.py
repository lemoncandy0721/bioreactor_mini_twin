"""생성된 배치 데이터를 그래프로 확인한다 (측정점 + 정답 곡선).

사용법:  python scripts/plot_batches.py [--data data/virtual] [--out figures/virtual_batches.png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PANELS = [
    ("X", "VCD [1e6 cells/mL]"),
    ("G", "Glucose [mM]"),
    ("L", "Lactate [mM]"),
    ("P", "Titer [mg/L]"),
]
COLORS = {"batch": "#6b7280", "continuous": "#2563eb", "bolus": "#d97706"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "virtual")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "virtual_batches.png")
    args = ap.parse_args()

    meas = pd.read_csv(args.data / "batches.csv")
    truth = pd.read_csv(args.data / "truth.csv")
    design = pd.read_csv(args.data / "design.csv").set_index("batch_id")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (col, label) in zip(axes.flat, PANELS):
        for bid, g in truth.groupby("batch_id"):
            c = COLORS[design.loc[bid, "strategy"]]
            ax.plot(g["time_h"] / 24, g[col], color=c, lw=1, alpha=0.7)
        for bid, g in meas.groupby("batch_id"):
            c = COLORS[design.loc[bid, "strategy"]]
            ax.scatter(g["time_h"] / 24, g[col], color=c, s=10)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Day")
    handles = [plt.Line2D([], [], color=c, lw=2, label=s) for s, c in COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Virtual plant: line = truth, dot = noisy daily sample", y=0.95, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"그래프 저장 → {args.out}")


if __name__ == "__main__":
    main()

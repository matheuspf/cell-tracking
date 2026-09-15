"""Portable scientific figure from completed refinement-study measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.cellpose_refine.common import RESULTS, WORK


def main():
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    control = json.loads((RESULTS / "offset-control.json").read_text())
    methods = [("Cellpose baseline", metrics["baseline"], "#666666", "-", "o"),
               ("Learned head · seed 20260914", metrics["seeds"]["20260914"], "#b45309", "-", "s"),
               ("Learned head · seed 314159", metrics["seeds"]["314159"], "#b45309", "--", "^"),
               ("Source-only constant offset", control["groups"], "#1d4ed8", "-", "o")]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True)
    for ax, encoding in zip(axes, ("integer", "float")):
        for label, groups, color, linestyle, marker in methods:
            row = next(r for r in groups[encoding] if r["embryo"] == "pooled")
            ax.plot(range(1, 8), [100 * row["recall"][f"{r}.0"] for r in range(1, 8)],
                    label=label, color=color, linestyle=linestyle, marker=marker, markersize=4, linewidth=1.7)
        ax.set(title="Integer export" if encoding == "integer" else "Float diagnostic",
               xlabel="Maximum matching distance (µm)", xlim=(.8, 7.2), ylim=(0, 100), xticks=range(1, 8))
        ax.grid(axis="y", color="0.9")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Annotated nodes matched (%)")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Cellpose refinement: recall at 1–7 µm", fontsize=14)
    fig.text(.07, .02,
             "14 Sep 2026 · 400 fixed frames, 40 clips, 2,383 sparse GT nodes · Opposite-embryo fitting · Same 156,226 candidates\n"
             "Both embryos previously examined; inherited cpDINO exposure unresolved. Constant control tested after the head study.",
             fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, .09, 1, .96))
    fig.savefig(RESULTS / "recall-curves.svg")
    fig.savefig(WORK / "recall-curves.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()

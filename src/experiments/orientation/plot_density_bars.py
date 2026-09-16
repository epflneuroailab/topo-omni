"""TDANN Fig. 2K-style pinwheel-density bars, two model families x three conditions.

Topo-Omni bars are computed here from our own maps. TDANN bars are READ OFF TDANN's
published Fig. 2K (their map data is not redistributed), so they are hatched and flagged
-- do not quote them as recomputed values.

CAVEAT, printed and annotated: pinwheel density does NOT separate our conditions. It
counts topological defects, and a random orientation map has many of them; what suppresses
the count in a disorganised map is the sd<30deg homogeneity gate, not an absence of
defects. Both the value and the ORDERING flip with the smoothing applied beforehand.
"""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# TDANN Fig. 2K, read off the published panel (see orientation_reference.FIGURE_REFERENCE)
TDANN_FIG = {"TDANN": 2.1, "Task Only": 0.6, "Unoptimized": 0.1}
MACAQUE_V1 = 3.0

ap = argparse.ArgumentParser()
ap.add_argument("--density-json", default="results/pinwheel_density_all.json")
ap.add_argument("--fwhm", default="fwhm3", choices=["fwhm0", "fwhm3"])
ap.add_argument("--out", default="results/pinwheel_density_bars.png")
args = ap.parse_args()

D = json.load(open(args.density_json))
ours = {k.split("|")[1]: v for k, v in D.items() if k.startswith(args.fwhm + "|")}

GROUPS = (
    ("Topo-Omni", ("#5B2C9E", "#9B7FC4", "#D9CEEA"), False,
     (("Topo-Omni", "Spatial\n+ Task"), ("Topo-Omni Task Only", "Task\nOnly"),
      ("Topo-Omni Untrained", "Untrained"))),
    ("TDANN", ("#C25E00", "#E3A669", "#F5E0C9"), True,
     (("TDANN", "Spatial\n+ Task"), ("Task Only", "Task\nOnly"),
      ("Unoptimized", "Untrained"))),
)

fig, ax = plt.subplots(figsize=(6.6, 4.3))
xt, xl, spans, x = [], [], [], 0.0
for gname, palette, figread, members in GROUPS:
    start = x
    for (key, blabel), colour in zip(members, palette):
        val = TDANN_FIG[key] if figread else ours[key]["density"]
        ax.bar(x, val, width=0.82, color=colour, edgecolor="0.25", linewidth=0.7,
               hatch="//" if figread else None)
        if not figread:
            n = ours[key]["total"]
            ax.text(x, val + 0.06, f"{ours[key]['pos']}+/{ours[key]['neg']}−",
                    ha="center", fontsize=6.6, color="0.35")
        xt.append(x); xl.append(blabel); x += 1.0
    spans.append((gname, start, x - 1.0)); x += 1.0

ax.axhline(MACAQUE_V1, ls="--", lw=1.6, color="#1B7F3B")
ax.text(x - 0.55, MACAQUE_V1, " V1*", color="#1B7F3B", va="center",
        fontsize=9, weight="bold")

ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7.8)
ax.set_xlim(-0.7, x - 0.3)
ax.set_ylabel("Pinwheels / Column Spacing$^2$")
ax.set_title(f"Pinwheel density  ({args.fwhm.replace('fwhm', 'fwhm=')} mm)", fontsize=10)
ax.set_ylim(0, max(4.4, max([ours[k]["density"] for k in ours] + list(TDANN_FIG.values()),
                            default=1) * 1.25))
sns.despine(ax=ax)
for gname, lo, hi in spans:
    ax.annotate("", xy=(lo - 0.4, -0.155), xytext=(hi + 0.4, -0.155),
                xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-", lw=1.1, color="0.3"))
    ax.text(0.5 * (lo + hi), -0.20, gname, ha="center", va="top", fontsize=9.5,
            weight="bold", transform=ax.get_xaxis_transform())

notes = [
    "Hatched TDANN bars and the V1* line are READ OFF TDANN Fig. 2K -- their map data is "
    "not redistributed, so these\n  are approximate, not recomputed. Topo-Omni bars are "
    "computed from our maps (counts shown as +/− above each bar).",
    "Pinwheel density does NOT separate the Topo-Omni conditions: a random map is full of "
    "defects, and the count is\n  gated by local homogeneity (sd<30°), so both the "
    "value and the ordering move with the smoothing. At fwhm=0 the\n  same three models "
    "give 0.10 / 0.13 / 6.90. Treat this panel as descriptive.",
]
for j, note in enumerate(notes):
    ax.text(0.0, -0.30 - j * 0.155, note, transform=ax.transAxes, fontsize=6.3,
            color="0.35", va="top")

fig.tight_layout()
fig.savefig(args.out, dpi=300, bbox_inches="tight")
fig.savefig(args.out.replace(".png", ".pdf"), bbox_inches="tight")
print("wrote", args.out)
for gname, _, figread, members in GROUPS:
    for key, lbl in members:
        v = TDANN_FIG[key] if figread else ours[key]["density"]
        print(f"  {gname:<10} {lbl.replace(chr(10),' '):<14} {v:5.2f}"
              + ("  (figure-read)" if figread else
                 f"  ({ours[key]['total']} pinwheels)"))

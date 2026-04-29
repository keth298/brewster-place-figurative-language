import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

merged = pd.read_csv("merged_output.csv")
correlations = pd.read_csv("correlations_output.csv")

CHARS = merged["character"].tolist()
TRAIT_LABELS = {
    "racial_consciousness": "Racial\nConsciousness",
    "emotional_register": "Emotional\nRegister",
    "domestic_score": "Domestic\nScore",
}


# ── Chart 1: Figurative Frequency (stacked horizontal bar) ──────────────────

fig, ax = plt.subplots(figsize=(9, 5))

y = range(len(CHARS))
bar_h = 0.5

bars_sim = ax.barh(list(y), merged["simile_count"], height=bar_h,
                   color="#4C72B0", label="Similes")
bars_met = ax.barh(list(y), merged["metaphor_count"], height=bar_h,
                   left=merged["simile_count"],
                   color="#DD8452", label="Metaphors")

for i, row in merged.iterrows():
    total = row["simile_count"] + row["metaphor_count"]
    density = row["figurative_per_1000_words"]
    ax.text(total + 0.3, i, f"{total}  ({density:.1f}/1k)",
            va="center", fontsize=9, color="#333333")

ax.set_yticks(list(y))
ax.set_yticklabels(CHARS, fontsize=10)
ax.set_xlabel("Figurative Language Count", fontsize=10)
ax.set_title("Figurative Language by Character\n(Similes + Metaphors)", fontsize=12, pad=12)
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.set_xlim(0, merged[["simile_count", "metaphor_count"]].sum(axis=1).max() * 1.35)

plt.tight_layout()
plt.savefig("figurative_frequency_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figurative_frequency_chart.png")


# ── Chart 2: Trait Scores (grouped bar, normalized 0-1 per trait) ───────────

import numpy as np

traits = list(TRAIT_LABELS.keys())
n_chars = len(CHARS)
bar_w = 0.25
colors = ["#4C72B0", "#55A868", "#C44E52"]

# Normalize each trait independently to 0-1 so all three are visible
normalized = merged.copy()
for trait in traits:
    mn, mx = merged[trait].min(), merged[trait].max()
    normalized[trait] = (merged[trait] - mn) / (mx - mn) if mx > mn else 0.0

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(n_chars)

for j, (trait, color) in enumerate(zip(traits, colors)):
    offset = (j - 1) * bar_w
    ax.bar(x + offset, normalized[trait], width=bar_w,
           label=TRAIT_LABELS[trait], color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(CHARS, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("Normalized Score (0–1 per trait)", fontsize=10)
ax.set_title("Character Trait Scores (Normalized)", fontsize=12, pad=12)
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
plt.savefig("trait_scores_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trait_scores_chart.png")


# ── Chart 3: Pearson Correlations (horizontal bar) ──────────────────────────

fig, ax = plt.subplots(figsize=(7, 3.5))

labels = [TRAIT_LABELS[t] for t in correlations["trait"]]
r_vals = correlations["r"].tolist()
p_vals = correlations["p_value"].tolist()
colors_corr = ["#55A868" if r >= 0 else "#C44E52" for r in r_vals]

y = range(len(labels))
ax.barh(list(y), r_vals, height=0.5, color=colors_corr, alpha=0.85)
ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")

for i, (r, p) in enumerate(zip(r_vals, p_vals)):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    label = f"r = {r:.2f}, p = {p:.3f} {sig}".strip()
    # Always place label to the right of zero so it never overlaps the y-axis
    x_pos = max(r, 0) + 0.04
    ax.text(x_pos, i, label, va="center", fontsize=9, color="#333333", ha="left")

ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Pearson r", fontsize=10)
ax.set_title("Trait–Figurative Language Correlations", fontsize=12, pad=12)
ax.set_xlim(-0.7, 1.3)

plt.tight_layout()
plt.savefig("correlation_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved correlation_chart.png")

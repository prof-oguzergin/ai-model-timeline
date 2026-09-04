import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image

# Data: (Model, Company, Date, is_milestone)
data = [
    # OpenAI
    ("GPT-3.5", "OpenAI", "2022-11-30", True),
    ("GPT-4", "OpenAI", "2023-03-14", True),
    ("GPT-4 Turbo", "OpenAI", "2023-11-06", False),
    ("GPT-4o", "OpenAI", "2024-05-13", True),
    ("GPT-4o mini", "OpenAI", "2024-07-18", False),
    ("o1-preview", "OpenAI", "2024-09-12", True),
    ("o3-mini", "OpenAI", "2025-01-31", False),
    ("o3", "OpenAI", "2025-04-16", False),
    ("o4-mini", "OpenAI", "2025-04-16", False),
    ("GPT-5", "OpenAI", "2025-08-07", True),
    ("GPT-5.1", "OpenAI", "2025-11-12", False),
    ("GPT-5.2", "OpenAI", "2025-12-11", False),
    ("codex-1", "OpenAI", "2025-05-16", False),
    ("GPT-5.3-Codex", "OpenAI", "2026-02-05", True),

    # Google
    ("Gemini 1.0 Pro", "Google", "2023-12-06", True),
    ("Gemini 1.5 Pro", "Google", "2024-02-15", True),
    ("Gemini 1.5 Flash", "Google", "2024-05-14", False),
    ("Gemini 2.0 Flash", "Google", "2025-01-30", False),
    ("Gemini 2.5 Pro", "Google", "2025-03-25", True),
    ("Gemini 2.5 Flash", "Google", "2025-05-20", False),
    ("Gemini 3 Pro", "Google", "2025-11-18", True),
    ("Gemini 3 Flash", "Google", "2025-12-17", False),
    ("Gemini 3 Deep Think", "Google", "2026-02-12", True),

    # Anthropic
    ("Claude 2", "Anthropic", "2023-07-11", True),
    ("Claude 3 Opus", "Anthropic", "2024-03-04", True),
    ("Claude 3.5 Sonnet", "Anthropic", "2024-06-20", True),
    ("Claude 3.7 Sonnet", "Anthropic", "2025-02-24", False),
    ("Claude Opus 4", "Anthropic", "2025-05-22", True),
    ("Claude Sonnet 4", "Anthropic", "2025-05-22", False),
    ("Claude Opus 4.1", "Anthropic", "2025-08-05", False),
    ("Claude Sonnet 4.5", "Anthropic", "2025-09-29", False),
    ("Claude Opus 4.5", "Anthropic", "2025-11-24", True),
    ("Claude Opus 4.6", "Anthropic", "2026-02-05", True),

    # xAI
    ("Grok-1", "xAI", "2023-11-04", True),
    ("Grok-1.5", "xAI", "2024-03-29", False),
    ("Grok-2", "xAI", "2024-08-14", True),
    ("Grok-3", "xAI", "2025-02-17", True),
    ("Grok-3 Mini", "xAI", "2025-02-17", False),
    ("Grok-4", "xAI", "2025-07-09", True),
    ("Grok Code Fast", "xAI", "2025-08-28", False),
    ("Grok-4.1", "xAI", "2025-11-17", False),

    # Meta
    ("Llama 2", "Meta", "2023-07-18", True),
    ("Llama 3", "Meta", "2024-04-18", True),
    ("Llama 3.1", "Meta", "2024-07-23", False),
    ("Llama 3.2", "Meta", "2024-09-25", False),
    ("Llama 3.3", "Meta", "2024-12-06", False),
    ("Llama 4", "Meta", "2025-04-05", True),

    # Microsoft Phi (Open Source)
    ("Phi-1", "Microsoft", "2023-06-20", True),
    ("Phi-2", "Microsoft", "2023-12-12", True),
    ("Phi-3", "Microsoft", "2024-04-23", True),
    ("Phi-3.5", "Microsoft", "2024-08-20", False),
    ("Phi-4", "Microsoft", "2024-12-12", True),
    ("Phi-4-reasoning", "Microsoft", "2025-04-30", True),

    # Google Gemma (Open Source / Open Weight)
    ("Gemma 1", "Google Gemma", "2024-02-21", True),
    ("Gemma 2", "Google Gemma", "2024-06-27", True),
    ("Gemma 3", "Google Gemma", "2025-03-12", True),
    ("Gemma 3n", "Google Gemma", "2025-06-26", False),

    # Mistral
    ("Mistral 7B", "Mistral", "2023-09-27", True),
    ("Mixtral 8x7B", "Mistral", "2023-12-11", True),
    ("Mistral Large", "Mistral", "2024-02-26", True),
    ("Codestral", "Mistral", "2024-05-29", False),
    ("Mistral Large 2", "Mistral", "2024-07-24", False),
    ("Pixtral 12B", "Mistral", "2024-09-11", False),
    ("Mistral Medium 3", "Mistral", "2025-05-07", False),
    ("Mistral Small 3", "Mistral", "2025-01-30", False),
    ("Magistral", "Mistral", "2025-06-10", True),
    ("Mistral Large 3", "Mistral", "2025-12-02", True),

    # Qwen (Alibaba)
    ("Qwen 2", "Qwen", "2024-06-06", True),
    ("Qwen 2.5", "Qwen", "2024-09-19", True),
    ("QwQ-32B", "Qwen", "2025-03-05", False),
    ("Qwen 3", "Qwen", "2025-04-28", True),
    ("Qwen3-Coder", "Qwen", "2026-02-02", False),

    # DeepSeek
    ("DeepSeek-V2", "DeepSeek", "2024-05-06", True),
    ("DeepSeek-V3", "DeepSeek", "2024-12-26", True),
    ("DeepSeek-R1", "DeepSeek", "2025-01-20", True),
    ("DeepSeek-R1-0528", "DeepSeek", "2025-05-28", False),
    ("DeepSeek-V3.1", "DeepSeek", "2025-08-21", False),
    ("DeepSeek-V3.2", "DeepSeek", "2025-12-01", False),

    # Z.ai (Zhipu AI)
    ("GLM-4.5", "Z.ai", "2025-07-28", True),
    ("GLM-4.6", "Z.ai", "2025-09-30", False),
    ("GLM-4.7", "Z.ai", "2025-12-22", False),
    ("GLM-5", "Z.ai", "2026-02-11", True),

    # Kimi (Moonshot AI)
    ("Kimi K1.5", "Kimi", "2025-01-20", True),
    ("Kimi K2", "Kimi", "2025-07-11", True),
    ("Kimi K2 Thinking", "Kimi", "2025-11-06", False),
    ("Kimi K2.5", "Kimi", "2026-01-27", True),

    # MiniMax
    ("MiniMax-Text-01", "MiniMax", "2025-01-15", True),
    ("MiniMax-M1", "MiniMax", "2025-06-16", True),
    ("MiniMax-M2", "MiniMax", "2025-10-27", True),
    ("MiniMax-M2.1", "MiniMax", "2025-12-23", False),
    ("MiniMax-M2.5", "MiniMax", "2026-02-12", True),
]

df = pd.DataFrame(data, columns=["Model", "Company", "Date", "Milestone"])
df["Date"] = pd.to_datetime(df["Date"])

colors = {
    "OpenAI":    "#10a37f",
    "Google":    "#4285F4",
    "Anthropic": "#d97757",
    "xAI":       "#1DA1F2",
    "Meta":      "#0668E1",
    "Microsoft": "#F25022",
    "Mistral":   "#FF7000",
    "Qwen":      "#6C3CE1",
    "DeepSeek":  "#00B4D8",
    "Z.ai":      "#00C853",
    "Kimi":      "#FF4D6D",
    "MiniMax":   "#C77DFF",
    "Google Gemma": "#34A853",
}

company_name = {
    "OpenAI":    "OpenAI",
    "Google":    "Google",
    "Anthropic": "Anthropic",
    "xAI":       "xAI",
    "Meta":      "Meta",
    "Microsoft": "Microsoft",
    "Mistral":   "Mistral",
    "Qwen":      "Qwen / Alibaba",
    "DeepSeek":  "DeepSeek",
    "Z.ai":      "Z.ai / Zhipu",
    "Kimi":      "Kimi / Moonshot",
    "MiniMax":   "MiniMax",
    "Google Gemma": "Google Gemma",
}

country_text = {
    "OpenAI":    "United States",
    "Google":    "United States",
    "Anthropic": "United States",
    "xAI":       "United States",
    "Meta":      "United States",
    "Microsoft": "United States",
    "Mistral":   "France",
    "Qwen":      "China",
    "DeepSeek":  "China",
    "Z.ai":      "China",
    "Kimi":      "China",
    "MiniMax":   "China",
    "Google Gemma": "United States",
}

# Flag image files
flag_images = {
    "United States": "C:/Users/Z GAMES/flags/us.png",
    "France":        "C:/Users/Z GAMES/flags/fr.png",
    "China":         "C:/Users/Z GAMES/flags/cn.png",
}

# Order: Closed source on top, then separator, then open source below
# Microsoft Phi goes in open source section (after Meta, before Google Gemma)
company_order = ["OpenAI", "Google", "Anthropic", "xAI", "Meta", "Microsoft", "Google Gemma", "Mistral",
                 "Qwen", "DeepSeek", "Z.ai", "Kimi", "MiniMax"]

# Vertical spacing: each company gets 3.5 units for more room
y_positions = {c: (len(company_order) - 1 - i) * 3.5 for i, c in enumerate(company_order)}

# Dark theme
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(60, 53))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# =========================================
# BACKGROUND SHADING for closed/open source
# =========================================
separator_y = (y_positions["xAI"] + y_positions["Meta"]) / 2
y_min = y_positions["MiniMax"] - 1.8
y_max = y_positions["OpenAI"] + 1.8

# Shading starts at the first model date (not covering Y-axis labels)
shade_left = mdates.date2num(df["Date"].min() - pd.Timedelta(days=15))
shade_right = mdates.date2num(df["Date"].max() + pd.Timedelta(days=80))
shade_width = shade_right - shade_left

# Closed source: warm red/purple tint
closed_rect = mpatches.FancyBboxPatch(
    (shade_left, separator_y),
    shade_width,
    y_max - separator_y,
    boxstyle="round,pad=0", fc="#2a1a2a", ec="none", alpha=0.5, zorder=0)
ax.add_patch(closed_rect)

# Open source: cool green/teal tint
open_rect = mpatches.FancyBboxPatch(
    (shade_left, y_min),
    shade_width,
    separator_y - y_min,
    boxstyle="round,pad=0", fc="#1a2a1a", ec="none", alpha=0.5, zorder=0)
ax.add_patch(open_rect)

# Section labels - large watermark-style, more visible
ax.text(df["Date"].min() + pd.Timedelta(days=30), y_positions["MiniMax"] + 0.5,
        "OPEN SOURCE / OPEN WEIGHT",
        fontsize=56, color="#3fb950", va="center", ha="left",
        fontweight="bold", alpha=0.22)
ax.text(df["Date"].min() + pd.Timedelta(days=30), y_positions["xAI"] + 0.5,
        "CLOSED SOURCE",
        fontsize=56, color="#f778ba", va="center", ha="left",
        fontweight="bold", alpha=0.22)

labels_y = []

# Manual overrides: model_name -> (x_offset, y_offset)
manual_overrides = {
    # OpenAI
    "GPT-3.5":              (60, -75),
    "GPT-5.2":              (0, 55),
    "codex-1":              (0, -80),
    "GPT-5.3-Codex":        (0, -85),

    # Google
    "Gemini 1.0 Pro":       (0, -80),
    "Gemini 3 Pro":         (-60, -85),
    "Gemini 3 Flash":       (0, 75),
    "Gemini 3 Deep Think":  (60, -85),

    # DeepSeek
    "DeepSeek-V3.1":        (0, 75),

    # Mistral
    "Mixtral 8x7B":         (0, -75),

    # xAI - Grok-3 and Grok-3 Mini same day
    "Grok-3":               (-80, 65),
    "Grok-3 Mini":          (80, -75),

    # Meta - Llama 2 clashes with "OPEN SOURCE" text
    "Llama 2":              (0, -75),

    # Microsoft - Phi-2 and Phi-4 both in December, spread them
    "Phi-2":                (0, -75),

    # Anthropic - 10 models, very crowded row
    "Claude 2":             (0, 65),
    "Claude 3 Opus":        (0, -80),
    "Claude 3.5 Sonnet":    (0, 65),
    "Claude 3.7 Sonnet":    (0, -80),
    "Claude Opus 4":        (-100, 75),
    "Claude Sonnet 4":      (100, -85),
    "Claude Opus 4.1":      (0, 65),
    "Claude Sonnet 4.5":    (30, -90),
    "Claude Opus 4.5":      (0, 85),
    "Claude Opus 4.6":      (0, -100),
}


def smart_offsets(dates, milestones, names, n, company=""):
    """Calculate label offsets - bigger gaps for large fonts."""
    offsets = []
    dates_num = [d.timestamp() for d in dates]
    day_in_sec = 86400
    threshold = 55 * day_in_sec

    clusters = []
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs(dates_num[j+1] - dates_num[j]) < threshold:
            j += 1
        clusters.append((i, j))
        i = j + 1

    if company == "Anthropic":
        levels_up   = [60, 145, 230, 310]
        levels_down = [-75, -160, -245, -325]
    else:
        levels_up   = [60, 130, 200]
        levels_down = [-68, -140, -210]

    for start, end in clusters:
        cluster_size = end - start + 1
        if cluster_size == 1:
            model_name = names[start]
            if model_name in manual_overrides:
                offsets.append(manual_overrides[model_name][1])
            else:
                offsets.append(60 if milestones[start] else -68)
        else:
            up_idx = 0
            down_idx = 0
            for k in range(start, end + 1):
                model_name = names[k]
                if model_name in manual_overrides:
                    offsets.append(manual_overrides[model_name][1])
                elif k % 2 == 0:
                    offsets.append(levels_up[min(up_idx, len(levels_up)-1)])
                    up_idx += 1
                else:
                    offsets.append(levels_down[min(down_idx, len(levels_down)-1)])
                    down_idx += 1

    return offsets


for company in company_order:
    group = df[df["Company"] == company].sort_values("Date").reset_index(drop=True)
    y_pos = y_positions[company]
    labels_y.append((y_pos, company))

    # Timeline line
    ax.hlines(y_pos, df["Date"].min(), df["Date"].max(),
              color=colors[company], alpha=0.18, linewidth=3.5)

    # Dots
    for idx, (_, row) in enumerate(group.iterrows()):
        dot_size = 350 if row["Milestone"] else 140
        glow_size = 1000 if row["Milestone"] else 500
        ax.scatter(row["Date"], y_pos,
                   color=colors[company], s=glow_size, zorder=3, alpha=0.12)
        ax.scatter(row["Date"], y_pos,
                   color=colors[company], s=dot_size, zorder=4,
                   edgecolors="white",
                   linewidths=2.0 if row["Milestone"] else 0.5)

    # Offsets
    n = len(group)
    milestones_list = group["Milestone"].tolist()
    names_list = group["Model"].tolist()
    offsets_y = smart_offsets(group["Date"].tolist(), milestones_list, names_list, n, company=company)

    # Horizontal nudge for same-day or very close releases
    dates_num = [d.timestamp() for d in group["Date"].tolist()]
    offsets_x = [0] * n
    for i in range(n):
        model_name = names_list[i]
        if model_name in manual_overrides and manual_overrides[model_name][0] != 0:
            offsets_x[i] = manual_overrides[model_name][0]
            continue
    for i in range(1, n):
        if names_list[i] in manual_overrides and manual_overrides[names_list[i]][0] != 0:
            continue
        if names_list[i-1] in manual_overrides and manual_overrides[names_list[i-1]][0] != 0:
            continue
        gap_days = abs(dates_num[i] - dates_num[i-1]) / 86400
        if gap_days < 2:
            offsets_x[i-1] = -80
            offsets_x[i] = 80
        elif gap_days < 40:
            offsets_x[i-1] = offsets_x[i-1] - 30
            offsets_x[i] = offsets_x[i] + 30

    for idx, (_, row) in enumerate(group.iterrows()):
        is_ms = row["Milestone"]
        fsize = 28 if is_ms else 22
        pad = 0.7 if is_ms else 0.5
        alpha = 0.95 if is_ms else 0.72
        ec = "white" if is_ms else "none"
        ew = 2.5 if is_ms else 0

        ax.annotate(row["Model"],
                    (row["Date"], y_pos),
                    xytext=(offsets_x[idx], offsets_y[idx]),
                    textcoords="offset points",
                    ha="center", va="center",
                    fontsize=fsize,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle=f"round,pad={pad}",
                              fc=colors[company],
                              ec=ec,
                              linewidth=ew,
                              alpha=alpha),
                    arrowprops=dict(arrowstyle="-",
                                   color=colors[company],
                                   alpha=0.4,
                                   linewidth=1.5))

# Y axis - manual two-line labels
ax.set_yticks([y[0] for y in labels_y])
ax.set_yticklabels([""] * len(labels_y))

date_min = df["Date"].min()
x_label_pos = date_min - pd.Timedelta(days=30)

for y_pos, company in labels_y:
    # Company name - big, bold, colored
    ax.text(x_label_pos, y_pos + 0.5, company_name[company],
            fontsize=36, fontweight="bold", color=colors[company],
            ha="right", va="center")
    # Country name - below, shifted left to make room for flag
    country = country_text[company]
    flag_x_offset = date_min - pd.Timedelta(days=65)
    ax.text(mdates.date2num(flag_x_offset), y_pos - 0.55, country,
            fontsize=28, fontweight="normal", color="#8b949e",
            ha="right", va="center", fontstyle="italic")

    # Place real flag image to the right of country text
    flag_path = flag_images[country]
    flag_img = plt.imread(flag_path)
    imagebox = OffsetImage(flag_img, zoom=0.45)
    flag_date = date_min - pd.Timedelta(days=45)
    ab = AnnotationBbox(imagebox, (mdates.date2num(flag_date), y_pos - 0.55),
                        frameon=False, zorder=10,
                        clip_on=False, annotation_clip=False)
    ax.add_artist(ab)

# X axis
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_xlim(mdates.date2num(pd.Timestamp("2022-11-01")), mdates.date2num(pd.Timestamp("2026-03-15")))
plt.xticks(rotation=45, fontsize=22, color="#8b949e")

# Grid
ax.grid(True, axis="x", linestyle="--", alpha=0.06, color="white")

# Year markers - vertical dashed lines at Jan 1 of each year
for year in [2023, 2024, 2025, 2026]:
    year_date = mdates.date2num(pd.Timestamp(f"{year}-01-01"))
    ax.axvline(x=year_date, color="#58a6ff", linewidth=1.5, linestyle=":", alpha=0.4, zorder=1)
    ax.text(year_date, y_max - 0.2, str(year),
            fontsize=32, fontweight="bold", color="#58a6ff", alpha=0.5,
            ha="center", va="top")

# Separator line
ax.axhline(y=separator_y, color="#30363d", linewidth=2.5, linestyle="-", alpha=0.6)

# Spines
for spine in ax.spines.values():
    spine.set_color("#30363d")
    spine.set_linewidth(0.5)

ax.tick_params(axis="y", colors="white", length=0, pad=20)
ax.tick_params(axis="x", colors="#8b949e", length=6, pad=15)

# Title
plt.title("AI Model Release Timeline",
          fontsize=64, pad=80, color="white", fontweight="bold")

ax.text(0.5, 1.013,
        "12 Companies  |  90+ Models  |  Nov 2022 – Feb 2026  |  ● Large = Milestone  |  ● Small = Update",
        transform=ax.transAxes, ha="center", fontsize=30,
        color="#8b949e", fontstyle="italic")

# Legend
legend_elements = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=colors[c], markersize=18,
                          label=company_name[c], linewidth=0)
                   for c in company_order]
legend_elements.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#888", markersize=22,
                              markeredgecolor="white", markeredgewidth=2,
                              label="Milestone", linewidth=0))
legend_elements.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#888", markersize=13,
                              label="Update", linewidth=0))

legend = ax.legend(handles=legend_elements, loc="upper center",
                   fontsize=20, framealpha=0.4, facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="white",
                   ncol=8, bbox_to_anchor=(0.5, -0.035),
                   columnspacing=1.8, handletextpad=0.8)

plt.margins(y=0.01, x=0.06)
ax.set_ylim(y_min, y_max)
plt.tight_layout(rect=[0.11, 0.03, 0.98, 0.98])
plt.savefig("C:/Users/Z GAMES/ai_timeline_microsoft_test.png", dpi=150, bbox_inches="tight",
            pad_inches=1.0, facecolor="#0d1117", edgecolor="none")
print("Microsoft test grafik kaydedildi!")

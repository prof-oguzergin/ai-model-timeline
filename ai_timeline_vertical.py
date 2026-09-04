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
    ("Gemini 3.1 Pro", "Google", "2026-02-19", True),

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
    ("Claude Sonnet 4.6", "Anthropic", "2026-02-17", True),

    # xAI
    ("Grok-1", "xAI", "2023-11-04", True),
    ("Grok-1.5", "xAI", "2024-03-29", False),
    ("Grok-2", "xAI", "2024-08-14", True),
    ("Grok-3", "xAI", "2025-02-17", True),
    ("Grok-3 Mini", "xAI", "2025-02-17", False),
    ("Grok-4", "xAI", "2025-07-09", True),
    ("Grok Code Fast", "xAI", "2025-08-28", False),
    ("Grok-4.1", "xAI", "2025-11-17", False),
    ("Grok 4.20", "xAI", "2026-02-17", True),

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
    ("Qwen 3.5", "Qwen", "2026-02-16", True),

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

    # Google Gemma (Open Source / Open Weight)
    ("Gemma 1", "Google Gemma", "2024-02-21", True),
    ("Gemma 2", "Google Gemma", "2024-06-27", True),
    ("Gemma 3", "Google Gemma", "2025-03-12", True),
    ("Gemma 3n", "Google Gemma", "2025-06-26", False),
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
    "Qwen":      "Qwen",
    "DeepSeek":  "DeepSeek",
    "Z.ai":      "Z.ai",
    "Kimi":      "Kimi",
    "MiniMax":   "MiniMax",
    "Google Gemma": "Gemma",
}

company_order = ["OpenAI", "Google", "Anthropic", "xAI", "Meta", "Microsoft", "Google Gemma", "Mistral",
                 "Qwen", "DeepSeek", "Z.ai", "Kimi", "MiniMax"]

# X positions: each company gets a column
x_positions = {c: i * 4.0 for i, c in enumerate(company_order)}

# Flag image files
flag_images = {
    "United States": "C:/Users/Z GAMES/flags/us.png",
    "France":        "C:/Users/Z GAMES/flags/fr.png",
    "China":         "C:/Users/Z GAMES/flags/cn.png",
}

country_map = {
    "OpenAI": "United States", "Google": "United States", "Anthropic": "United States",
    "xAI": "United States", "Meta": "United States", "Microsoft": "United States",
    "Mistral": "France", "Qwen": "China", "DeepSeek": "China",
    "Z.ai": "China", "Kimi": "China", "MiniMax": "China", "Google Gemma": "United States",
}

# =====================
# CLOSED / OPEN SOURCE boundary
# =====================
closed_source = ["OpenAI", "Google", "Anthropic", "xAI"]
open_source = ["Meta", "Microsoft", "Google Gemma", "Mistral", "Qwen", "DeepSeek", "Z.ai", "Kimi", "MiniMax"]

# Separator between xAI and Meta
separator_x = (x_positions["xAI"] + x_positions["Meta"]) / 2

# Dark theme
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(58, 72))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# Date range for Y axis (inverted: newest at bottom, oldest at top... actually top=old, bottom=new)
date_min = pd.Timestamp("2022-11-01")
date_max = pd.Timestamp("2026-04-01")

# Convert dates to numeric for Y axis
y_min_num = mdates.date2num(date_min)
y_max_num = mdates.date2num(date_max)

# X range
x_min = x_positions["OpenAI"] - 2.5
x_max = x_positions["MiniMax"] + 2.5

# =========================================
# BACKGROUND SHADING for closed/open source
# =========================================
# Closed source: left side (warm red/purple tint)
closed_rect = mpatches.FancyBboxPatch(
    (x_min, y_min_num),
    separator_x - x_min,
    y_max_num - y_min_num,
    boxstyle="round,pad=0", fc="#2a1a2a", ec="none", alpha=0.45, zorder=0)
ax.add_patch(closed_rect)

# Open source: right side (cool green/teal tint)
open_rect = mpatches.FancyBboxPatch(
    (separator_x, y_min_num),
    x_max - separator_x,
    y_max_num - y_min_num,
    boxstyle="round,pad=0", fc="#1a2a1a", ec="none", alpha=0.45, zorder=0)
ax.add_patch(open_rect)

# Separator vertical line
ax.axvline(x=separator_x, color="#30363d", linewidth=2.5, linestyle="-", alpha=0.6, zorder=1)

# Section labels - rotated watermark style
mid_closed_x = (x_positions["OpenAI"] + x_positions["xAI"]) / 2
mid_open_x = (x_positions["Meta"] + x_positions["MiniMax"]) / 2
mid_y = mdates.date2num(pd.Timestamp("2024-10-01"))

ax.text(mid_closed_x, mid_y, "CLOSED SOURCE",
        fontsize=60, color="#f778ba", va="center", ha="center",
        fontweight="bold", alpha=0.15, rotation=90)
ax.text(mid_open_x, mid_y, "OPEN SOURCE / OPEN WEIGHT",
        fontsize=60, color="#3fb950", va="center", ha="center",
        fontweight="bold", alpha=0.15, rotation=90)

# Year markers - horizontal dashed lines
for year in [2023, 2024, 2025, 2026]:
    year_num = mdates.date2num(pd.Timestamp(f"{year}-01-01"))
    ax.axhline(y=year_num, color="#58a6ff", linewidth=1.5, linestyle=":", alpha=0.35, zorder=1)
    ax.text(x_min + 0.3, year_num, str(year),
            fontsize=30, fontweight="bold", color="#58a6ff", alpha=0.5,
            ha="left", va="top")

# Grid
ax.grid(True, axis="y", linestyle="--", alpha=0.05, color="white")


def smart_label_offsets_vertical(dates_num, milestones, names, n):
    """Calculate horizontal label offsets for vertical layout.
    Labels alternate left/right of the vertical timeline line.
    Close dates get staggered more aggressively.
    """
    offsets_x = []
    offsets_y = []
    day_in_num = 1.0  # mdates: 1 unit = 1 day
    threshold = 40 * day_in_num

    # Cluster nearby dates
    clusters = []
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs(dates_num[j+1] - dates_num[j]) < threshold:
            j += 1
        clusters.append((i, j))
        i = j + 1

    levels_right = [90, 190, 280]
    levels_left = [-90, -190, -280]

    for start, end in clusters:
        cluster_size = end - start + 1
        if cluster_size == 1:
            if milestones[start]:
                offsets_x.append(levels_right[0])
                offsets_y.append(0)
            else:
                offsets_x.append(levels_left[0])
                offsets_y.append(0)
        else:
            right_idx = 0
            left_idx = 0
            for k in range(start, end + 1):
                # Alternate: even -> right, odd -> left
                if k % 2 == 0:
                    offsets_x.append(levels_right[min(right_idx, len(levels_right)-1)])
                    right_idx += 1
                else:
                    offsets_x.append(levels_left[min(left_idx, len(levels_left)-1)])
                    left_idx += 1
                # Small vertical nudge for very close dates
                local_idx = k - start
                if cluster_size > 2:
                    offsets_y.append((local_idx - cluster_size / 2) * 8)
                else:
                    offsets_y.append(0)

    return offsets_x, offsets_y


# Draw each company column
for company in company_order:
    group = df[df["Company"] == company].sort_values("Date").reset_index(drop=True)
    x_pos = x_positions[company]

    # Vertical timeline line
    ax.vlines(x_pos, y_min_num, y_max_num,
              color=colors[company], alpha=0.18, linewidth=3.5)

    # Convert dates to numeric
    dates_num = [mdates.date2num(d) for d in group["Date"].tolist()]
    milestones_list = group["Milestone"].tolist()
    names_list = group["Model"].tolist()
    n = len(group)

    # Get label offsets
    off_x, off_y = smart_label_offsets_vertical(dates_num, milestones_list, names_list, n)

    for idx, (_, row) in enumerate(group.iterrows()):
        y_val = mdates.date2num(row["Date"])
        is_ms = row["Milestone"]

        # Dots
        dot_size = 280 if is_ms else 110
        glow_size = 800 if is_ms else 400
        ax.scatter(x_pos, y_val,
                   color=colors[company], s=glow_size, zorder=3, alpha=0.12)
        ax.scatter(x_pos, y_val,
                   color=colors[company], s=dot_size, zorder=4,
                   edgecolors="white",
                   linewidths=2.0 if is_ms else 0.5)

        # Labels
        fsize = 19 if is_ms else 15
        pad = 0.6 if is_ms else 0.4
        alpha = 0.95 if is_ms else 0.72
        ec = "white" if is_ms else "none"
        ew = 2.0 if is_ms else 0

        ax.annotate(row["Model"],
                    (x_pos, y_val),
                    xytext=(off_x[idx], off_y[idx]),
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
                                   linewidth=1.2),
                    zorder=5)

# =====================
# X axis: company names at top
# =====================
ax.set_xticks([x_positions[c] for c in company_order])
ax.set_xticklabels([""] * len(company_order))
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

# Company name labels at top with flag
for company in company_order:
    x_pos = x_positions[company]
    y_label = y_max_num + 12  # above the chart

    # Company name
    ax.text(x_pos, y_label, company_name[company],
            fontsize=28, fontweight="bold", color=colors[company],
            ha="center", va="bottom", rotation=0)

    # Flag below company name
    country = country_map[company]
    flag_path = flag_images[country]
    flag_img = plt.imread(flag_path)
    imagebox = OffsetImage(flag_img, zoom=0.35)
    ab = AnnotationBbox(imagebox, (x_pos, y_label - 5),
                        frameon=False, zorder=10,
                        clip_on=False, annotation_clip=False)
    ax.add_artist(ab)

# Y axis: dates (top = newest, bottom = oldest)
ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_ylim(y_min_num, y_max_num)  # min at bottom, max at top -> newest on top

ax.set_xlim(x_min, x_max)

# Tick styling
ax.tick_params(axis="y", colors="#8b949e", labelsize=20, length=6, pad=10)
ax.tick_params(axis="x", colors="white", length=0, pad=25)

# Spines
for spine in ax.spines.values():
    spine.set_color("#30363d")
    spine.set_linewidth(0.5)

# Title
plt.title("AI Model Release Timeline (Vertical)",
          fontsize=54, pad=120, color="white", fontweight="bold",
          loc="center")

ax.text(0.5, 1.025,
        "13 Companies  |  90+ Models  |  Nov 2022 \u2013 Feb 2026  |  \u25cf Large = Milestone  |  \u25cf Small = Update",
        transform=ax.transAxes, ha="center", fontsize=24,
        color="#8b949e", fontstyle="italic")

# Legend at bottom
legend_elements = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=colors[c], markersize=16,
                          label=company_name[c], linewidth=0)
                   for c in company_order]
legend_elements.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#888", markersize=20,
                              markeredgecolor="white", markeredgewidth=2,
                              label="Milestone", linewidth=0))
legend_elements.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#888", markersize=12,
                              label="Update", linewidth=0))

legend = ax.legend(handles=legend_elements, loc="upper center",
                   fontsize=18, framealpha=0.4, facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="white",
                   ncol=8, bbox_to_anchor=(0.5, -0.015),
                   columnspacing=1.5, handletextpad=0.6)

plt.tight_layout(rect=[0.05, 0.02, 0.98, 0.95])
output_path = "G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi/ai_timeline_vertical.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight",
            pad_inches=1.0, facecolor="#0d1117", edgecolor="none")
print(f"Vertical timeline saved to: {output_path}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import os

# ============================================================
# Vending-Bench 2 leaderboard data
# Source: andonlabs.com
# 43 models, 1-year vending machine operation simulation
# Value: final balance including $500 starting capital
# ============================================================

# (Model name, company, release date, final balance $)
leaderboard = [
    ("Claude Opus 4.7",              "Anthropic", "2026-04-16", 10936.76),
    ("Claude Opus 4.8 High",         "Anthropic", "2026-05-28",  5787.43),
    ("Claude Opus 4.8 Max",          "Anthropic", "2026-05-28",  2992.34),
    ("Claude Fable 5 High",          "Anthropic", "2026-06-09",  5680.26),
    ("Claude Fable 5 Max",           "Anthropic", "2026-06-09",  4966.64),
    ("Claude Sonnet 5",              "Anthropic", "2026-06-30",  6377.70),
    ("Claude Opus 4.6",              "Anthropic", "2026-02-05",  8017.59),
    ("GPT-5.5",                      "OpenAI",    "2026-04-23",  7523.84),
    ("Claude Sonnet 4.6",            "Anthropic", "2026-02-17",  7204.14),
    ("Kimi K2.6",                    "Kimi",      "2026-04-13",  6204.57),
    ("Kimi K2.7 Code",               "Kimi",      "2026-06-12",  5082.94),
    ("GPT-5.4",                      "OpenAI",    "2026-03-05",  6144.18),
    ("GPT-5.3-Codex",                "OpenAI",    "2026-02-05",  5940.12),
    ("GLM-5.2",                      "Z.ai",      "2026-06-13",  8313.78),
    ("GLM-5.1",                      "Z.ai",      "2026-03-27",  5634.41),
    ("Gemini 3 Pro",                 "Google",    "2025-11-18",  5478.16),
    ("Gemini 3.5 Flash",             "Google",    "2026-05-19",  5396.42),
    ("Qwen 3.6 Plus",                "Qwen",      "2026-03-31",  5114.87),
    ("Claude Opus 4.5",              "Anthropic", "2025-11-24",  4967.06),
    ("Grok 4.20",                    "xAI",       "2026-02-17",  4662.85),
    ("GLM-5",                        "Z.ai",      "2026-02-11",  4432.12),
    ("Qwen 3.6 Max",                 "Qwen",      "2026-03-31",  4254.19),
    ("Claude Sonnet 4.5",            "Anthropic", "2025-09-29",  3838.74),
    ("Gemini 3.1 Pro Custom Tools",  "Google",    "2026-02-19",  3774.25),
    ("Gemini 3 Flash",               "Google",    "2025-12-17",  3634.72),
    ("GPT-5.2",                      "OpenAI",    "2025-12-11",  3591.33),
    ("Deepseek V4 Pro",              "DeepSeek",  "2026-04-24",  3284.52),
    ("GLM-4.7",                      "Z.ai",      "2025-12-22",  2376.82),
    ("GPT-5.1",                      "OpenAI",    "2025-11-12",  1473.43),
    ("Kimi K2.5",                    "Kimi",      "2026-01-27",  1198.46),
    ("Grok 4.1 Fast",                "xAI",       "2025-11-17",  1106.63),
    ("DeepSeek-V3.2",                "DeepSeek",  "2025-12-01",  1034.00),
    ("Gemini 3.1 Pro",               "Google",    "2026-02-19",   911.21),
    ("Gemini 2.5 Pro",               "Google",    "2025-03-25",   573.64),
    ("Gemini 2.5 Flash",             "Google",    "2025-05-20",   548.84),
    ("Qwen 3.5 35B A3B",             "Qwen",      "2026-02-16",   462.69),
    ("Claude Haiku 4.5",             "Anthropic", "2025-09-29",   458.89),
    ("Qwen 3.5 27B",                 "Qwen",      "2026-02-16",   201.98),
    ("MiniMax-M2",                   "MiniMax",   "2025-10-23",   160.60),
    ("Qwen3 Max",                    "Qwen",      "2025-09-05",    71.57),
    ("Grok 4.3",                     "xAI",       "2026-04-17",    35.26),
    ("Qwen 3.5 Plus",                "Qwen",      "2026-02-16",     0.54),
    ("Qwen3 235B A22B Thinking",     "Qwen",      "2025-04-28",   -11.34),
    ("GPT-OSS-120b",                 "OpenAI",    "2025-08-05",   -21.53),
    ("MiniMax-M2.5",                 "MiniMax",   "2026-02-12",   -23.16),
    ("MiniMax-M3",                   "MiniMax",   "2026-06-01",  2157.77),
    ("GPT-5 mini",                   "OpenAI",    "2025-08-07",   -31.18),
]

# Date mapping sources (for verification)
date_sources = {
    "Claude Opus 4.7":             "ai_timeline_final_tr.py (exact)",
    "Claude Opus 4.8 High":        "Andon Labs leaderboard verified 2026-05-29",
    "Claude Opus 4.8 Max":         "Andon Labs leaderboard verified 2026-05-29",
    "Claude Fable 5 High":         "Andon Labs leaderboard 2026-06-09",
    "Claude Fable 5 Max":          "Andon Labs leaderboard 2026-06-09",
    "Claude Opus 4.6":             "ai_timeline_final_tr.py (exact)",
    "GPT-5.5":                     "ai_timeline_final_tr.py (exact)",
    "Claude Sonnet 4.6":           "ai_timeline_final_tr.py (exact)",
    "Claude Sonnet 5":             "Andon Labs leaderboard 2026-06-30",
    "Kimi K2.6":                   "ai_timeline_final_tr.py (exact)",
    "GPT-5.4":                     "ai_timeline_final_tr.py (exact)",
    "GPT-5.3-Codex":               "ai_timeline_final_tr.py (exact)",
    "GLM-5.2":                     "ai_timeline_final_tr.py (exact)",
    "GLM-5.1":                     "ai_timeline_final_tr.py (exact)",
    "Gemini 3 Pro":                "ai_timeline_final_tr.py (exact)",
    "Gemini 3.5 Flash":            "ai_timeline_final_tr.py (exact)",
    "Qwen 3.6 Plus":               "Qwen 3.6 base date used (2026-03-31)",
    "Claude Opus 4.5":             "ai_timeline_final_tr.py (exact)",
    "Grok 4.20":                   "ai_timeline_final_tr.py (exact)",
    "GLM-5":                       "ai_timeline_final_tr.py (exact)",
    "Qwen 3.6 Max":                "Qwen 3.6 base date used (2026-03-31)",
    "Claude Sonnet 4.5":           "ai_timeline_final_tr.py (exact)",
    "Gemini 3.1 Pro Custom Tools": "Gemini 3.1 Pro date used (2026-02-19)",
    "Gemini 3 Flash":              "ai_timeline_final_tr.py (exact)",
    "GPT-5.2":                     "ai_timeline_final_tr.py (exact)",
    "Deepseek V4 Pro":             "DeepSeek-V4 date used (2026-04-24)",
    "GLM-4.7":                     "ai_timeline_final_tr.py (exact)",
    "GPT-5.1":                     "ai_timeline_final_tr.py (exact)",
    "Kimi K2.5":                   "ai_timeline_final_tr.py (exact)",
    "Grok 4.1 Fast":               "Grok-4.1 date used (2025-11-17)",
    "DeepSeek-V3.2":               "ai_timeline_final_tr.py (exact)",
    "Gemini 3.1 Pro":              "ai_timeline_final_tr.py (exact)",
    "Gemini 2.5 Pro":              "ai_timeline_final_tr.py (exact)",
    "Gemini 2.5 Flash":            "ai_timeline_final_tr.py (exact)",
    "Qwen 3.5 35B A3B":            "Qwen 3.5 base date used (2026-02-16)",
    "Claude Haiku 4.5":            "Sonnet 4.5 same-day assumption (4.5 family)",
    "Qwen 3.5 27B":                "Qwen 3.5 base date used (2026-02-16)",
    "MiniMax-M2":                  "ai_timeline_final_tr.py (exact)",
    "Qwen3 Max":                   "ai_timeline_final_tr.py (exact: Qwen3-Max)",
    "Grok 4.3":                    "ai_timeline_final_tr.py (exact: Grok-4.3)",
    "Qwen 3.5 Plus":               "Qwen 3.5 base date used (2026-02-16)",
    "Qwen3 235B A22B Thinking":    "Qwen3 base date used (2025-04-28)",
    "GPT-OSS-120b":                "assumed 2025-08-05 (OpenAI open-weight release)",
    "MiniMax-M2.5":                "ai_timeline_final_tr.py (exact)",
    "GPT-5 mini":                  "GPT-5 date used (2025-08-07)",
}

print("=" * 72)
print("DATE MAPPING TABLE")
print("=" * 72)
print(f"{'Model':<37} {'Date':<12} {'Source'}")
print("-" * 72)
for model, company, date, balance in leaderboard:
    source = date_sources.get(model, "?")
    print(f"{model:<37} {date:<12} {source}")

# Color palette
colors = {
    "OpenAI":    "#10a37f",
    "Google":    "#4285F4",
    "Anthropic": "#d97757",
    "xAI":       "#1DA1F2",
    "Qwen":      "#6C3CE1",
    "DeepSeek":  "#00B4D8",
    "Z.ai":      "#00C853",
    "Kimi":      "#FF4D6D",
    "MiniMax":   "#C77DFF",
}

company_display = {
    "OpenAI":    "OpenAI",
    "Google":    "Google",
    "Anthropic": "Anthropic",
    "xAI":       "xAI",
    "Qwen":      "Qwen / Alibaba",
    "DeepSeek":  "DeepSeek",
    "Z.ai":      "Z.ai (GLM)",
    "Kimi":      "Kimi / Moonshot",
    "MiniMax":   "MiniMax",
}

# ============================================================
# FIGURE
# ============================================================
fig = plt.figure(figsize=(10, 6.25), facecolor="#0f1419")
ax = fig.add_axes([0.09, 0.12, 0.86, 0.73], facecolor="#1a1f2e")

# Data prep
dates = [datetime.strptime(row[2], "%Y-%m-%d") for row in leaderboard]
values = [row[3] for row in leaderboard]
companies = [row[1] for row in leaderboard]
names = [row[0] for row in leaderboard]

# Scatter dots
for i, (d, v, c, n) in enumerate(zip(dates, values, companies, names)):
    col = colors.get(c, "#888888")
    ax.scatter(d, v, color=col, s=55, zorder=5, alpha=0.92,
               edgecolors="white", linewidths=0.3)

# ============================================================
# LABELS - smart offsets
# ============================================================
manual_offsets = {
    "Kimi K2.7 Code":             (-70,  6),
    "MiniMax-M3":                 (  4,  6),
    "Claude Opus 4.7":            ( 4,  6),
    "Claude Opus 4.8 High":       ( 4,  8),
    "Claude Opus 4.8 Max":        ( 4, -10),
    "Claude Fable 5 High":        ( 10,  16),
    "Claude Fable 5 Max":         ( 10, -14),
    "Claude Opus 4.6":            ( 4,  5),
    "GPT-5.5":                    (-80,  8),
    "Claude Sonnet 4.6":          ( 4, -10),
    "Claude Sonnet 5":            ( 8,  10),
    "Kimi K2.6":                  ( 4,  5),
    "GPT-5.4":                    (-60, -11),
    "GPT-5.3-Codex":              (-80,  5),
    "GLM-5.2":                    (-58, 12),
    "GLM-5.1":                    ( 4,  5),
    "Gemini 3 Pro":               (-90, -10),
    "Gemini 3.5 Flash":           ( 4, -10),
    "Qwen 3.6 Plus":              ( 4,  5),
    "Claude Opus 4.5":            (-90,  5),
    "Grok 4.20":                  (-60,  5),
    "GLM-5":                      (-40, -11),
    "Qwen 3.6 Max":               ( 4,  5),
    "Claude Sonnet 4.5":          (-90,  5),
    "Gemini 3.1 Pro Custom Tools": ( 4,  5),
    "Gemini 3 Flash":             (-80,  5),
    "GPT-5.2":                    ( 4, -10),
    "Deepseek V4 Pro":            ( 4,  5),
    "GLM-4.7":                    ( 4,  5),
    "GPT-5.1":                    (-60,  5),
    "Kimi K2.5":                  ( 4, -10),
    "Grok 4.1 Fast":              ( 4,  5),
    "DeepSeek-V3.2":              (-80,  5),
    "Gemini 3.1 Pro":             ( 4, -10),
    "Gemini 2.5 Pro":             (-80,  5),
    "Gemini 2.5 Flash":           ( 4, -10),
    "Qwen 3.5 35B A3B":           ( 4,  5),
    "Claude Haiku 4.5":           (-90, -11),
    "Qwen 3.5 27B":               ( 4,  5),
    "MiniMax-M2":                 (-75,  5),
    "Qwen3 Max":                  ( 4,  5),
    "Grok 4.3":                   ( 4,  5),
    "Qwen 3.5 Plus":              ( 4,  5),
    "Qwen3 235B A22B Thinking":   ( 4, -12),
    "GPT-OSS-120b":               (-80,  5),
    "MiniMax-M2.5":               ( 4,  5),
    "GPT-5 mini":                 (-60, -12),
}

for i, (d, v, c, n) in enumerate(zip(dates, values, companies, names)):
    col = colors.get(c, "#888888")
    ox, oy = manual_offsets.get(n, (4, 4))
    ax.annotate(
        n,
        xy=(d, v),
        xytext=(ox, oy),
        textcoords="offset points",
        fontsize=5.5,
        color=col,
        alpha=0.92,
        ha="left" if ox >= 0 else "right",
        va="center",
    )

# ============================================================
# REFERENCE LINES
# ============================================================
ax.axhline(0, color="#e05252", linewidth=0.9, linestyle="--", alpha=0.7, zorder=3)
ax.text(
    datetime(2024, 2, 1), 80,
    "Break-even ($0)",
    color="#e05252", fontsize=7, alpha=0.85
)

ax.axhline(500, color="#f0c040", linewidth=0.6, linestyle=":", alpha=0.5, zorder=3)
ax.text(
    datetime(2024, 2, 1), 550,
    "Starting capital: $500",
    color="#f0c040", fontsize=6.5, alpha=0.75
)

# ============================================================
# AXES
# ============================================================
ax.set_xlim(datetime(2024, 1, 1), datetime(2026, 8, 1))
ymin = min(values) - 300
ymax = max(values) + 800
ax.set_ylim(ymin, ymax)

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right",
         fontsize=8, color="#b0b8c8")

def fmt_dollar(x, pos):
    if x >= 1000:
        return f"${x/1000:.0f}K"
    elif x < 0:
        return f"-${abs(int(x))}"
    else:
        return f"${int(x)}"

import matplotlib.ticker as mticker
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_dollar))
ax.tick_params(axis="y", colors="#b0b8c8", labelsize=8)
ax.tick_params(axis="x", colors="#b0b8c8")

ax.grid(axis="y", color="#2a3044", linewidth=0.5, alpha=0.7, zorder=1)
ax.grid(axis="x", color="#2a3044", linewidth=0.3, alpha=0.4, zorder=1)

for spine in ax.spines.values():
    spine.set_edgecolor("#2a3044")

# ============================================================
# AXIS LABELS
# ============================================================
ax.set_xlabel("Model Release Date", color="#c8d0e0", fontsize=9, labelpad=6)
ax.set_ylabel("Bank Balance at End of Year ($)", color="#c8d0e0", fontsize=9, labelpad=6)

# ============================================================
# TITLE
# ============================================================
fig.text(
    0.5, 0.93,
    "Vending-Bench 2  -  AI Models Running a One-Year Vending Machine Business",
    ha="center", va="center", fontsize=10.5, color="white", fontweight="bold"
)
fig.text(
    0.5, 0.895,
    "43 Models  -  Source: andonlabs.com  -  Starting: $500 (1-year sim balance from $-31 to $10,936)",
    ha="center", va="center", fontsize=7.5, color="#8090a8"
)

# ============================================================
# TOP RIGHT: attribution
# ============================================================
fig.text(0.96, 0.97, "Prof. Dr. Oguz Ergin",
         ha="right", va="top", fontsize=7, color="#606878", style="italic")

# ============================================================
# LEGEND top left
# ============================================================
legend_handles = []
seen = set()
for row in leaderboard:
    c = row[1]
    if c not in seen:
        seen.add(c)
        col = colors.get(c, "#888888")
        lbl = company_display.get(c, c)
        legend_handles.append(
            mpatches.Patch(facecolor=col, edgecolor="white",
                           linewidth=0.4, label=lbl)
        )

legend = ax.legend(
    handles=legend_handles,
    loc="upper left",
    fontsize=6.5,
    framealpha=0.25,
    facecolor="#0f1419",
    edgecolor="#2a3044",
    ncol=1,
    handlelength=1,
    handleheight=0.9,
    borderpad=0.6,
    labelcolor="white",
    title="Company",
    title_fontsize=7,
)
legend.get_title().set_color("#c8d0e0")

# ============================================================
# SAVE
# ============================================================
out_path = os.path.join(os.path.dirname(__file__), "vending_bench_2.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1419")
plt.close()

file_size = os.path.getsize(out_path)
print(f"\nPNG saved: {out_path}")
print(f"Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

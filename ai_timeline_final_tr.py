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
    ("o1", "OpenAI", "2024-12-05", True),
    ("o3-mini", "OpenAI", "2025-01-31", False),
    ("GPT-4.5", "OpenAI", "2025-02-27", False),
    ("GPT-4.1", "OpenAI", "2025-04-14", True),
    ("o3", "OpenAI", "2025-04-16", True),
    ("o4-mini", "OpenAI", "2025-04-16", False),
    ("o3-pro", "OpenAI", "2025-06-10", False),
    ("GPT-5", "OpenAI", "2025-08-07", True),
    ("GPT-5.1", "OpenAI", "2025-11-12", False),
    ("GPT-5.2", "OpenAI", "2025-12-11", False),
    ("GPT-5.3-Codex", "OpenAI", "2026-02-05", False),
    ("GPT-5.4", "OpenAI", "2026-03-05", True),
    ("GPT-5.4 mini", "OpenAI", "2026-03-17", False),
    ("GPT-5.5", "OpenAI", "2026-04-23", True),

    # Google
    ("Gemini 1.0 Pro", "Google", "2023-12-06", True),
    ("Gemini 1.5 Pro", "Google", "2024-02-15", True),
    ("Gemini 1.5 Flash", "Google", "2024-05-14", False),
    ("Gemini 2.0 Flash", "Google", "2025-01-30", False),
    ("Gemini 2.0 Pro", "Google", "2025-02-05", True),
    ("Gemini 2.5 Pro", "Google", "2025-03-25", True),
    ("Gemini 2.5 Flash", "Google", "2025-05-20", False),
    ("Gemini 3 Pro", "Google", "2025-11-18", True),
    ("Gemini 3 Flash", "Google", "2025-12-17", False),
    ("Gemini 3 Flash GA", "Google", "2026-04-22", True),
    ("Gemini 3 Deep Think", "Google", "2026-02-12", True),
    ("Gemini 3.1 Pro", "Google", "2026-02-19", True),
    ("Gemini 3.1 Flash-Lite", "Google", "2026-03-03", False),

    # Anthropic
    ("Claude 1", "Anthropic", "2023-03-14", True),
    ("Claude 2", "Anthropic", "2023-07-11", True),
    ("Claude 3 Opus", "Anthropic", "2024-03-04", True),
    ("Claude 3.5 Sonnet", "Anthropic", "2024-06-20", True),
    ("Claude 3.5 Haiku", "Anthropic", "2024-10-22", False),
    ("Claude 3.7 Sonnet", "Anthropic", "2025-02-24", False),
    ("Claude Opus 4", "Anthropic", "2025-05-22", True),
    ("Claude Sonnet 4", "Anthropic", "2025-05-22", False),
    ("Claude Opus 4.1", "Anthropic", "2025-08-05", False),
    ("Claude Sonnet 4.5", "Anthropic", "2025-09-29", False),
    ("Claude Opus 4.5", "Anthropic", "2025-11-24", False),
    ("Claude Opus 4.6", "Anthropic", "2026-02-05", True),
    ("Claude Sonnet 4.6", "Anthropic", "2026-02-17", False),
    ("Claude Opus 4.7", "Anthropic", "2026-04-16", True),

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
    ("Grok-4.3", "xAI", "2026-04-17", False),

    # Meta
    ("Llama 1", "Meta", "2023-02-24", True),
    ("Llama 2", "Meta", "2023-07-18", True),
    ("Llama 3", "Meta", "2024-04-18", True),
    ("Llama 3.1", "Meta", "2024-07-23", False),
    ("Llama 3.2", "Meta", "2024-09-25", False),
    ("Llama 3.3", "Meta", "2024-12-06", False),
    ("Llama 4", "Meta", "2025-04-05", True),

    # Meta Muse (Closed Source)
    ("Muse Spark", "Meta Muse", "2026-04-08", True),

    # Microsoft Phi (Open Source)
    ("Phi-1", "Microsoft", "2023-06-20", True),
    ("Phi-2", "Microsoft", "2023-12-12", True),
    ("Phi-3", "Microsoft", "2024-04-23", True),
    ("Phi-3.5", "Microsoft", "2024-08-20", False),
    ("Phi-4", "Microsoft", "2024-12-12", True),
    ("Phi-4-multimodal", "Microsoft", "2025-02-26", False),
    ("Phi-4-reasoning", "Microsoft", "2025-05-01", False),

    # Mistral
    ("Mistral 7B", "Mistral", "2023-09-27", True),
    ("Mixtral 8x7B", "Mistral", "2023-12-11", True),
    ("Mistral Large", "Mistral", "2024-02-26", True),
    ("Codestral", "Mistral", "2024-05-29", False),
    ("Mistral Large 2", "Mistral", "2024-07-24", False),
    ("Pixtral 12B", "Mistral", "2024-09-11", False),
    ("Mistral Small 3", "Mistral", "2025-01-30", False),
    ("Mistral Medium 3", "Mistral", "2025-05-07", False),
    ("Magistral", "Mistral", "2025-06-10", True),
    ("Mistral Large 3", "Mistral", "2025-12-02", True),
    ("Mistral Small 4", "Mistral", "2026-03-16", False),

    # Qwen (Alibaba)
    ("Qwen 1", "Qwen", "2023-09-28", True),
    ("Qwen 1.5", "Qwen", "2024-02-04", False),
    ("Qwen 2", "Qwen", "2024-06-06", True),
    ("Qwen 2.5", "Qwen", "2024-09-19", True),
    ("QwQ-32B", "Qwen", "2025-03-05", False),
    ("Qwen 3", "Qwen", "2025-04-28", True),
    ("Qwen3-Max", "Qwen", "2025-09-05", False),
    ("Qwen3-Coder", "Qwen", "2026-02-02", False),
    ("Qwen 3.5", "Qwen", "2026-02-16", True),
    ("Qwen 3.6", "Qwen", "2026-03-31", False),

    # DeepSeek
    ("DeepSeek Coder", "DeepSeek", "2023-11-02", True),
    ("DeepSeek LLM", "DeepSeek", "2023-11-29", True),
    ("DeepSeek-V2", "DeepSeek", "2024-05-06", True),
    ("DeepSeek-V3", "DeepSeek", "2024-12-26", True),
    ("DeepSeek-R1", "DeepSeek", "2025-01-20", True),
    ("DeepSeek-V3.1", "DeepSeek", "2025-08-21", False),
    ("DeepSeek-V3.2", "DeepSeek", "2025-12-01", False),
    ("DeepSeek-V4", "DeepSeek", "2026-04-24", True),

    # Z.ai (Zhipu AI)
    ("ChatGLM", "Z.ai", "2023-03-14", True),
    ("GLM-4", "Z.ai", "2024-01-16", True),
    ("GLM-4.5", "Z.ai", "2025-07-28", True),
    ("GLM-4.6", "Z.ai", "2025-09-30", False),
    ("GLM-4.7", "Z.ai", "2025-12-22", False),
    ("GLM-5", "Z.ai", "2026-02-11", True),
    ("GLM-5.1", "Z.ai", "2026-03-27", False),

    # Kimi (Moonshot AI)
    ("Kimi Chat", "Kimi", "2023-10-09", True),
    ("Kimi K1.5", "Kimi", "2025-01-20", True),
    ("Kimi K2", "Kimi", "2025-07-11", True),
    ("Kimi K2 Thinking", "Kimi", "2025-11-06", False),
    ("Kimi K2.5", "Kimi", "2026-01-27", False),
    ("Kimi K2.6", "Kimi", "2026-04-13", True),

    # MiniMax
    ("MiniMax-Text-01", "MiniMax", "2025-01-15", True),
    ("MiniMax-M1", "MiniMax", "2025-06-16", True),
    ("MiniMax-M2", "MiniMax", "2025-10-23", True),
    ("MiniMax-M2.1", "MiniMax", "2025-12-25", False),
    ("MiniMax-M2.5", "MiniMax", "2026-02-12", False),
    ("MiniMax-M2.7", "MiniMax", "2026-03-18", False),

    # Google Gemma (Open Source / Open Weight)
    ("Gemma 1", "Google Gemma", "2024-02-21", True),
    ("Gemma 2", "Google Gemma", "2024-06-27", True),
    ("Gemma 3", "Google Gemma", "2025-03-12", True),
    ("Gemma 3n", "Google Gemma", "2025-06-26", False),
    ("Gemma 4", "Google Gemma", "2026-04-02", True),

    # ByteDance (Doubao) - Closed Source
    ("Doubao Pro", "ByteDance", "2024-05-15", True),
    ("Doubao-1.5-Pro", "ByteDance", "2025-01-22", False),
    ("Doubao-Seed-2.0", "ByteDance", "2026-02-14", True),

    # Amazon (Nova) - Closed Source
    ("Amazon Nova Pro", "Amazon", "2024-12-03", True),
    ("Amazon Nova Premier", "Amazon", "2025-04-30", True),
    ("Amazon Nova 2 Pro", "Amazon", "2025-12-02", True),

    # Cohere (Command) - Closed Source
    ("Command R", "Cohere", "2024-03-11", True),
    ("Command R+", "Cohere", "2024-04-04", False),
    ("Command A", "Cohere", "2025-03-13", True),
    ("Command A Vision", "Cohere", "2025-07-31", False),
    ("Command A Reasoning", "Cohere", "2025-08-21", False),
    ("Command A Translate", "Cohere", "2025-08-28", False),
]

df = pd.DataFrame(data, columns=["Model", "Company", "Date", "Milestone"])
df["Date"] = pd.to_datetime(df["Date"])

# =============================================
# SHORT LABELS: strip company/series prefix
# for compact display on timeline
# =============================================
short_labels = {
    # OpenAI - mixed series (GPT, o, codex), keep GPT prefix short, others as-is
    "GPT-3.5":          "3.5",
    "GPT-4":            "4",
    "GPT-4 Turbo":      "4 Turbo",
    "GPT-4o":           "4o",
    "GPT-4.5":          "4.5",
    "GPT-4o mini":      "4o mini",
    "o1-preview":       "o1-preview",
    "o1":               "o1",
    "o3-mini":          "o3-mini",
    "GPT-4.1":          "4.1",
    "o3":               "o3",
    "o4-mini":          "o4-mini",
    "codex-1":          "codex-1",
    "o3-pro":           "o3-pro",
    "GPT-5":            "5",
    "GPT-5.1":          "5.1",
    "GPT-5.2":          "5.2",
    "GPT-5.3-Codex":    "5.3-Codex",
    "GPT-5.4":          "5.4",
    "GPT-5.4 mini":     "5.4 mini",
    "GPT-5.5":          "5.5",

    # Google Gemini
    "Gemini 1.0 Pro":       "1.0 Pro",
    "Gemini 1.5 Pro":       "1.5 Pro",
    "Gemini 1.5 Flash":     "1.5 Flash",
    "Gemini 2.0 Flash":     "2.0 Flash",
    "Gemini 2.0 Pro":       "2.0 Pro",
    "Gemini 2.5 Pro":       "2.5 Pro",
    "Gemini 2.5 Flash":     "2.5 Flash",
    "Gemini 3 Pro":         "3 Pro",
    "Gemini 3 Flash":       "3 Flash",
    "Gemini 3 Flash GA":    "3 Flash GA",
    "Gemini 3 Deep Think":  "3 D. Think",
    "Gemini 3.1 Pro":       "3.1 Pro",
    "Gemini 3.1 Flash-Lite": "3.1 FL",

    # Anthropic Claude
    "Claude 1":             "1",
    "Claude 2":             "2",
    "Claude 3 Opus":        "Opus 3",
    "Claude 3.5 Sonnet":    "Sonnet 3.5",
    "Claude 3.5 Haiku":     "Haiku 3.5",
    "Claude 3.7 Sonnet":    "Sonnet 3.7",
    "Claude Opus 4":        "Opus 4",
    "Claude Sonnet 4":      "Sonnet 4",
    "Claude Opus 4.1":      "Opus 4.1",
    "Claude Sonnet 4.5":    "Sonnet 4.5",
    "Claude Opus 4.5":      "Opus 4.5",
    "Claude Opus 4.6":      "Opus 4.6",
    "Claude Sonnet 4.6":    "Sonnet 4.6",
    "Claude Opus 4.7":      "Opus 4.7",

    # xAI Grok
    "Grok-1":           "1",
    "Grok-1.5":         "1.5",
    "Grok-2":           "2",
    "Grok-3":           "3",
    "Grok-3 Mini":      "3 Mini",
    "Grok-4":           "4",
    "Grok Code Fast":   "Code Fast",
    "Grok-4.1":         "4.1",
    "Grok 4.20":        "4.20",
    "Grok-4.3":         "4.3",

    # Meta Llama
    "Llama 1":          "1",
    "Llama 2":          "2",
    "Llama 3":          "3",
    "Llama 3.1":        "3.1",
    "Llama 3.2":        "3.2",
    "Llama 3.3":        "3.3",
    "Llama 4":          "4",

    # Microsoft Phi
    "Phi-1":            "1",
    "Phi-2":            "2",
    "Phi-3":            "3",
    "Phi-3.5":          "3.5",
    "Phi-4":            "4",
    "Phi-4-multimodal": "4-multimodal",
    "Phi-4-reasoning":  "4-reasoning",

    # Mistral - mixed names, keep distinctive ones full
    "Mistral 7B":       "7B",
    "Mixtral 8x7B":     "Mixtral 8x7B",
    "Mistral Large":    "Large",
    "Codestral":        "Codestral",
    "Mistral Large 2":  "Large 2",
    "Pixtral 12B":      "Pixtral 12B",
    "Mistral Small 3":  "Small 3",
    "Mistral Small 3.1":"Small 3.1",
    "Mistral Medium 3": "Medium 3",
    "Magistral":        "Magistral",
    "Mistral Large 3":  "Large 3",
    "Mistral Small 4":  "Small 4",

    # Qwen
    "Qwen 1":           "1",
    "Qwen 1.5":         "1.5",
    "Qwen 2":           "2",
    "Qwen 2.5":         "2.5",
    "QwQ-32B":          "QwQ-32B",
    "Qwen 3":           "3",
    "Qwen3-Max":        "3-Max",
    "Qwen3-Coder":      "3-Coder",
    "Qwen 3.5":         "3.5",
    "Qwen 3.6":         "3.6",

    # DeepSeek
    "DeepSeek Coder":   "Coder",
    "DeepSeek LLM":     "LLM",
    "DeepSeek-V2":      "V2",
    "DeepSeek-V3":      "V3",
    "DeepSeek-R1":      "R1",
    "DeepSeek-V3-0324": "V3-0324",
    "DeepSeek-R1-0528": "R1-0528",
    "DeepSeek-V3.1":    "V3.1",
    "DeepSeek-V3.2":    "V3.2",
    "DeepSeek-V4":      "V4",

    # Z.ai GLM
    "ChatGLM":          "ChatGLM",
    "GLM-4":            "4",
    "GLM-4.5":          "4.5",
    "GLM-4.6":          "4.6",
    "GLM-4.7":          "4.7",
    "GLM-5":            "5",
    "GLM-5.1":          "5.1",

    # Kimi
    "Kimi Chat":        "Chat",
    "Kimi K1.5":        "K1.5",
    "Kimi K2":          "K2",
    "Kimi K2 Thinking": "K2 Thinking",
    "Kimi K2.5":        "K2.5",
    "Kimi K2.6":        "K2.6",

    # MiniMax
    "MiniMax-Text-01":  "Text-01",
    "MiniMax-M1":       "M1",
    "MiniMax-M2":       "M2",
    "MiniMax-M2.1":     "M2.1",
    "MiniMax-M2.5":     "M2.5",
    "MiniMax-M2.7":     "M2.7",

    # Google Gemma
    "Gemma 1":          "1",
    "Gemma 2":          "2",
    "Gemma 3":          "3",
    "Gemma 3n":         "3n",
    "Gemma 4":          "4",

    # ByteDance Doubao
    "Doubao Pro":           "Pro",
    "Doubao-1.5-Pro":       "1.5 Pro",
    "Doubao-Seed-2.0":      "Seed 2.0",

    # Meta Muse
    "Muse Spark":           "Spark",

    # Amazon Nova
    "Amazon Nova Pro":      "Pro",
    "Amazon Nova Premier":  "Premier",
    "Amazon Nova 2 Pro":    "2 Pro",

    # Cohere Command
    "Command R":            "R",
    "Command R+":           "R+",
    "Command A":            "A",
    "Command A Vision":     "A Vision",
    "Command A Reasoning":  "A Reasoning",
    "Command A Translate":  "A Translate",
}

# Series name shown on Y-axis next to company name
series_name = {
    "OpenAI":       "GPT / o",
    "Google":       "Gemini",
    "Anthropic":    "Claude",
    "xAI":          "Grok",
    "Meta":         "Llama",
    "Meta Muse":    "Muse",
    "Microsoft":    "Phi",
    "Mistral":      "Mistral",
    "Qwen":         "Qwen",
    "DeepSeek":     "DeepSeek",
    "Z.ai":         "GLM",
    "Kimi":         "Kimi",
    "MiniMax":      "MiniMax",
    "Google Gemma": "Gemma",
    "ByteDance":    "Doubao",
    "Amazon":       "Nova",
    "Cohere":       "Command",
}

colors = {
    "OpenAI":    "#10a37f",
    "Google":    "#4285F4",
    "Anthropic": "#d97757",
    "xAI":       "#1DA1F2",
    "Meta":      "#0668E1",
    "Meta Muse": "#0668E1",
    "Microsoft": "#F25022",
    "Mistral":   "#FF7000",
    "Qwen":      "#6C3CE1",
    "DeepSeek":  "#00B4D8",
    "Z.ai":      "#00C853",
    "Kimi":      "#FF4D6D",
    "MiniMax":   "#C77DFF",
    "Google Gemma": "#34A853",
    "ByteDance":    "#00F0FF",
    "Amazon":       "#FF9900",
    "Cohere":       "#D18EE2",
}

company_name = {
    "OpenAI":    "OpenAI",
    "Google":    "Google",
    "Anthropic": "Anthropic",
    "xAI":       "xAI",
    "Meta":      "Meta",
    "Meta Muse": "Meta",
    "Microsoft": "Microsoft",
    "Mistral":   "Mistral AI",
    "Qwen":      "Qwen / Alibaba",
    "DeepSeek":  "DeepSeek",
    "Z.ai":      "Z.ai / Zhipu",
    "Kimi":      "Kimi / Moonshot",
    "MiniMax":   "MiniMax",
    "Google Gemma": "Google",
    "ByteDance":    "ByteDance",
    "Amazon":       "Amazon / AWS",
    "Cohere":       "Cohere",
}

country_text = {
    "OpenAI":    "ABD",
    "Google":    "ABD",
    "Anthropic": "ABD",
    "xAI":       "ABD",
    "Meta":      "ABD",
    "Meta Muse": "ABD",
    "Microsoft": "ABD",
    "Mistral":   "Fransa",
    "Qwen":      "Çin",
    "DeepSeek":  "Çin",
    "Z.ai":      "Çin",
    "Kimi":      "Çin",
    "MiniMax":   "Çin",
    "Google Gemma": "ABD",
    "ByteDance":    "Çin",
    "Amazon":       "ABD",
    "Cohere":       "Kanada",
}

# Flag image files
flag_images = {
    "ABD":    "C:/Users/Z GAMES/flags/us.png",
    "Fransa": "C:/Users/Z GAMES/flags/fr.png",
    "Çin":    "C:/Users/Z GAMES/flags/cn.png",
    "Kanada": "C:/Users/Z GAMES/flags/ca.png",
}

company_order = ["OpenAI", "Google", "Anthropic", "xAI", "Meta Muse", "Amazon", "ByteDance", "Cohere",
                 "Meta", "Microsoft", "Google Gemma", "Mistral",
                 "Qwen", "DeepSeek", "Z.ai", "Kimi", "MiniMax"]

# Vertical spacing: each company gets 3.0 units
y_positions = {c: (len(company_order) - 1 - i) * 3.0 for i, c in enumerate(company_order)}

# Dark theme
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(60, 58.5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# =========================================
# BACKGROUND SHADING for closed/open source
# =========================================
separator_y = (y_positions["Cohere"] + y_positions["Meta"]) / 2
y_min = y_positions[company_order[-1]] - 1.8
y_max = y_positions[company_order[0]] + 2.8

# Shading starts at the first model date, extends to match x-axis limit (Jun 2026)
shade_left = mdates.date2num(df["Date"].min() - pd.Timedelta(days=15))
shade_right = mdates.date2num(pd.Timestamp("2026-06-01"))
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
        "AÇIK KAYNAK / AÇIK AĞIRLIK",
        fontsize=56, color="#3fb950", va="center", ha="left",
        fontweight="bold", alpha=0.22)
ax.text(df["Date"].min() + pd.Timedelta(days=30), y_positions["Cohere"] + 0.5,
        "KAPALI KAYNAK",
        fontsize=56, color="#f778ba", va="center", ha="left",
        fontweight="bold", alpha=0.22)

labels_y = []

# Manual overrides: SHORT_LABEL -> (x_offset, y_offset)
manual_overrides = {
    # OpenAI
    "3.5":              (50, -55),
    "5.1":              (0, -55),
    "o1":               (0, -55),
    "4.1":              (-80, -55),
    "o3-pro":           (65, -55),
    "5.4 mini":         (0, -55),

    # Google - end of timeline very crowded
    "2.0 Pro":          (-40, 55),
    "2.5 Pro":          (40, 55),
    "1.0 Pro":          (0, -55),
    "3 Pro":            (-80, 55),
    "3 Flash":          (0, 55),
    "3 D. Think":       (0, -55),
    "3.1 Pro":          (0, 55),
    "3.1 FL":           (120, -55),

    # DeepSeek
    "V3.1":             (0, 55),

    # Mistral
    "7B":               (0, 55),
    "Mixtral 8x7B":     (0, -55),
    "Pixtral 12B":      (0, 55),
    "Small 3.1":        (-50, 55),
    "Medium 3":         (0, 55),
    "Magistral":        (0, -55),

    # xAI - Grok-3 and Grok-3 Mini same day
    "3":                (0, 55),
    "3 Mini":           (65, -55),

    # Meta - Llama 2 -> üste (milestone)
    "2":                (0, 55),

    # Microsoft
    "4-multimodal":     (0, 55),
    "4-reasoning":      (0, -55),

    # Anthropic - 11 models, very crowded row
    # "2" already used above for Meta
    "Opus 3":           (0, 55),
    "Sonnet 3.5":       (0, 55),
    "Haiku 3.5":        (0, -55),
    "Sonnet 3.7":       (0, -55),
    "Opus 4":           (0, 55),
    "Sonnet 4":         (0, -55),
    "Opus 4.1":         (0, 55),
    "Sonnet 4.5":       (0, -55),
    "Opus 4.5":         (0, 55),
    "Opus 4.6":         (0, 55),
    "Sonnet 4.6":       (0, -55),
}

# Company-specific overrides for labels that clash across companies
# (e.g. "2" is used by Meta/Llama, Anthropic/Claude, xAI/Grok, Microsoft/Phi)
company_overrides = {
    ("OpenAI", "4"):        (0, -55),
    ("OpenAI", "4.5"):      (0, 55),
    ("OpenAI", "o3"):       (0, 55),
    ("OpenAI", "o4-mini"):  (60, -55),
    ("OpenAI", "5.2"):      (0, 55),
    ("OpenAI", "5.3-Codex"): (-60, -55),
    ("DeepSeek", "V3"):     (0, 55),
    ("DeepSeek", "R1"):     (0, -55),
    ("OpenAI", "5.4"):      (0, 55),
    ("OpenAI", "5.5"):      (0, 55),
    ("Anthropic", "2"):     (0, 55),
    ("xAI", "2"):           (0, 55),
    ("Microsoft", "2"):     (0, 55),
    ("Google Gemma", "2"):  (0, 55),
    ("Qwen", "2"):          (0, 55),
    ("Qwen", "3"):          (0, 55),
    ("Qwen", "3.5"):        (0, 55),
    ("Qwen", "3.6"):        (0, -55),
    ("Meta", "3"):          (0, 55),
    ("Microsoft", "3"):     (0, 55),
    ("Microsoft", "3.5"):   (0, -55),
    ("Google Gemma", "3"):  (0, 55),
    # xAI - Grok 3 centered above, 3 Mini right below, 4.1 centered below
    ("xAI", "3"):           (0, 55),
    ("xAI", "3 Mini"):      (0, -55),
    ("xAI", "4.1"):         (0, -55),
    ("xAI", "4.3"):         (0, -55),
    # Google - Gemini 3.1 Pro tam dot üstüne otursun, gap adjustment nudge atlansın
    ("Google", "3.1 Pro"):  (0, 55),
    # Z.ai - GLM-4 uses "4" which clashes with other companies
    ("Z.ai", "4"):          (0, 55),
    # Anthropic - Opus 4 and Sonnet 4 same day, aligned with dots
    ("Anthropic", "Opus 4"):    (0, 55),
    ("Anthropic", "Sonnet 4"):  (0, -55),
    ("Anthropic", "Sonnet 4.6"): (0, -55),
    ("Anthropic", "Opus 4.7"):   (0, 55),
    # Mistral - Medium 3 aligned with dot
    ("Mistral", "Medium 3"): (0, 55),
    # Cohere - R and R+ aligned with dots
    ("Cohere", "R"):        (0, 55),
    ("Cohere", "R+"):       (0, -55),
    # DeepSeek - Coder and LLM very close (Nov 2 vs Nov 29)
    ("DeepSeek", "Coder"):  (0, 55),
    ("DeepSeek", "LLM"):    (0, -55),
    # Cohere - A Vision, A Reasoning, A Translate close together
    ("Cohere", "A Vision"):     (-50, -55),
    ("Cohere", "A Reasoning"):  (0, 55),
    ("Cohere", "A Translate"):  (50, -55),
    # MiniMax - M2.1 and M2.5 both False in same cluster, keep M2.5 close
    ("MiniMax", "M2.5"):        (0, -55),
    ("MiniMax", "M2.7"):        (0, 55),
    ("ByteDance", "Seed 2.0"):  (0, -55),
    ("Z.ai", "5.1"):            (0, 55),
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

    L1 = 55  # standard distance from line
    if company == "Anthropic":
        levels_up   = [L1, L1*2.2, L1*3.4, L1*4.6]
        levels_down = [-L1, -L1*2.2, -L1*3.4, -L1*4.6]
    else:
        levels_up   = [L1, L1*2.2, L1*3.4]
        levels_down = [-L1, -L1*2.2, -L1*3.4]

    for start, end in clusters:
        cluster_size = end - start + 1
        if cluster_size == 1:
            model_name = names[start]
            # Check company-specific override first
            if (company, model_name) in company_overrides:
                offsets.append(company_overrides[(company, model_name)][1])
            elif model_name in manual_overrides:
                offsets.append(manual_overrides[model_name][1])
            else:
                offsets.append(L1 if milestones[start] else -L1)
        else:
            up_idx = 0
            down_idx = 0
            for k in range(start, end + 1):
                model_name = names[k]
                if (company, model_name) in company_overrides:
                    offsets.append(company_overrides[(company, model_name)][1])
                elif model_name in manual_overrides:
                    offsets.append(manual_overrides[model_name][1])
                elif milestones[k]:
                    # Milestone models go above the line when possible
                    offsets.append(levels_up[min(up_idx, len(levels_up)-1)])
                    up_idx += 1
                else:
                    # Non-milestone models go below the line
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

    # Build short label list for this company
    n = len(group)
    milestones_list = group["Milestone"].tolist()
    full_names_list = group["Model"].tolist()
    short_names_list = [short_labels.get(m, m) for m in full_names_list]

    offsets_y = smart_offsets(group["Date"].tolist(), milestones_list, short_names_list, n, company=company)

    # Horizontal nudge for same-day or very close releases
    dates_num = [d.timestamp() for d in group["Date"].tolist()]
    offsets_x = [0] * n
    for i in range(n):
        sname = short_names_list[i]
        # Company override takes full priority (even if x=0)
        if (company, sname) in company_overrides:
            offsets_x[i] = company_overrides[(company, sname)][0]
            continue
        if sname in manual_overrides and manual_overrides[sname][0] != 0:
            offsets_x[i] = manual_overrides[sname][0]
            continue
    for i in range(1, n):
        sname = short_names_list[i]
        sname_prev = short_names_list[i-1]
        # Skip gap adjustment if either model has a company override
        if (company, sname) in company_overrides:
            continue
        if (company, sname_prev) in company_overrides:
            continue
        if sname in manual_overrides and manual_overrides[sname][0] != 0:
            continue
        if sname_prev in manual_overrides and manual_overrides[sname_prev][0] != 0:
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

        display_name = short_labels.get(row["Model"], row["Model"])

        ax.annotate(display_name,
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
    # Model series name - top, big, bold, colored
    sname = series_name[company]
    display_top = sname if sname else company_name[company]
    ax.text(x_label_pos, y_pos + 0.7, display_top,
            fontsize=36, fontweight="bold", color=colors[company],
            ha="right", va="center")
    # Company name - middle, smaller, white/gray
    if sname:
        ax.text(x_label_pos, y_pos + 0.05, company_name[company],
                fontsize=26, fontweight="normal", color="#8b949e",
                ha="right", va="center")
    # Country name - below, shifted left to make room for flag
    country = country_text[company]
    flag_x_offset = date_min - pd.Timedelta(days=65)
    ax.text(mdates.date2num(flag_x_offset), y_pos - 0.65, country,
            fontsize=24, fontweight="normal", color="#8b949e",
            ha="right", va="center", fontstyle="italic")

    # Place real flag image to the right of country text
    flag_path = flag_images[country]
    flag_img = plt.imread(flag_path)
    imagebox = OffsetImage(flag_img, zoom=0.40)
    flag_date = date_min - pd.Timedelta(days=45)
    ab = AnnotationBbox(imagebox, (mdates.date2num(flag_date), y_pos - 0.65),
                        frameon=False, zorder=10,
                        clip_on=False, annotation_clip=False)
    ax.add_artist(ab)

# X axis
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_xlim(mdates.date2num(pd.Timestamp("2022-11-01")), mdates.date2num(pd.Timestamp("2026-07-01")))
plt.xticks(rotation=45, fontsize=22, color="#8b949e")

# Grid
ax.grid(True, axis="x", linestyle="--", alpha=0.06, color="white")

# Count models per year
models_per_year = df.groupby(df["Date"].dt.year).size().to_dict()

# Year markers - vertical dashed lines at Jan 1 of each year
for year in [2023, 2024, 2025, 2026]:
    year_date = mdates.date2num(pd.Timestamp(f"{year}-01-01"))
    ax.axvline(x=year_date, color="#58a6ff", linewidth=1.5, linestyle=":", alpha=0.4, zorder=1)
    count = models_per_year.get(year, 0)
    ax.text(year_date, y_max - 0.2, str(year),
            fontsize=32, fontweight="bold", color="#58a6ff", alpha=0.5,
            ha="center", va="top")
    ax.annotate(f"({count} model)", (year_date, y_max - 0.2),
                xytext=(70, -2), textcoords="offset points",
                fontsize=26, fontweight="normal", color="#58a6ff", alpha=0.38,
                ha="left", va="top")

# Separator line
ax.axhline(y=separator_y, color="#30363d", linewidth=2.5, linestyle="-", alpha=0.6)

# Spines
for spine in ax.spines.values():
    spine.set_color("#30363d")
    spine.set_linewidth(0.5)

ax.tick_params(axis="y", colors="white", length=0, pad=20)
ax.tick_params(axis="x", colors="#8b949e", length=6, pad=15)

# Title
plt.title("Yapay Zeka Model Yayınlanma Zaman Çizelgesi",
          fontsize=58, pad=80, color="white", fontweight="bold")

# Top-left corner icon: speech bubble (chat/LLM symbol)
bx, by = 0.065, 1.025
# Main bubble (rounded rectangle)
bubble = mpatches.FancyBboxPatch(
    (bx - 0.018, by - 0.010), 0.036, 0.022,
    boxstyle="round,pad=0.004", fc="#58a6ff", ec="white",
    linewidth=2, alpha=0.85, transform=ax.transAxes, zorder=22)
bubble.set_clip_on(False)
ax.add_patch(bubble)
# Bubble tail (triangle)
from matplotlib.patches import Polygon
tail = Polygon(
    [(bx - 0.010, by - 0.010), (bx - 0.005, by - 0.010), (bx - 0.018, by - 0.020)],
    closed=True, fc="#58a6ff", ec="white", linewidth=2, alpha=0.85,
    transform=ax.transAxes, zorder=21)
tail.set_clip_on(False)
ax.add_patch(tail)
# Cover tail-bubble junction line
cover = mpatches.FancyBboxPatch(
    (bx - 0.012, by - 0.011), 0.010, 0.004,
    boxstyle="square,pad=0", fc="#58a6ff", ec="none", alpha=0.85,
    transform=ax.transAxes, zorder=23)
cover.set_clip_on(False)
ax.add_patch(cover)
# Three dots inside bubble (typing indicator)
for i, dot_x in enumerate([-0.008, 0.000, 0.008]):
    d = mpatches.Circle((bx + dot_x, by + 0.002), 0.003,
                        fc="white", ec="none", alpha=0.9,
                        transform=ax.transAxes, zorder=24)
    d.set_clip_on(False)
    ax.add_patch(d)

ax.text(0.5, 1.008,
        f"15 Şirket  |  {len(df)} Model  |  Kas 2022 – Nis 2026  |  ● Büyük = Dönüm Noktası  |  ● Küçük = Güncelleme",
        transform=ax.transAxes, ha="center", fontsize=28,
        color="#8b949e", fontstyle="italic")

# Legend - use series name (model name) instead of company name
legend_elements = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=colors[c], markersize=18,
                          label=series_name[c], linewidth=0)
                   for c in company_order]
legend_elements.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#888", markersize=22,
                              markeredgecolor="white", markeredgewidth=2,
                              label="Dönüm Noktası", linewidth=0))
legend_elements.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#888", markersize=13,
                              label="Güncelleme", linewidth=0))

legend = ax.legend(handles=legend_elements, loc="upper center",
                   fontsize=20, framealpha=0.4, facecolor="#161b22",
                   edgecolor="#30363d", labelcolor="white",
                   ncol=9, bbox_to_anchor=(0.5, -0.035),
                   columnspacing=1.8, handletextpad=0.8)

# Signature / watermark - top right (handwriting font)
sig_box = mpatches.FancyBboxPatch(
    (0.82, 1.012), 0.170, 0.036,
    boxstyle="round,pad=0.006", fc="#161b22", ec="#58a6ff",
    linewidth=1.5, alpha=0.85, transform=ax.transAxes, zorder=25)
sig_box.set_clip_on(False)
ax.add_patch(sig_box)
ax.text(0.905, 1.030, "Prof. Dr. Oğuz Ergin",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=36, fontfamily="Segoe Script", color="#58a6ff",
        alpha=0.9, zorder=26).set_clip_on(False)

plt.margins(y=0.01, x=0.06)
ax.set_ylim(y_min, y_max)
plt.tight_layout(rect=[0.11, 0.03, 0.98, 0.98])
plt.savefig("G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi/ai_timeline_final_tr.png", dpi=150, bbox_inches="tight",
            pad_inches=1.0, facecolor="#0d1117", edgecolor="none")
print("Türkçe grafik kaydedildi!")

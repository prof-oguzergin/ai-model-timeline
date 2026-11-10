# -*- coding: utf-8 -*-
# Video generation Elo - NEW SCALE (AA-Video-T2V v2.1)
# Data: artificialanalysis.ai/video/leaderboard/text-to-video, snapshot 10 Nov 2026
# The old-scale chart is FROZEN in video_elo.py; the two scales are not comparable.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# [tarih, endeks, model, sirket]
DATA = [
["2026-01-30", 939.0, "Vidu Q3 Pro", "Vidu"],
["2026-01-30", 956.0, "Veo 3.1 Fast", "Google"],
["2026-01-30", 965.0, "Veo 3.1", "Google"],
["2026-02-04", 1000.0, "Kling 3.0 1080p (Pro)", "Kling"],
["2026-02-04", 1021.0, "Kling 3.0 Omni 1080p (Pro)", "Kling"],
["2026-02-09", 910.0, "Vidu Q3 Turbo", "Vidu"],
["2026-03-01", 1137.0, "Dreamina Seedance 2.0", "ByteDance"],
["2026-03-05", 841.0, "LTX-2.3 Fast", "Lightricks"],
["2026-03-05", 884.0, "LTX-2.3 Pro", "Lightricks"],
["2026-03-18", 1138.0, "SkyReels V4", "Skywork AI"],
["2026-03-23", 981.0, "PixVerse V6", "PixVerse"],
["2026-03-31", 949.0, "Veo 3.1 Lite", "Google"],
["2026-04-07", 1032.0, "HappyHorse-1.0", "Alibaba-ATH"],
["2026-05-01", 1056.0, "Grok Imagine Video 1.5", "xAI"],
["2026-06-12", 1037.0, "Wan2.7-260612", "Alibaba"],
["2026-06-22", 1049.0, "HappyHorse-1.1", "Alibaba-ATH"],
["2026-07-01", 1138.0, "FLUX 3", "FLUX"],
["2026-07-01", 1155.0, "Dreamina Seedance 2.5", "ByteDance"],
["2026-07-01", 1161.0, "MiniMax H3 (768p)", "MiniMax"],
["2026-08-01", 989.0, "Grok Imagine Video 1.5 Lite", "xAI"],
["2026-08-01", 1064.0, "Agnes-Video-2.5", "Sapiens AI"],
["2026-08-01", 1147.0, "MiniMax H3 Max", "Fal"],
["2026-08-11", 939.0, "LTX-2.5 Pro", "Lightricks"],
["2026-08-11", 943.0, "LTX-2.5 Fast", "Lightricks"],
["2026-08-24", 1175.0, "Wan 3.0", "Alibaba"],
["2026-08-27", 1128.0, "Gemini Omni Flash 1.1", "Google"],
["2026-09-01", 828.0, "Boreal", "Creatify"],
["2026-09-01", 1041.0, "MAGI-2 Preview", "Sand.ai"],
["2026-09-01", 1129.0, "P-Video-2-Pro", "Pruna AI"],
["2026-09-30", 1153.0, "HeyGen Video 1", "HeyGen"],
["2026-09-30", 1168.0, "Utopai X", "Utopai"],
["2026-10-01", 1155.0, "FLUX 3 Max", "Fal"],
["2026-10-29", 1125.0, "Pika H3 Flex", "Pika"]
]
L = {'ylabel': 'Video Quality Elo (AA-Video-T2V v2.1)', 'title': 'Video Generation Capability — New Scale', 'sub': 'Elo from blind human votes for {n} models ({lo} – {hi})  ·  yellow staircase: best at each point in time  ·  scale: AA-Video-T2V v2.1, snapshot of 10 November 2026', 'growth': '{a:.0f} → {b:.0f}\n(+{d:.0f} points)', 'cloud': 'other measured models', 'credit': 'Source: artificialanalysis.ai (Video Arena, v2.1)  ·  Compiled by Prof. Dr. Oğuz Ergin'}

COLORS = {"Anthropic": "#d97757", "OpenAI": "#10a37f", "Google": "#4285F4", "xAI": "#1da1f2",
          "Meta": "#0668E1", "DeepSeek": "#ef4444", "Alibaba": "#7C3AED", "Moonshot": "#14B8A6",
          "Z.ai": "#BE185D", "MiniMax": "#C77DFF", "Mistral": "#fa8005", "ByteDance": "#22D3EE",
          "Microsoft": "#F25022", "Amazon": "#ff9900", "NVIDIA": "#76b900", "Midjourney": "#a78bfa",
          "FLUX": "#fbbf24", "Ideogram": "#f472b6", "Stability": "#38bdf8", "Kling": "#fb923c",
          "Luma": "#2dd4bf", "Runway": "#8b5cf6", "Baidu": "#2932e1", "Recraft": "#e879f9",
          "Tencent": "#0ea5e9", "Alibaba Wan": "#7C3AED", "Pika": "#f43f5e",
          "Skywork AI": "#84cc16", "Lightricks": "#6366f1", "Vidu": "#0ea5e9",
          "PixVerse": "#f43f5e", "Alibaba-ATH": "#2dd4bf", "Pruna AI": "#94a3b8",
          "HeyGen": "#8b5cf6", "Utopai": "#14b8a6", "Fal": "#f59e0b", "Sand.ai": "#eab308",
          "Creatify": "#a3a3a3", "Sapiens AI": "#737373", "Krea": "#f97316",
          "Leonardo.Ai": "#c084fc", "HiDream": "#14b8a6", "Bria": "#a3e635",
          "Meituan": "#fbbf24", "InclusionAI": "#9ca3af", "VectorSpaceLab": "#9ca3af",
          }
OTHER = "#4a5160"

df = pd.DataFrame(DATA, columns=["date", "ii", "name", "comp"])
df["Date"] = pd.to_datetime(df["date"])
df = df.sort_values("Date").reset_index(drop=True)

# --- sinir: o gune kadarki en iyi ---
front, best = [], -1
for _, r in df.iterrows():
    if r["ii"] > best:
        best = r["ii"]
        if front and front[-1]["Date"] == r["Date"]:
            front[-1] = r
        else:
            front.append(r)
fr = pd.DataFrame(front).reset_index(drop=True)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(26, 14))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")

# arka plan bulutu
ax.scatter(df["Date"], df["ii"], s=34, c="#262c36", alpha=.85, edgecolors="none", zorder=1)

# sinir merdiveni
ax.step(fr["Date"], fr["ii"], where="post", color="#ffd166", lw=3.0, zorder=3, alpha=.95)
ax.fill_between(fr["Date"], fr["ii"], step="post", color="#ffd166", alpha=.05, zorder=2)
for _, r in fr.iterrows():
    ax.scatter([r["Date"]], [r["ii"]], s=230, c=COLORS.get(r["comp"], OTHER),
               edgecolors="white", linewidths=2.0, zorder=5)

# sinir etiketleri: PIKSEL uzayinda 2 boyutlu cakisma kontrolu
# (kademe farki tek basina yetmiyor: noktalarin kendi yuksekligi de degisiyor)
PX_DAY = (26 * 105 * 0.93) / max(1, (df["Date"].max() - df["Date"].min()).days)
YLIM = fr["ii"].max() + (fr["ii"].max() - min(df["ii"])) * 0.22
PX_UNIT = (14 * 105 * 0.78) / max(1, YLIM - (min(df["ii"]) - 40))
TIERS = [26, -34, 66, -74, 106, -114, 146, -154]
x0 = df["Date"].min().toordinal()
boxes = []
for _, r in fr.iterrows():
    cx = (r["Date"].toordinal() - x0) * PX_DAY
    hw = len(r["name"]) * 4.8 + 18
    tier = TIERS[-1]
    for t in TIERS:
        cy = r["ii"] * PX_UNIT + t
        if all(abs(cx - bx) > (hw + bw) or abs(cy - by) > 34 for bx, by, bw in boxes):
            tier = t; break
    boxes.append((cx, r["ii"] * PX_UNIT + tier, hw))
    ax.annotate(r["name"], (r["Date"], r["ii"]), xytext=(0, tier), textcoords="offset points",
                fontsize=13, color="#e6edf3", fontweight="bold", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.30", fc="#161b22", ec=COLORS.get(r["comp"], OTHER), lw=1.6, alpha=.96),
                arrowprops=dict(arrowstyle="-", color=COLORS.get(r["comp"], OTHER), lw=1.1, alpha=.55,
                                shrinkA=2, shrinkB=6))

ax.set_ylabel(L["ylabel"], fontsize=17, color="#8b949e", labelpad=16)
ax.set_ylim(min(df["ii"]) - 40, YLIM)
ax.set_xlim(df["Date"].min() - pd.Timedelta(days=25),
            df["Date"].max() + pd.Timedelta(days=95))   # sagdaki etiket tasmasin
ax.grid(True, axis="y", color="#21262d", lw=1.0)
ax.grid(True, axis="x", color="#161b22", lw=.7)
for s in ax.spines.values(): s.set_color("#30363d")
ax.tick_params(colors="#8b949e", labelsize=14)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

# baslik + alt baslik
son = fr.iloc[-1]
bir_yil = fr.iloc[0]["ii"]
plt.title(L["title"], fontsize=30, color="white", pad=54, fontweight="bold")
ax.text(0.5, 1.045, L["sub"].format(n=len(df), lo=df["Date"].min().strftime("%b %Y"),
                                    hi=df["Date"].max().strftime("%b %Y")),
        transform=ax.transAxes, ha="center", fontsize=15, color="#8b949e", style="italic")

# buyume kutusu
ax.text(0.015, 0.965, L["growth"].format(a=bir_yil, b=son["ii"], k=son["ii"] / bir_yil, d=son["ii"] - bir_yil),
        transform=ax.transAxes, ha="left", va="top", fontsize=17, color="#ffd166", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", fc="#161b22", ec="#ffd166", lw=1.8, alpha=.95))

# lejant (sinirdaki sirketler)
import matplotlib.lines as mlines
comps = list(dict.fromkeys(fr["comp"]))
handles = [mlines.Line2D([], [], marker="o", linestyle="", markersize=13, markerfacecolor=COLORS.get(c, OTHER),
                         markeredgecolor="white", label=c) for c in comps]
handles.append(mlines.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="#262c36",
                             markeredgecolor="none", label=L["cloud"]))
ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="#161b22", edgecolor="#30363d",
          fontsize=14, labelcolor="#c9d1d9", ncol=2)

ax.text(0.995, -0.115, L["credit"], transform=ax.transAxes, ha="right", fontsize=13,
        color="#6e7681", style="italic")
plt.tight_layout()
plt.savefig("video_elo_v21.png", dpi=105, facecolor="#0d1117", bbox_inches="tight")
print("kaydedildi: video_elo_v21.png", len(df), " sinir:", len(fr))

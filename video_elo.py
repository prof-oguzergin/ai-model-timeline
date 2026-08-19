# -*- coding: utf-8 -*-
# Yapay zeka basarimi tek sayiyla: Artificial Analysis Zeka Endeksi (Intelligence Index)
# Veri kaynagi: artificialanalysis.ai  (443 model, 2022-11-30 - 2026-08-06)
# Bu dosya scratchpad/make_ii_scripts.py ile uretildi; veri asagida GOMULU.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# [tarih, endeks, model, sirket]
DATA = [
[
"2024-06-01",
948.1,
"Kling 1.0",
"Kling"
],
[
"2024-06-12",
973.7,
"Ray 1",
"Luma"
],
[
"2024-06-17",
982.7,
"Runway Gen 3 Alpha",
"Runway"
],
[
"2024-08-27",
801.1,
"CogVideoX-5B",
"Z.ai"
],
[
"2024-09-01",
1020.3,
"T2V-01",
"MiniMax"
],
[
"2024-09-19",
1026.8,
"Kling 1.5 Pro",
"Kling"
],
[
"2024-10-01",
934.3,
"Pika 1.5",
"Pika Art"
],
[
"2024-10-10",
762.9,
"Pyramid Flow",
"Open Source"
],
[
"2024-10-21",
935.5,
"Haiper 2.0",
"Haiper"
],
[
"2024-10-22",
1000.0,
"Mochi 1",
"Genmo"
],
[
"2024-12-03",
995.5,
"Hunyuan Video (Fal)",
"Tencent"
],
[
"2024-12-09",
1040.0,
"Sora",
"OpenAI"
],
[
"2024-12-13",
1023.9,
"Pika 2.0",
"Pika Art"
],
[
"2024-12-16",
1111.5,
"Veo 2",
"Google"
],
[
"2024-12-19",
1012.4,
"Kling 1.6 Pro",
"Kling"
],
[
"2024-12-19",
1013.4,
"Kling 1.6 Standard",
"Kling"
],
[
"2025-01-28",
1017.8,
"T2V-01-Director",
"MiniMax"
],
[
"2025-02-10",
943.3,
"Ray 2",
"Luma"
],
[
"2025-02-17",
914.6,
"Step-Video-T2V",
"StepFun"
],
[
"2025-02-25",
1013.4,
"Wan 2.1 14B",
"Alibaba"
],
[
"2025-02-27",
946.8,
"Pika 2.2",
"Pika Art"
],
[
"2025-04-15",
1082.3,
"Kling 2.0",
"Kling"
],
[
"2025-04-21",
1003.6,
"Vidu Q1",
"Vidu"
],
[
"2025-05-16",
1076.3,
"PixVerse V4.5",
"PixVerse"
],
[
"2025-05-21",
1209.9,
"Veo 3 Preview",
"Google"
],
[
"2025-05-29",
1129.1,
"Kling 2.1 Master",
"Kling"
],
[
"2025-06-01",
1173.1,
"Veo 3 Fast Preview",
"Google"
],
[
"2025-06-09",
1135.8,
"Seedance 1.0",
"ByteDance"
],
[
"2025-06-10",
1079.5,
"Seedance 1.0 Mini",
"ByteDance"
],
[
"2025-06-18",
1148.2,
"Hailuo 02 Pro",
"MiniMax"
],
[
"2025-06-18",
1169.5,
"Hailuo 02 Standard",
"MiniMax"
],
[
"2025-07-08",
1049.1,
"Marey",
"Moonvalley"
],
[
"2025-07-28",
948.8,
"Wan 2.2 5B",
"Alibaba"
],
[
"2025-07-28",
1106.7,
"Wan 2.2 A14B",
"Alibaba"
],
[
"2025-07-29",
1211.6,
"Veo 3",
"Google"
],
[
"2025-07-30",
1009.9,
"Motion 2.0",
"Leonardo.Ai"
],
[
"2025-08-26",
1167.7,
"PixVerse V5",
"PixVerse"
],
[
"2025-09-19",
1184.2,
"Ray 3",
"Luma"
],
[
"2025-09-23",
1197.8,
"Kling 2.5 Turbo 1080p",
"Kling"
],
[
"2025-09-24",
1157.4,
"Wan 2.5 Preview",
"Alibaba"
],
[
"2025-09-30",
1160.5,
"Sora 2 (October)",
"OpenAI"
],
[
"2025-09-30",
1179.9,
"Sora 2 Pro",
"OpenAI"
],
[
"2025-10-15",
1164.2,
"Vidu Q2",
"Vidu"
],
[
"2025-10-15",
1209.7,
"Veo 3.1 Fast Preview",
"Google"
],
[
"2025-10-15",
1212.5,
"Veo 3.1 Preview",
"Google"
],
[
"2025-10-20",
964.9,
"Krea Realtime",
"Krea"
],
[
"2025-10-23",
1120.4,
"LTX-2 Fast",
"Lightricks"
],
[
"2025-10-23",
1121.0,
"LTX-2 Pro",
"Lightricks"
],
[
"2025-10-28",
1170.5,
"Hailuo 2.3",
"MiniMax"
],
[
"2025-11-20",
1016.3,
"HunyuanVideo-1.5 (Fal)",
"Tencent"
],
[
"2025-11-22",
1076.4,
"Pika 2.5",
"Pika Art"
],
[
"2025-12-01",
1187.1,
"Kling O1 Standard (December)",
"Kling"
],
[
"2025-12-01",
1190.9,
"Kling O1 Pro (December)",
"Kling"
],
[
"2025-12-01",
1213.7,
"Runway Gen-4.5",
"Runway"
],
[
"2025-12-03",
1184.6,
"Kling 2.6 Pro (December)",
"Kling"
],
[
"2025-12-03",
1194.3,
"PixVerse V5.5",
"PixVerse"
],
[
"2025-12-15",
1167.5,
"Sora 2 (December)",
"OpenAI"
],
[
"2025-12-15",
1170.9,
"Seedance 1.5 pro",
"ByteDance"
],
[
"2025-12-16",
1184.8,
"Wan 2.6",
"Alibaba"
],
[
"2026-01-13",
1181.8,
"Kling 2.6 Standard (January)",
"Kling"
],
[
"2026-01-13",
1186.2,
"Kling O1 Standard (January)",
"Kling"
],
[
"2026-01-13",
1194.6,
"Kling O1 Pro (January)",
"Kling"
],
[
"2026-01-13",
1195.6,
"Kling 2.6 Pro (January)",
"Kling"
],
[
"2026-01-26",
1128.5,
"PixVerse V5.6 (January)",
"PixVerse"
],
[
"2026-01-27",
1221.6,
"grok-imagine-video",
"xAI"
],
[
"2026-01-30",
1197.9,
"Veo 3.1 Fast",
"Google"
],
[
"2026-01-30",
1199.2,
"Veo 3.1",
"Google"
],
[
"2026-01-30",
1214.4,
"Vidu Q3 Pro",
"Vidu"
],
[
"2026-02-04",
1209.6,
"Kling 3.0 720p (Standard)",
"Kling"
],
[
"2026-02-04",
1213.7,
"Kling 3.0 Omni 720p (Standard)",
"Kling"
],
[
"2026-02-04",
1227.4,
"Kling 3.0 Omni 1080p (Pro)",
"Kling"
],
[
"2026-02-04",
1239.0,
"Kling 3.0 1080p (Pro)",
"Kling"
],
[
"2026-02-09",
1105.8,
"Vidu Q3 Turbo",
"Vidu"
],
[
"2026-02-25",
1211.3,
"PixVerse V5.6",
"PixVerse"
],
[
"2026-02-26",
1062.0,
"P-Video",
"Pruna AI"
],
[
"2026-03-05",
1105.1,
"LTX-2.3 Pro",
"Lightricks"
],
[
"2026-03-05",
1120.1,
"LTX-2.3 Fast",
"Lightricks"
],
[
"2026-03-18",
1209.7,
"SkyReels V4",
"Skywork AI"
],
[
"2026-03-18",
1265.4,
"Dreamina Seedance 2.0 720p",
"ByteDance"
],
[
"2026-03-23",
1209.2,
"PixVerse V6",
"PixVerse"
],
[
"2026-03-31",
1206.5,
"Veo 3.1 Lite",
"Google"
],
[
"2026-04-07",
1283.9,
"HappyHorse-1.0",
"Alibaba-ATH"
],
[
"2026-04-25",
1214.6,
"Wan 2.7",
"Alibaba"
],
[
"2026-04-30",
1215.2,
"Bach-1.0 Preview",
"Video Rebirth"
],
[
"2026-05-19",
1322.3,
"Gemini Omni Flash",
"Google"
],
[
"2026-05-20",
1051.3,
"Agnes-Video-V2.0",
"Sapiens AI"
],
[
"2026-06-12",
1242.0,
"Wan2.7-260612",
"Alibaba"
],
[
"2026-06-22",
1261.8,
"HappyHorse-1.1",
"Alibaba-ATH"
],
[
"2026-07-30",
1304.9,
"MiniMax H3",
"MiniMax"
]
]

L = {'ylabel': 'Video Quality Elo (Artificial Analysis Arena)', 'title': 'Video Generation Capability Over Time — A Single Number', 'sub': 'Elo from blind human votes for {n} text-to-video models ({lo} – {hi})  ·  yellow steps: best at the time', 'growth': 'Last 12 months\n{a:.0f} → {b:.0f}  (+{d:.0f} pts)', 'cloud': 'other measured models', 'credit': 'Source: artificialanalysis.ai (Video Arena)  ·  Compiled by Prof. Dr. Oğuz Ergin'}

COLORS = {"Anthropic": "#d97757", "OpenAI": "#10a37f", "Google": "#4285F4", "xAI": "#1da1f2",
          "Meta": "#0668E1", "DeepSeek": "#ef4444", "Alibaba": "#7C3AED", "Moonshot": "#14B8A6",
          "Z.ai": "#BE185D", "MiniMax": "#C77DFF", "Mistral": "#fa8005", "ByteDance": "#22D3EE",
          "Microsoft": "#F25022", "Amazon": "#ff9900", "NVIDIA": "#76b900", "Midjourney": "#a78bfa",
          "FLUX": "#fbbf24", "Ideogram": "#f472b6", "Stability": "#38bdf8", "Kling": "#fb923c",
          "Luma": "#2dd4bf", "Runway": "#8b5cf6", "Baidu": "#2932e1", "Recraft": "#e879f9",
          "Tencent": "#0ea5e9", "Alibaba Wan": "#7C3AED", "Pika": "#f43f5e"}
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
ax.grid(True, axis="y", color="#21262d", lw=1.0)
ax.grid(True, axis="x", color="#161b22", lw=.7)
for s in ax.spines.values(): s.set_color("#30363d")
ax.tick_params(colors="#8b949e", labelsize=14)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

# baslik + alt baslik
son = fr.iloc[-1]
bir_yil = fr[fr["Date"] <= son["Date"] - pd.Timedelta(days=365)]["ii"].max()
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
plt.savefig("video_elo.png", dpi=105, facecolor="#0d1117", bbox_inches="tight")
print("kaydedildi: video_elo.png", len(df), " sinir:", len(fr))

# -*- coding: utf-8 -*-
# Yapay zeka basarimi tek sayiyla: Artificial Analysis Zeka Endeksi (Intelligence Index)
# Veri kaynagi: artificialanalysis.ai  (435 model, 2022-11-30 - 2026-07-24)
# Bu dosya scratchpad/make_ii_scripts.py ile uretildi; veri asagida GOMULU.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# [tarih, endeks, model, sirket]
DATA = [
[
"2022-09-28",
743.9,
"DALLE 2",
"OpenAI"
],
[
"2022-10-01",
659.9,
"Stable Diffusion 1.5",
"Stability"
],
[
"2022-12-07",
752.0,
"Stable Diffusion 2.1",
"Stability"
],
[
"2023-07-23",
876.1,
"Stable Diffusion XL 1.0",
"Stability"
],
[
"2023-09-20",
950.2,
"DALLE 3",
"OpenAI"
],
[
"2023-09-26",
959.2,
"DALLE 3 HD",
"OpenAI"
],
[
"2023-11-10",
903.5,
"Stable Diffusion 1.6",
"Stability"
],
[
"2023-11-29",
900.4,
"Amazon Titan G1 (Standard)",
"Amazon"
],
[
"2023-12-20",
1067.7,
"Midjourney v6",
"Midjourney"
],
[
"2024-02-21",
906.1,
"SDXL Lightning",
"Bytedance"
],
[
"2024-02-22",
1026.8,
"Stable Diffusion 3 Large",
"Stability"
],
[
"2024-02-24",
908.1,
"Stable Diffusion 3 Large Turbo",
"Stability"
],
[
"2024-02-27",
967.2,
"Playground v2.5",
"Playground AI"
],
[
"2024-02-28",
1034.1,
"Ideogram v1",
"Ideogram"
],
[
"2024-03-13",
976.3,
"Recraft 20B",
"Recraft"
],
[
"2024-06-12",
912.9,
"Stable Diffusion 3 Medium",
"Stability"
],
[
"2024-06-13",
1009.0,
"Phoenix 0.9 Ultra",
"Leonardo.Ai"
],
[
"2024-07-30",
1053.6,
"Midjourney v6.1",
"Midjourney"
],
[
"2024-08-01",
1027.8,
"FLUX.1 [dev]",
"FLUX"
],
[
"2024-08-01",
1068.1,
"FLUX.1 [pro]",
"FLUX"
],
[
"2024-08-02",
1000.0,
"FLUX.1 [schnell]",
"FLUX"
],
[
"2024-08-06",
912.0,
"Amazon Titan G1 v2 (Standard)",
"Amazon"
],
[
"2024-08-21",
1058.4,
"Ideogram v2 Turbo",
"Ideogram"
],
[
"2024-08-21",
1072.9,
"Ideogram v2",
"Ideogram"
],
[
"2024-09-16",
1009.8,
"Playground v3 (beta)",
"Playground AI"
],
[
"2024-10-02",
1088.2,
"FLUX1.1 [pro]",
"FLUX"
],
[
"2024-10-22",
1021.7,
"Stable Diffusion 3.5 Large",
"Stability"
],
[
"2024-10-22",
1024.5,
"Stable Diffusion 3.5 Large Turbo",
"Stability"
],
[
"2024-10-29",
947.4,
"Stable Diffusion 3.5 Medium",
"Stability"
],
[
"2024-10-30",
1068.1,
"Recraft V3",
"Recraft"
],
[
"2024-11-06",
1096.9,
"FLUX1.1 [pro] Ultra",
"FLUX"
],
[
"2024-11-25",
977.4,
"Runway Gen-4 Image",
"Runway"
],
[
"2024-12-02",
986.6,
"Luma Photon Flash",
"Luma"
],
[
"2024-12-02",
1064.4,
"Luma Photon",
"Luma"
],
[
"2024-12-12",
923.5,
"Grok 2",
"xAI"
],
[
"2024-12-16",
1107.6,
"Imagen 3",
"Google"
],
[
"2024-12-18",
995.2,
"Phoenix 1.0 Fast",
"Leonardo.Ai"
],
[
"2024-12-18",
1032.4,
"Phoenix 1.0 Ultra",
"Leonardo.Ai"
],
[
"2025-01-25",
968.8,
"Lumina Image v2",
"OpenGVLab"
],
[
"2025-01-27",
712.1,
"Janus Pro",
"DeepSeek"
],
[
"2025-02-18",
1045.5,
"Infinity 8B",
"Bytedance"
],
[
"2025-02-27",
1015.1,
"Ideogram v2a Turbo",
"Ideogram"
],
[
"2025-02-27",
1015.6,
"Ideogram v2a",
"Ideogram"
],
[
"2025-02-28",
1055.3,
"Image-01",
"MiniMax"
],
[
"2025-03-22",
933.1,
"Sana Sprint 1.6B",
"NVIDIA"
],
[
"2025-03-24",
1098.9,
"Reve Image (Halfmoon)",
"Reve"
],
[
"2025-03-26",
1079.1,
"Ideogram 3.0",
"Ideogram"
],
[
"2025-04-03",
1070.6,
"Midjourney v7 Alpha",
"Midjourney"
],
[
"2025-04-07",
1058.1,
"HiDream-I1-Fast",
"HiDream"
],
[
"2025-04-07",
1062.2,
"HiDream-I1-Dev",
"HiDream"
],
[
"2025-04-15",
1144.7,
"Seedream 3.0",
"ByteDance"
],
[
"2025-04-23",
1137.0,
"GPT Image 1",
"OpenAI"
],
[
"2025-05-20",
900.1,
"Bagel",
"Bytedance"
],
[
"2025-05-20",
1091.1,
"FLUX.1 Kontext [pro]",
"FLUX"
],
[
"2025-05-29",
1124.2,
"FLUX.1 Kontext [max]",
"FLUX"
],
[
"2025-06-10",
1106.7,
"Vivago 2.0",
"HiDream"
],
[
"2025-06-16",
904.8,
"OmniGen V2",
"VectorSpaceLab"
],
[
"2025-06-17",
1005.2,
"Krea 1",
"Krea"
],
[
"2025-06-26",
1076.5,
"Imagen 4 Fast",
"Google"
],
[
"2025-06-26",
1101.5,
"Imagen 4 Standard",
"Google"
],
[
"2025-06-26",
1172.2,
"Imagen 4 Ultra",
"Google"
],
[
"2025-07-22",
901.3,
"Bria 3.2",
"Bria"
],
[
"2025-07-28",
1129.0,
"Kolors 2.1",
"Kling"
],
[
"2025-07-31",
1028.8,
"FLUX.1 Krea [dev]",
"FLUX"
],
[
"2025-08-01",
1091.0,
"Dreamina 3.1",
"Bytedance"
],
[
"2025-08-04",
1062.3,
"Qwen Image",
"Alibaba"
],
[
"2025-08-05",
1109.0,
"Lucid Origin Fast",
"Leonardo.Ai"
],
[
"2025-08-05",
1120.5,
"Lucid Origin Ultra",
"Leonardo.Ai"
],
[
"2025-08-26",
1152.4,
"Nano Banana (Gemini 2.5 Flash Image)",
"Google"
],
[
"2025-09-08",
1064.0,
"HunyuanImage 2.1",
"Tencent"
],
[
"2025-09-08",
1192.4,
"Seedream 4.0",
"ByteDance"
],
[
"2025-09-11",
1079.5,
"SRPO",
"Tencent"
],
[
"2025-09-23",
1146.6,
"Wan 2.5 Preview",
"Alibaba"
],
[
"2025-09-28",
1125.4,
"HunyuanImage 3.0 (Fal)",
"Tencent"
],
[
"2025-10-06",
1092.7,
"GPT Image 1 Mini",
"OpenAI"
],
[
"2025-10-20",
1135.4,
"Vivago 2.1",
"HiDream"
],
[
"2025-10-30",
1068.8,
"FIBO",
"Bria"
],
[
"2025-11-04",
1043.2,
"MAI Image 1",
"Microsoft AI"
],
[
"2025-11-20",
1148.9,
"ImagineArt 1.5 Preview",
"ImagineArt"
],
[
"2025-11-20",
1225.5,
"Nano Banana Pro (Gemini 3 Pro Image)",
"Google"
],
[
"2025-11-25",
1153.7,
"FLUX.2 [dev]",
"FLUX"
],
[
"2025-11-25",
1180.2,
"FLUX.2 [flex]",
"FLUX"
],
[
"2025-11-25",
1186.9,
"FLUX.2 [pro]",
"FLUX"
],
[
"2025-11-28",
1102.0,
"Vidu Q2",
"Vidu"
],
[
"2025-12-02",
1071.9,
"P-Image",
"Pruna AI"
],
[
"2025-12-02",
1102.1,
"Z-Image Turbo",
"Alibaba"
],
[
"2025-12-05",
1031.2,
"LongCat Image",
"Meituan"
],
[
"2025-12-05",
1169.2,
"Seedream 4.5",
"ByteDance"
],
[
"2025-12-16",
1196.0,
"FLUX.2 [max]",
"FLUX"
],
[
"2025-12-16",
1262.7,
"GPT Image 1.5",
"OpenAI"
],
[
"2025-12-17",
1136.3,
"Wan 2.6 Image",
"Alibaba"
],
[
"2025-12-21",
1136.5,
"FLUX.2 [dev] Flash",
"Fal"
],
[
"2025-12-21",
1153.6,
"FLUX.2 [dev] Turbo",
"Fal"
],
[
"2025-12-30",
1157.5,
"Qwen Image Max 2512",
"Alibaba"
],
[
"2026-01-13",
1050.0,
"GLM-Image",
"Z.ai"
],
[
"2026-01-15",
968.8,
"FLUX.2 [klein] Base 4B",
"FLUX"
],
[
"2026-01-15",
1057.5,
"FLUX.2 [klein] 4B",
"FLUX"
],
[
"2026-01-15",
1087.2,
"FLUX.2 [klein] Base 9B",
"FLUX"
],
[
"2026-01-15",
1120.5,
"FLUX.2 [klein] 9B",
"FLUX"
],
[
"2026-01-16",
1078.9,
"Qwen Image Plus 2601",
"Alibaba"
],
[
"2026-01-19",
1138.6,
"Wan2.6 Text to Image",
"Alibaba"
],
[
"2026-01-25",
1119.2,
"HunyuanImage 3.0 Instruct (Fal)",
"Tencent"
],
[
"2026-01-27",
1037.6,
"Z-Image Base",
"Alibaba"
],
[
"2026-01-28",
1110.0,
"Eigen Image",
"Eigen AI"
],
[
"2026-01-28",
1180.9,
"grok-imagine-image",
"xAI"
],
[
"2026-02-02",
1254.1,
"Riverflow 2.0",
"Sourceful"
],
[
"2026-02-04",
1097.3,
"Kling Image 3.0 Omni",
"Kling"
],
[
"2026-02-13",
1118.7,
"Seedream 5.0 Lite",
"Bytedance"
],
[
"2026-02-17",
1136.5,
"Recraft V4",
"Recraft"
],
[
"2026-02-17",
1139.7,
"Recraft V4 Pro",
"Recraft"
],
[
"2026-02-26",
1262.4,
"Nano Banana 2 (Gemini 3.1 Flash Image Preview)",
"Google"
],
[
"2026-03-03",
1104.6,
"Qwen Image 2.0 (2026-03-03)",
"Alibaba"
],
[
"2026-03-19",
1189.2,
"MAI-Image-2",
"Microsoft AI"
],
[
"2026-04-03",
1109.1,
"Wan 2.7",
"Alibaba"
],
[
"2026-04-03",
1119.8,
"Wan 2.7 Pro",
"Alibaba"
],
[
"2026-04-03",
1204.4,
"grok-imagine-image-quality",
"xAI"
],
[
"2026-04-08",
1125.0,
"image-1",
"Api Airforce"
],
[
"2026-04-14",
1159.2,
"MAI-Image-2-Efficient",
"Microsoft AI"
],
[
"2026-04-15",
1162.8,
"ERNIE Image Turbo",
"Baidu"
],
[
"2026-04-15",
1166.4,
"ERNIE Image",
"Baidu"
],
[
"2026-04-16",
1168.5,
"ImagineArt 2.0",
"ImagineArt"
],
[
"2026-04-21",
1339.4,
"GPT Image 2",
"OpenAI"
],
[
"2026-04-22",
1173.5,
"Qwen Image 2.0 Pro (2026-04-22)",
"Alibaba"
],
[
"2026-05-05",
1150.2,
"Luma UNI 1",
"Luma"
],
[
"2026-05-05",
1174.7,
"Luma UNI 1 Max",
"Luma"
],
[
"2026-05-08",
1078.8,
"HiDream-O1-Image-Dev",
"HiDream"
],
[
"2026-05-08",
1115.5,
"HiDream-O1-Image",
"HiDream"
],
[
"2026-05-12",
984.1,
"Step Image Edit 2",
"StepFun"
],
[
"2026-05-14",
1153.5,
"Recraft V4.1",
"Recraft"
],
[
"2026-05-14",
1157.4,
"Recraft V4.1 Pro",
"Recraft"
],
[
"2026-05-14",
1188.4,
"HiDream-O1-Image-Dev-2604",
"HiDream"
],
[
"2026-05-14",
1200.8,
"Recraft V4.1 Utility Pro",
"Recraft"
],
[
"2026-05-14",
1204.9,
"Recraft V4.1 Utility",
"Recraft"
],
[
"2026-05-26",
1190.3,
"Krea 2 Large",
"Krea"
],
[
"2026-05-26",
1190.6,
"Krea 2 Medium",
"Krea"
],
[
"2026-05-31",
1219.4,
"Cosmos3-Super-Text2Image (agentic)",
"NVIDIA"
],
[
"2026-06-02",
1208.1,
"MAI-Image-2.5-Flash",
"Microsoft AI"
],
[
"2026-06-02",
1269.7,
"MAI-Image-2.5",
"Microsoft AI"
],
[
"2026-06-03",
1153.7,
"Ideogram 4.0",
"Ideogram"
],
[
"2026-06-03",
1174.1,
"Ideogram 4.0 (Quality)",
"Ideogram"
],
[
"2026-06-03",
1187.7,
"Krea 2 Medium Turbo",
"Krea"
],
[
"2026-06-03",
1238.4,
"Reve 2.0",
"Reve"
],
[
"2026-06-04",
1244.9,
"HiDream-O1-Image-1.5",
"HiDream"
],
[
"2026-06-30",
1262.7,
"Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
"Google"
],
[
"2026-07-08",
1239.5,
"Seedream 5.0 Pro",
"Bytedance"
],
[
"2026-07-09",
1299.0,
"Reve 2.1",
"Reve"
],
[
"2026-07-13",
1143.3,
"Ideogram 4.0 Fast",
"Fal"
],
[
"2026-07-13",
1148.8,
"Ideogram 4.0 Instant",
"Fal"
],
[
"2026-07-13",
1159.0,
"Ideogram 4.0 Fast (Quality)",
"Fal"
],
[
"2026-07-31",
1176.0,
"Cosmos3-Super-Text2Image-4Step",
"NVIDIA"
]
]

L = {'ylabel': 'Image Quality Elo (Artificial Analysis Arena)', 'title': 'Image Generation Capability Over Time — A Single Number', 'sub': 'Elo from blind human votes for {n} models ({lo} – {hi})  ·  yellow steps: best at the time', 'growth': 'Last 12 months\n{a:.0f} → {b:.0f}  (+{d:.0f} pts)', 'cloud': 'other measured models', 'credit': 'Source: artificialanalysis.ai (Image Arena)  ·  Compiled by Prof. Dr. Oğuz Ergin'}

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
plt.savefig("image_elo.png", dpi=105, facecolor="#0d1117", bbox_inches="tight")
print("kaydedildi: image_elo.png", len(df), " sinir:", len(fr))

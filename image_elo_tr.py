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
"2022-09-28",
745.5,
"DALLE 2",
"OpenAI"
],
[
"2022-10-01",
669.6,
"Stable Diffusion 1.5",
"Stability"
],
[
"2022-12-07",
753.2,
"Stable Diffusion 2.1",
"Stability"
],
[
"2023-07-23",
883.1,
"Stable Diffusion XL 1.0",
"Stability"
],
[
"2023-09-20",
968.6,
"DALLE 3",
"OpenAI"
],
[
"2023-09-26",
967.3,
"DALLE 3 HD",
"OpenAI"
],
[
"2023-11-10",
914.4,
"Stable Diffusion 1.6",
"Stability"
],
[
"2023-11-29",
914.4,
"Amazon Titan G1 (Standard)",
"Amazon"
],
[
"2023-12-20",
1077.1,
"Midjourney v6",
"Midjourney"
],
[
"2024-02-21",
910.9,
"SDXL Lightning",
"Bytedance"
],
[
"2024-02-22",
1043.1,
"Stable Diffusion 3 Large",
"Stability"
],
[
"2024-02-24",
917.0,
"Stable Diffusion 3 Large Turbo",
"Stability"
],
[
"2024-02-27",
971.8,
"Playground v2.5",
"Playground AI"
],
[
"2024-02-28",
1048.5,
"Ideogram v1",
"Ideogram"
],
[
"2024-03-13",
983.8,
"Recraft 20B",
"Recraft"
],
[
"2024-06-12",
922.6,
"Stable Diffusion 3 Medium",
"Stability"
],
[
"2024-06-13",
1029.4,
"Phoenix 0.9 Ultra",
"Leonardo.Ai"
],
[
"2024-07-30",
1058.4,
"Midjourney v6.1",
"Midjourney"
],
[
"2024-08-01",
1040.2,
"FLUX.1 [dev]",
"FLUX"
],
[
"2024-08-01",
1083.8,
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
923.1,
"Amazon Titan G1 v2 (Standard)",
"Amazon"
],
[
"2024-08-21",
1076.2,
"Ideogram v2 Turbo",
"Ideogram"
],
[
"2024-08-21",
1083.5,
"Ideogram v2",
"Ideogram"
],
[
"2024-09-16",
1016.0,
"Playground v3 (beta)",
"Playground AI"
],
[
"2024-10-02",
1094.9,
"FLUX1.1 [pro]",
"FLUX"
],
[
"2024-10-22",
1034.7,
"Stable Diffusion 3.5 Large",
"Stability"
],
[
"2024-10-22",
1034.8,
"Stable Diffusion 3.5 Large Turbo",
"Stability"
],
[
"2024-10-29",
963.5,
"Stable Diffusion 3.5 Medium",
"Stability"
],
[
"2024-10-30",
1077.1,
"Recraft V3",
"Recraft"
],
[
"2024-11-06",
1105.9,
"FLUX1.1 [pro] Ultra",
"FLUX"
],
[
"2024-11-25",
992.6,
"Runway Gen-4 Image",
"Runway"
],
[
"2024-12-02",
1003.9,
"Luma Photon Flash",
"Luma"
],
[
"2024-12-02",
1076.0,
"Luma Photon",
"Luma"
],
[
"2024-12-16",
1126.0,
"Imagen 3",
"Google"
],
[
"2024-12-18",
1007.3,
"Phoenix 1.0 Fast",
"Leonardo.Ai"
],
[
"2024-12-18",
1041.8,
"Phoenix 1.0 Ultra",
"Leonardo.Ai"
],
[
"2025-01-25",
977.7,
"Lumina Image v2",
"OpenGVLab"
],
[
"2025-01-27",
721.5,
"Janus Pro",
"DeepSeek"
],
[
"2025-02-18",
1056.7,
"Infinity 8B",
"Bytedance"
],
[
"2025-02-27",
1027.3,
"Ideogram v2a",
"Ideogram"
],
[
"2025-02-27",
1028.5,
"Ideogram v2a Turbo",
"Ideogram"
],
[
"2025-02-28",
1071.5,
"Image-01",
"MiniMax"
],
[
"2025-03-22",
940.0,
"Sana Sprint 1.6B",
"NVIDIA"
],
[
"2025-03-24",
1113.8,
"Reve Image (Halfmoon)",
"Reve"
],
[
"2025-03-26",
1100.2,
"Ideogram 3.0",
"Ideogram"
],
[
"2025-04-03",
1093.4,
"Midjourney v7 Alpha",
"Midjourney"
],
[
"2025-04-07",
1063.5,
"HiDream-I1-Fast",
"HiDream"
],
[
"2025-04-07",
1071.4,
"HiDream-I1-Dev",
"HiDream"
],
[
"2025-04-15",
1160.7,
"Seedream 3.0",
"ByteDance"
],
[
"2025-04-23",
1202.0,
"GPT Image 1",
"OpenAI"
],
[
"2025-05-20",
908.2,
"Bagel",
"Bytedance"
],
[
"2025-05-20",
1110.6,
"FLUX.1 Kontext [pro]",
"FLUX"
],
[
"2025-05-29",
1144.3,
"FLUX.1 Kontext [max]",
"FLUX"
],
[
"2025-06-10",
1126.2,
"Vivago 2.0",
"HiDream"
],
[
"2025-06-16",
917.0,
"OmniGen V2",
"VectorSpaceLab"
],
[
"2025-06-17",
1013.2,
"Krea 1",
"Krea"
],
[
"2025-06-26",
1098.8,
"Imagen 4 Fast",
"Google"
],
[
"2025-06-26",
1124.3,
"Imagen 4 Standard",
"Google"
],
[
"2025-06-26",
1189.4,
"Imagen 4 Ultra",
"Google"
],
[
"2025-07-22",
906.2,
"Bria 3.2",
"Bria"
],
[
"2025-07-28",
1143.0,
"Kolors 2.1",
"Kling"
],
[
"2025-07-31",
1039.8,
"FLUX.1 Krea [dev]",
"FLUX"
],
[
"2025-08-01",
1107.2,
"Dreamina 3.1",
"Bytedance"
],
[
"2025-08-04",
1076.2,
"Qwen Image",
"Alibaba"
],
[
"2025-08-05",
1111.4,
"Lucid Origin Fast",
"Leonardo.Ai"
],
[
"2025-08-05",
1123.2,
"Lucid Origin Ultra",
"Leonardo.Ai"
],
[
"2025-08-26",
1187.0,
"Nano Banana (Gemini 2.5 Flash Image)",
"Google"
],
[
"2025-09-08",
1078.0,
"HunyuanImage 2.1",
"Tencent"
],
[
"2025-09-08",
1224.8,
"Seedream 4.0",
"ByteDance"
],
[
"2025-09-11",
1084.0,
"SRPO",
"Tencent"
],
[
"2025-09-23",
1147.5,
"Wan 2.5 Preview",
"Alibaba"
],
[
"2025-09-28",
1136.2,
"HunyuanImage 3.0 (Fal)",
"Tencent"
],
[
"2025-10-06",
1107.1,
"GPT Image 1 Mini",
"OpenAI"
],
[
"2025-10-20",
1152.0,
"Vivago 2.1",
"HiDream"
],
[
"2025-10-30",
1073.8,
"FIBO",
"Bria"
],
[
"2025-11-04",
1062.4,
"MAI Image 1",
"Microsoft AI"
],
[
"2025-11-20",
1164.6,
"ImagineArt 1.5 Preview",
"ImagineArt"
],
[
"2025-11-20",
1296.7,
"Nano Banana Pro (Gemini 3 Pro Image)",
"Google"
],
[
"2025-11-25",
1198.7,
"FLUX.2 [dev]",
"FLUX"
],
[
"2025-11-25",
1206.2,
"FLUX.2 [pro]",
"FLUX"
],
[
"2025-11-25",
1221.9,
"FLUX.2 [flex]",
"FLUX"
],
[
"2025-11-27",
1031.7,
"FIBO Lite",
"Bria"
],
[
"2025-11-28",
1116.5,
"Vidu Q2",
"Vidu"
],
[
"2025-12-02",
1094.0,
"P-Image",
"Pruna AI"
],
[
"2025-12-02",
1133.3,
"Z-Image Turbo",
"Alibaba"
],
[
"2025-12-05",
1051.3,
"LongCat Image",
"Meituan"
],
[
"2025-12-05",
1204.5,
"Seedream 4.5",
"ByteDance"
],
[
"2025-12-16",
1225.6,
"FLUX.2 [max]",
"FLUX"
],
[
"2025-12-16",
1306.7,
"GPT Image 1.5",
"OpenAI"
],
[
"2025-12-17",
1206.4,
"Wan 2.6 Image",
"Alibaba"
],
[
"2025-12-21",
1176.7,
"FLUX.2 [dev] Flash",
"Fal"
],
[
"2025-12-21",
1197.9,
"FLUX.2 [dev] Turbo",
"Fal"
],
[
"2025-12-30",
1171.2,
"Qwen Image Max 2512",
"Alibaba"
],
[
"2026-01-13",
1063.3,
"GLM-Image",
"Z.ai"
],
[
"2026-01-15",
972.8,
"FLUX.2 [klein] Base 4B",
"FLUX"
],
[
"2026-01-15",
1059.6,
"FLUX.2 [klein] 4B",
"FLUX"
],
[
"2026-01-15",
1098.4,
"FLUX.2 [klein] Base 9B",
"FLUX"
],
[
"2026-01-15",
1140.6,
"FLUX.2 [klein] 9B",
"FLUX"
],
[
"2026-01-16",
1106.6,
"Qwen Image Plus 2601",
"Alibaba"
],
[
"2026-01-19",
1207.0,
"Wan2.6 Text to Image",
"Alibaba"
],
[
"2026-01-25",
1151.9,
"HunyuanImage 3.0 Instruct (Fal)",
"Tencent"
],
[
"2026-01-27",
1063.4,
"Z-Image Base",
"Alibaba"
],
[
"2026-01-28",
1121.4,
"Eigen Image",
"Eigen AI"
],
[
"2026-01-28",
1216.4,
"grok-imagine-image",
"xAI"
],
[
"2026-02-02",
1275.8,
"Riverflow 2.0",
"Sourceful"
],
[
"2026-02-04",
1113.3,
"Kling Image 3.0 Omni",
"Kling"
],
[
"2026-02-13",
1197.8,
"Seedream 5.0 Lite",
"ByteDance"
],
[
"2026-02-17",
1179.2,
"Recraft V4",
"Recraft"
],
[
"2026-02-17",
1193.3,
"Recraft V4 Pro",
"Recraft"
],
[
"2026-02-26",
1321.0,
"Nano Banana 2 (Gemini 3.1 Flash Image Preview)",
"Google"
],
[
"2026-03-03",
1134.8,
"Qwen Image 2.0 (2026-03-03)",
"Alibaba"
],
[
"2026-03-19",
1210.4,
"MAI-Image-2",
"Microsoft AI"
],
[
"2026-04-03",
1166.7,
"Wan 2.7",
"Alibaba"
],
[
"2026-04-03",
1179.3,
"Wan 2.7 Pro",
"Alibaba"
],
[
"2026-04-03",
1235.3,
"grok-imagine-image-quality",
"xAI"
],
[
"2026-04-08",
1140.7,
"image-1",
"Api Airforce"
],
[
"2026-04-14",
1183.1,
"MAI-Image-2-Efficient",
"Microsoft AI"
],
[
"2026-04-15",
1115.3,
"ERNIE Image",
"Baidu"
],
[
"2026-04-15",
1115.9,
"ERNIE Image Turbo",
"Baidu"
],
[
"2026-04-16",
1178.7,
"ImagineArt 2.0",
"ImagineArt"
],
[
"2026-04-21",
1370.0,
"GPT Image 2",
"OpenAI"
],
[
"2026-04-22",
1235.0,
"Qwen Image 2.0 Pro (2026-04-22)",
"Alibaba"
],
[
"2026-05-05",
1199.3,
"Luma UNI 1",
"Luma"
],
[
"2026-05-05",
1221.6,
"Luma UNI 1 Max",
"Luma"
],
[
"2026-05-08",
1069.4,
"HiDream-O1-Image-Dev",
"HiDream"
],
[
"2026-05-08",
1174.1,
"HiDream-O1-Image",
"HiDream"
],
[
"2026-05-12",
1003.9,
"Step Image Edit 2",
"StepFun"
],
[
"2026-05-14",
1187.4,
"Recraft V4.1 Pro",
"Recraft"
],
[
"2026-05-14",
1198.1,
"Recraft V4.1",
"Recraft"
],
[
"2026-05-14",
1216.1,
"Recraft V4.1 Utility",
"Recraft"
],
[
"2026-05-14",
1218.1,
"Recraft V4.1 Utility Pro",
"Recraft"
],
[
"2026-05-26",
1214.2,
"Krea 2 Medium",
"Krea"
],
[
"2026-05-26",
1219.7,
"Krea 2 Large",
"Krea"
],
[
"2026-05-31",
1176.6,
"Cosmos3-Super-Text2Image",
"NVIDIA"
],
[
"2026-05-31",
1186.9,
"Cosmos3-Super-Text2Image (agentic)",
"NVIDIA"
],
[
"2026-06-02",
1225.2,
"MAI-Image-2.5-Flash",
"Microsoft AI"
],
[
"2026-06-02",
1302.8,
"MAI-Image-2.5",
"Microsoft AI"
],
[
"2026-06-03",
1212.4,
"Krea 2 Medium Turbo",
"Krea"
],
[
"2026-06-03",
1214.2,
"Ideogram 4.0 (Quality)",
"Ideogram"
],
[
"2026-06-03",
1219.8,
"Ideogram 4.0",
"Ideogram"
],
[
"2026-06-03",
1261.6,
"Reve 2.0",
"Reve"
],
[
"2026-06-04",
1227.1,
"HiDream-O1-Image-1.5",
"HiDream"
],
[
"2026-06-30",
1288.9,
"Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
"Google"
],
[
"2026-07-08",
1279.8,
"Seedream 5.0 Pro",
"ByteDance"
],
[
"2026-07-09",
1322.6,
"Reve 2.1",
"Reve"
],
[
"2026-07-13",
1179.5,
"Ideogram 4.0 Fast",
"Fal"
],
[
"2026-07-13",
1185.4,
"Ideogram 4.0 Instant",
"Fal"
],
[
"2026-07-13",
1199.6,
"Ideogram 4.0 Fast (Quality)",
"Fal"
],
[
"2026-07-20",
1158.0,
"Cosmos3-Super-Text2Image-4Step",
"NVIDIA"
],
[
"2026-07-21",
1269.5,
"Qwen-Image-3.0",
"Alibaba"
],
[
"2026-07-21",
1284.0,
"Qwen-Image-3.0-Pro",
"Alibaba"
],
[
"2026-07-23",
1293.3,
"MAI-Image-2.5-Pro",
"Microsoft AI"
],
[
"2026-07-30",
1108.6,
"P-Image-Ideogram (Very Low)",
"Pruna AI"
],
[
"2026-07-30",
1167.3,
"P-Image-Ideogram (Low)",
"Pruna AI"
],
[
"2026-07-30",
1182.2,
"P-Image-Ideogram (Medium)",
"Pruna AI"
],
[
"2026-07-30",
1199.3,
"P-Image-Ideogram (High)",
"Pruna AI"
],
[
"2026-08-10",
1350.8,
"MAI-Image-2.6-Preview",
"Microsoft AI"
]
]

L = {'ylabel': 'Görsel Kalite Elo (Artificial Analysis Arena)', 'title': 'Görsel Üretme Modellerinin Başarımı — Tek Sayıyla', 'sub': '{n} modelin körlemesine insan oylamasıyla Elo puanı ({lo} – {hi})  ·  sarı merdiven: o güne kadarki en iyi', 'growth': 'Son 12 ayda\n{a:.0f} → {b:.0f}  (+{d:.0f} puan)', 'cloud': 'diğer ölçülen modeller', 'credit': 'Kaynak: artificialanalysis.ai (Image Arena)  ·  Derleyen: Prof. Dr. Oğuz Ergin'}

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
plt.savefig("image_elo_tr.png", dpi=105, facecolor="#0d1117", bbox_inches="tight")
print("kaydedildi: image_elo_tr.png", len(df), " sinir:", len(fr))

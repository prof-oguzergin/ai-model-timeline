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
542.2,
"DALLE 2",
"OpenAI"
],
[
"2022-10-01",
455.0,
"Stable Diffusion 1.5",
"Stability"
],
[
"2022-12-07",
548.0,
"Stable Diffusion 2.1",
"Stability"
],
[
"2023-07-23",
677.2,
"Stable Diffusion XL 1.0",
"Stability"
],
[
"2023-09-20",
776.3,
"DALLE 3",
"OpenAI"
],
[
"2023-09-26",
774.3,
"DALLE 3 HD",
"OpenAI"
],
[
"2023-11-10",
718.5,
"Stable Diffusion 1.6",
"Stability"
],
[
"2023-11-29",
707.1,
"Amazon Titan G1 (Standard)",
"Amazon"
],
[
"2023-12-20",
873.9,
"Midjourney v6",
"Midjourney"
],
[
"2024-02-21",
719.0,
"SDXL Lightning",
"ByteDance"
],
[
"2024-02-22",
837.6,
"Stable Diffusion 3 Large",
"Stability"
],
[
"2024-02-24",
720.8,
"Stable Diffusion 3 Large Turbo",
"Stability"
],
[
"2024-02-27",
776.1,
"Playground v2.5",
"Playground AI"
],
[
"2024-02-28",
843.7,
"Ideogram v1",
"Ideogram"
],
[
"2024-03-13",
773.0,
"Recraft 20B",
"Recraft"
],
[
"2024-06-12",
719.5,
"Stable Diffusion 3 Medium",
"Stability"
],
[
"2024-06-13",
833.0,
"Phoenix 0.9 Ultra",
"Leonardo.Ai"
],
[
"2024-07-30",
847.5,
"Midjourney v6.1",
"Midjourney"
],
[
"2024-08-01",
841.6,
"FLUX.1 [dev]",
"FLUX"
],
[
"2024-08-01",
890.0,
"FLUX.1 [pro]",
"FLUX"
],
[
"2024-08-02",
804.2,
"FLUX.1 [schnell]",
"FLUX"
],
[
"2024-08-06",
730.8,
"Amazon Titan G1 v2 (Standard)",
"Amazon"
],
[
"2024-08-21",
869.8,
"Ideogram v2 Turbo",
"Ideogram"
],
[
"2024-08-21",
882.2,
"Ideogram v2",
"Ideogram"
],
[
"2024-09-16",
809.1,
"Playground v3 (beta)",
"Playground AI"
],
[
"2024-10-02",
889.5,
"FLUX1.1 [pro]",
"FLUX"
],
[
"2024-10-22",
828.0,
"Stable Diffusion 3.5 Large Turbo",
"Stability"
],
[
"2024-10-22",
839.0,
"Stable Diffusion 3.5 Large",
"Stability"
],
[
"2024-10-29",
758.7,
"Stable Diffusion 3.5 Medium",
"Stability"
],
[
"2024-10-30",
871.3,
"Recraft V3",
"Recraft"
],
[
"2024-11-06",
892.6,
"FLUX1.1 [pro] Ultra",
"FLUX"
],
[
"2024-11-25",
791.0,
"Runway Gen-4 Image",
"Runway"
],
[
"2024-12-02",
797.4,
"Luma Photon Flash",
"Luma"
],
[
"2024-12-02",
885.5,
"Luma Photon",
"Luma"
],
[
"2024-12-16",
915.5,
"Imagen 3",
"Google"
],
[
"2024-12-18",
822.1,
"Phoenix 1.0 Fast",
"Leonardo.Ai"
],
[
"2024-12-18",
842.1,
"Phoenix 1.0 Ultra",
"Leonardo.Ai"
],
[
"2025-01-25",
780.9,
"Lumina Image v2",
"OpenGVLab"
],
[
"2025-01-27",
532.4,
"Janus Pro",
"DeepSeek"
],
[
"2025-02-18",
873.8,
"Infinity 8B",
"ByteDance"
],
[
"2025-02-27",
819.1,
"Ideogram v2a",
"Ideogram"
],
[
"2025-02-27",
831.7,
"Ideogram v2a Turbo",
"Ideogram"
],
[
"2025-02-28",
860.5,
"Image-01",
"MiniMax"
],
[
"2025-03-22",
753.4,
"Sana Sprint 1.6B",
"NVIDIA"
],
[
"2025-03-24",
912.0,
"Reve Image (Halfmoon)",
"Reve"
],
[
"2025-03-26",
898.3,
"Ideogram 3.0",
"Ideogram"
],
[
"2025-04-03",
877.8,
"Midjourney v7 Alpha",
"Midjourney"
],
[
"2025-04-07",
869.0,
"HiDream-I1-Fast",
"HiDream"
],
[
"2025-04-07",
873.0,
"HiDream-I1-Dev",
"HiDream"
],
[
"2025-04-15",
959.0,
"Seedream 3.0",
"ByteDance"
],
[
"2025-04-23",
1006.9,
"GPT Image 1",
"OpenAI"
],
[
"2025-05-20",
704.5,
"Bagel",
"ByteDance"
],
[
"2025-05-20",
906.2,
"FLUX.1 Kontext [pro]",
"FLUX"
],
[
"2025-05-29",
937.5,
"FLUX.1 Kontext [max]",
"FLUX"
],
[
"2025-06-10",
922.8,
"Vivago 2.0",
"HiDream"
],
[
"2025-06-16",
727.1,
"OmniGen V2",
"VectorSpaceLab"
],
[
"2025-06-17",
819.0,
"Krea 1",
"Krea"
],
[
"2025-06-26",
882.7,
"Imagen 4 Fast",
"Google"
],
[
"2025-06-26",
909.6,
"Imagen 4 Standard",
"Google"
],
[
"2025-06-26",
997.2,
"Imagen 4 Ultra",
"Google"
],
[
"2025-07-22",
704.5,
"Bria 3.2",
"Bria"
],
[
"2025-07-28",
943.8,
"Kolors 2.1",
"Kling"
],
[
"2025-07-31",
843.9,
"FLUX.1 Krea [dev]",
"FLUX"
],
[
"2025-08-01",
912.0,
"Dreamina 3.1",
"ByteDance"
],
[
"2025-08-04",
885.9,
"Qwen Image",
"Alibaba"
],
[
"2025-08-05",
900.7,
"Lucid Origin Fast",
"Leonardo.Ai"
],
[
"2025-08-05",
911.7,
"Lucid Origin Ultra",
"Leonardo.Ai"
],
[
"2025-08-26",
984.6,
"Nano Banana (Gemini 2.5 Flash Image)",
"Google"
],
[
"2025-09-08",
886.5,
"HunyuanImage 2.1",
"Tencent"
],
[
"2025-09-08",
1025.8,
"Seedream 4.0",
"ByteDance"
],
[
"2025-09-11",
865.8,
"SRPO",
"Tencent"
],
[
"2025-09-23",
960.9,
"Wan 2.5 Preview",
"Alibaba"
],
[
"2025-09-28",
945.2,
"HunyuanImage 3.0 (Fal)",
"Tencent"
],
[
"2025-10-06",
914.0,
"GPT Image 1 Mini",
"OpenAI"
],
[
"2025-10-20",
944.2,
"Vivago 2.1",
"HiDream"
],
[
"2025-10-30",
879.6,
"FIBO",
"Bria"
],
[
"2025-11-04",
863.8,
"MAI Image 1",
"Microsoft AI"
],
[
"2025-11-20",
956.7,
"ImagineArt 1.5 Preview",
"ImagineArt"
],
[
"2025-11-20",
1100.0,
"Nano Banana Pro (Gemini 3 Pro Image)",
"Google"
],
[
"2025-11-25",
1000.0,
"FLUX.2 [dev]",
"FLUX"
],
[
"2025-11-25",
1001.4,
"FLUX.2 [pro]",
"FLUX"
],
[
"2025-11-25",
1025.8,
"FLUX.2 [flex]",
"FLUX"
],
[
"2025-11-27",
828.7,
"FIBO Lite",
"Bria"
],
[
"2025-11-28",
920.8,
"Vidu Q2",
"Vidu"
],
[
"2025-12-02",
882.8,
"P-Image",
"Pruna AI"
],
[
"2025-12-02",
941.8,
"Z-Image Turbo",
"Alibaba"
],
[
"2025-12-05",
860.6,
"LongCat Image",
"Meituan"
],
[
"2025-12-05",
1020.5,
"Seedream 4.5",
"ByteDance"
],
[
"2025-12-16",
1021.5,
"FLUX.2 [max]",
"FLUX"
],
[
"2025-12-16",
1102.5,
"GPT Image 1.5",
"OpenAI"
],
[
"2025-12-17",
1023.4,
"Wan 2.6 Image",
"Alibaba"
],
[
"2025-12-21",
983.5,
"FLUX.2 [dev] Flash",
"Fal"
],
[
"2025-12-21",
997.8,
"FLUX.2 [dev] Turbo",
"Fal"
],
[
"2025-12-30",
998.3,
"Qwen Image Max 2512",
"Alibaba"
],
[
"2026-01-13",
889.0,
"GLM-Image",
"Z.ai"
],
[
"2026-01-15",
793.5,
"FLUX.2 [klein] Base 4B",
"FLUX"
],
[
"2026-01-15",
863.8,
"FLUX.2 [klein] 4B",
"FLUX"
],
[
"2026-01-15",
901.8,
"FLUX.2 [klein] Base 9B",
"FLUX"
],
[
"2026-01-15",
939.8,
"FLUX.2 [klein] 9B",
"FLUX"
],
[
"2026-01-16",
936.5,
"Qwen Image Plus 2601",
"Alibaba"
],
[
"2026-01-19",
1013.0,
"Wan2.6 Text to Image",
"Alibaba"
],
[
"2026-01-25",
964.1,
"HunyuanImage 3.0 Instruct (Fal)",
"Tencent"
],
[
"2026-01-27",
874.1,
"Z-Image Base",
"Alibaba"
],
[
"2026-01-28",
923.7,
"Eigen Image",
"Eigen AI"
],
[
"2026-01-28",
1018.7,
"grok-imagine-image",
"xAI"
],
[
"2026-02-04",
915.1,
"Kling Image 3.0 Omni",
"Kling"
],
[
"2026-02-13",
1009.3,
"Seedream 5.0 Lite",
"ByteDance"
],
[
"2026-02-17",
983.6,
"Recraft V4",
"Recraft"
],
[
"2026-02-17",
983.9,
"Recraft V4 Pro",
"Recraft"
],
[
"2026-02-26",
1121.7,
"Nano Banana 2 (Gemini 3.1 Flash Image Preview)",
"Google"
],
[
"2026-03-03",
959.5,
"Qwen Image 2.0 (2026-03-03)",
"Alibaba"
],
[
"2026-03-19",
1008.6,
"MAI-Image-2",
"Microsoft AI"
],
[
"2026-04-03",
983.2,
"Wan 2.7 Pro",
"Alibaba"
],
[
"2026-04-03",
987.9,
"Wan 2.7",
"Alibaba"
],
[
"2026-04-03",
1043.1,
"grok-imagine-image-quality",
"xAI"
],
[
"2026-04-08",
940.5,
"image-1",
"Api Airforce"
],
[
"2026-04-14",
989.2,
"MAI-Image-2-Efficient",
"Microsoft AI"
],
[
"2026-04-15",
912.8,
"ERNIE Image",
"Baidu"
],
[
"2026-04-15",
923.7,
"ERNIE Image Turbo",
"Baidu"
],
[
"2026-04-16",
971.5,
"ImagineArt 2.0",
"ImagineArt"
],
[
"2026-04-21",
1170.9,
"GPT Image 2",
"OpenAI"
],
[
"2026-04-22",
1029.8,
"Qwen Image 2.0 Pro (2026-04-22)",
"Alibaba"
],
[
"2026-05-05",
989.5,
"Luma UNI 1",
"Luma"
],
[
"2026-05-05",
1013.0,
"Luma UNI 1 Max",
"Luma"
],
[
"2026-05-08",
880.6,
"HiDream-O1-Image-Dev",
"HiDream"
],
[
"2026-05-08",
979.2,
"HiDream-O1-Image",
"HiDream"
],
[
"2026-05-12",
797.0,
"Step Image Edit 2",
"StepFun"
],
[
"2026-05-14",
980.6,
"Recraft V4.1 Pro",
"Recraft"
],
[
"2026-05-14",
988.9,
"Recraft V4.1",
"Recraft"
],
[
"2026-05-14",
1016.6,
"Recraft V4.1 Utility Pro",
"Recraft"
],
[
"2026-05-14",
1020.1,
"Recraft V4.1 Utility",
"Recraft"
],
[
"2026-05-26",
1010.9,
"Krea 2 Medium",
"Krea"
],
[
"2026-05-26",
1023.5,
"Krea 2 Large",
"Krea"
],
[
"2026-05-31",
983.4,
"Cosmos3-Super-Text2Image",
"NVIDIA"
],
[
"2026-05-31",
992.2,
"Cosmos3-Super-Text2Image (agentic)",
"NVIDIA"
],
[
"2026-06-02",
1032.6,
"MAI-Image-2.5-Flash",
"Microsoft AI"
],
[
"2026-06-02",
1103.4,
"MAI-Image-2.5",
"Microsoft AI"
],
[
"2026-06-03",
1011.2,
"Ideogram 4.0",
"Ideogram"
],
[
"2026-06-03",
1016.5,
"Krea 2 Medium Turbo",
"Krea"
],
[
"2026-06-04",
1022.2,
"HiDream-O1-Image-1.5",
"HiDream"
],
[
"2026-06-30",
1091.9,
"Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
"Google"
],
[
"2026-07-07",
1111.9,
"Muse Image",
"Meta"
],
[
"2026-07-08",
1077.8,
"Seedream 5.0 Pro",
"ByteDance"
],
[
"2026-07-13",
980.1,
"Ideogram 4.0 Instant",
"Fal"
],
[
"2026-07-13",
984.5,
"Ideogram 4.0 Fast (Quality)",
"Fal"
],
[
"2026-07-20",
970.2,
"Cosmos3-Super-Text2Image-4Step",
"NVIDIA"
],
[
"2026-07-21",
1076.9,
"Qwen-Image-3.0",
"Alibaba"
],
[
"2026-07-21",
1087.5,
"Qwen-Image-3.0-Pro",
"Alibaba"
],
[
"2026-07-23",
1099.0,
"MAI-Image-2.5-Pro",
"Microsoft AI"
],
[
"2026-07-30",
915.6,
"P-Image-Ideogram (Very Low)",
"Pruna AI"
],
[
"2026-07-30",
964.2,
"P-Image-Ideogram (Low)",
"Pruna AI"
],
[
"2026-07-30",
973.9,
"P-Image-Ideogram (Medium)",
"Pruna AI"
],
[
"2026-07-30",
996.6,
"P-Image-Ideogram (High)",
"Pruna AI"
],
[
"2026-08-07",
1154.2,
"Grok Imagine Image 2.0",
"xAI"
],
[
"2026-08-10",
1147.4,
"MAI-Image-2.6-Preview",
"Microsoft AI"
],
[
"2026-09-04",
1100.0,
"MAI-Image-2.6-Flash",
"Microsoft AI"
],
[
"2026-09-08",
1190.4,
"GPT Image 2.5 Flare",
"OpenAI"
],
[
"2026-09-08",
1196.8,
"GPT Image 2.5 Sunburst",
"OpenAI"
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

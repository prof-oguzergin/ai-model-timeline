# -*- coding: utf-8 -*-
# Gorsel uretme modellerinin Elo puani - YENI OLCEK (AA-Image-T2I v2.1)
# Veri: artificialanalysis.ai/image/leaderboard/text-to-image, 10 Kas 2026 goruntusu
# Eski olcekteki cizelge image_elo_tr.py icinde DONDURULDU; iki olcek karsilastirilamaz.
# Uretildi: _kod/elo_v21_uret.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# [tarih, endeks, model, sirket]
DATA = [
["2024-06-13", 832.0, "Phoenix 0.9 Ultra", "Leonardo.Ai"],
["2024-08-01", 816.0, "FLUX.1 [dev]", "FLUX"],
["2024-08-02", 742.0, "FLUX.1 [schnell]", "FLUX"],
["2024-10-02", 871.0, "FLUX1.1 [pro]", "FLUX"],
["2024-11-06", 854.0, "FLUX1.1 [pro] Ultra", "FLUX"],
["2024-12-02", 774.0, "Luma Photon Flash", "Luma"],
["2024-12-02", 844.0, "Luma Photon", "Luma"],
["2024-12-18", 829.0, "Phoenix 1.0 Fast", "Leonardo.Ai"],
["2025-03-26", 911.0, "Ideogram 3.0", "Ideogram"],
["2025-04-01", 1015.0, "GPT Image 1 (high)", "OpenAI"],
["2025-05-20", 601.0, "Bagel", "ByteDance"],
["2025-05-20", 924.0, "FLUX.1 Kontext [pro]", "FLUX"],
["2025-05-29", 931.0, "FLUX.1 Kontext [max]", "FLUX"],
["2025-06-16", 710.0, "OmniGen V2", "VectorSpaceLab"],
["2025-07-31", 865.0, "FLUX.1 Krea [dev]", "FLUX"],
["2025-08-01", 932.0, "Dreamina 3.1", "ByteDance"],
["2025-08-04", 903.0, "Qwen Image", "Alibaba"],
["2025-08-05", 864.0, "Lucid Origin Fast", "Leonardo.Ai"],
["2025-08-05", 878.0, "Lucid Origin Ultra", "Leonardo.Ai"],
["2025-08-26", 986.0, "Nano Banana (Gemini 2.5 Flash Image)", "Google"],
["2025-09-01", 932.0, "HunyuanImage 3.0", "Tencent"],
["2025-09-01", 994.0, "HunyuanImage 3.0 (Tencent Cloud)", "Tencent"],
["2025-09-11", 831.0, "SRPO", "Tencent"],
["2025-10-01", 937.0, "GPT Image 1 Mini (medium)", "OpenAI"],
["2025-10-30", 855.0, "FIBO", "Bria"],
["2025-11-20", 1119.0, "Nano Banana Pro (Gemini 3 Pro Image)", "Google"],
["2025-11-25", 1000.0, "FLUX.2 [dev]", "FLUX"],
["2025-11-25", 1003.0, "FLUX.2 [pro]", "FLUX"],
["2025-11-25", 1028.0, "FLUX.2 [flex]", "FLUX"],
["2025-11-27", 814.0, "FIBO Lite", "Bria"],
["2025-11-28", 895.0, "Vidu Q2", "Vidu"],
["2025-12-01", 1124.0, "GPT Image 1.5 (high)", "OpenAI"],
["2025-12-02", 896.0, "P-Image", "Pruna AI"],
["2025-12-02", 936.0, "Z-Image Turbo", "Alibaba"],
["2025-12-05", 861.0, "LongCat Image", "Meituan"],
["2025-12-05", 1029.0, "Seedream 4.5", "ByteDance"],
["2025-12-16", 1021.0, "FLUX.2 [max]", "FLUX"],
["2025-12-17", 1027.0, "Wan 2.6 Image", "Alibaba"],
["2025-12-21", 989.0, "FLUX.2 [dev] Flash", "Fal"],
["2025-12-21", 995.0, "FLUX.2 [dev] Turbo", "Fal"],
["2026-01-01", 988.0, "HunyuanImage 3.0 Instruct", "Tencent"],
["2026-01-13", 892.0, "GLM-Image", "Z.ai"],
["2026-01-15", 840.0, "FLUX.2 [klein] 4B", "FLUX"],
["2026-01-15", 932.0, "FLUX.2 [klein] 9B", "FLUX"],
["2026-01-16", 960.0, "Qwen Image Plus 2601", "Alibaba"],
["2026-01-19", 1019.0, "Wan2.6 Text to Image", "Alibaba"],
["2026-01-27", 895.0, "Z-Image Base", "Alibaba"],
["2026-01-28", 1024.0, "grok-imagine-image", "xAI"],
["2026-02-01", 1142.0, "Nano Banana 2 (Gemini 3.1 Flash Image)", "Google"],
["2026-02-04", 921.0, "Kling Image 3.0 Omni", "Kling"],
["2026-02-13", 1020.0, "Seedream 5.0 Lite", "ByteDance"],
["2026-03-03", 992.0, "Qwen Image 2.0 (2026-03-03)", "Alibaba"],
["2026-04-01", 1189.0, "GPT Image 2 (high)", "OpenAI"],
["2026-04-03", 981.0, "Wan 2.7 Pro", "Alibaba"],
["2026-04-03", 988.0, "Wan 2.7", "Alibaba"],
["2026-04-03", 1049.0, "grok-imagine-image-quality", "xAI"],
["2026-04-15", 909.0, "ERNIE Image", "Baidu"],
["2026-04-15", 915.0, "ERNIE Image Turbo", "Baidu"],
["2026-04-22", 1031.0, "Qwen Image 2.0 Pro (2026-04-22)", "Alibaba"],
["2026-05-05", 994.0, "Luma UNI 1", "Luma"],
["2026-05-05", 1026.0, "Luma UNI 1 Max", "Luma"],
["2026-05-08", 813.0, "HiDream-O1-Image-Dev", "HiDream"],
["2026-05-08", 979.0, "HiDream-O1-Image", "HiDream"],
["2026-05-14", 1019.0, "Recraft V4.1 Utility Pro", "Recraft"],
["2026-05-14", 1028.0, "Recraft V4.1 Utility", "Recraft"],
["2026-05-26", 1013.0, "Krea 2 Medium", "Krea"],
["2026-05-26", 1035.0, "Krea 2 Large", "Krea"],
["2026-05-31", 988.0, "Cosmos3-Super-Text2Image", "NVIDIA"],
["2026-05-31", 995.0, "Cosmos3-Super-Text2Image (agentic)", "NVIDIA"],
["2026-06-01", 1028.0, "Ideogram 4.0 (Quality)", "Ideogram"],
["2026-06-02", 1033.0, "MAI-Image-2.5-Flash", "Microsoft"],
["2026-06-02", 1119.0, "MAI-Image-2.5", "Microsoft"],
["2026-06-03", 1003.0, "Ideogram 4.0", "Ideogram"],
["2026-06-03", 1021.0, "Krea 2 Medium Turbo", "Krea"],
["2026-06-04", 1018.0, "HiDream-O1-Image-1.5", "HiDream"],
["2026-06-30", 1110.0, "Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)", "Google"],
["2026-07-01", 965.0, "Ideogram 4.0 Fast", "Fal"],
["2026-07-01", 993.0, "P-Image-Ideogram (Very High)", "Pruna AI"],
["2026-07-07", 1128.0, "Muse Image", "Meta"],
["2026-07-08", 1101.0, "Seedream 5.0 Pro", "ByteDance"],
["2026-07-13", 984.0, "Ideogram 4.0 Fast (Quality)", "Fal"],
["2026-07-13", 986.0, "Ideogram 4.0 Instant", "Fal"],
["2026-07-20", 972.0, "Cosmos3-Super-Text2Image-4Step", "NVIDIA"],
["2026-07-21", 1074.0, "Qwen-Image-3.0", "Alibaba"],
["2026-07-21", 1094.0, "Qwen-Image-3.0-Pro", "Alibaba"],
["2026-07-23", 1116.0, "MAI-Image-2.5-Pro", "Microsoft"],
["2026-07-30", 912.0, "P-Image-Ideogram (Very Low)", "Pruna AI"],
["2026-07-30", 964.0, "P-Image-Ideogram (Low)", "Pruna AI"],
["2026-07-30", 976.0, "P-Image-Ideogram (Medium)", "Pruna AI"],
["2026-07-30", 998.0, "P-Image-Ideogram (High)", "Pruna AI"],
["2026-08-01", 1168.0, "MAI-Image-2.6", "Microsoft"],
["2026-08-07", 1167.0, "Grok Imagine Image 2.0", "xAI"],
["2026-09-01", 1000.0, "Ming-Image-0.1-Design", "InclusionAI"],
["2026-09-01", 1209.0, "GPT Image 2.5 Flare (max)", "OpenAI"],
["2026-09-01", 1228.0, "GPT Image 2.5 Sunburst (max)", "OpenAI"],
["2026-09-04", 1123.0, "MAI-Image-2.6-Flash", "Microsoft"],
["2026-09-20", 1046.0, "Qwen-Image-2.1", "Alibaba"],
["2026-09-22", 974.0, "HunyuanImage 3.5 (Preview)", "Tencent"],
["2026-09-23", 951.0, "Recraft V4.1 Flash", "Recraft"],
["2026-09-30", 1014.0, "Ideogram 4.5 (High)", "Ideogram"],
["2026-10-01", 1026.0, "Qwen-Image-2.1-Turbo", "Alibaba"],
["2026-10-01", 1042.0, "FLUX 3 Image Klein", "FLUX"],
["2026-10-01", 1058.0, "Qwen-Image-2.1-Pro", "Alibaba"],
["2026-10-01", 1116.0, "FLUX 3 Image", "FLUX"],
["2026-10-06", 1176.0, "Nano Banana 2.1", "Google"],
["2026-10-20", 1040.0, "Fibo 2.0", "Bria"],
["2026-11-04", 1227.0, "GPT Image 2.6 Flare (max)", "OpenAI"],
["2026-11-04", 1259.0, "GPT Image 2.6 Sunburst (max)", "OpenAI"]
]
L = {'ylabel': 'Görsel Kalite Elo (AA-Image-T2I v2.1)', 'title': 'Görsel Üretme Modellerinin Başarımı — Yeni Ölçek', 'sub': '{n} modelin körlemesine insan oylamasıyla Elo puanı ({lo} – {hi})  ·  sarı merdiven: o güne kadarki en iyi  ·  ölçek: AA-Image-T2I v2.1, 10 Kasım 2026 görüntüsü', 'growth': '{a:.0f} → {b:.0f}\n(+{d:.0f} puan)', 'cloud': 'diğer ölçülen modeller', 'credit': 'Kaynak: artificialanalysis.ai (Image Arena, v2.1)  ·  Derleyen: Prof. Dr. Oğuz Ergin'}

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
plt.savefig("image_elo_v21_tr.png", dpi=105, facecolor="#0d1117", bbox_inches="tight")
print("kaydedildi: image_elo_v21_tr.png", len(df), " sinir:", len(fr))

import json, re, sys, os
sys.stdout.reconfigure(encoding='utf-8')
OUT = r'G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi'
rows = json.load(open('ii_data_open.json', encoding='utf-8'))

CRE = {'SpaceXAI': 'xAI', 'Z AI': 'Z.ai', 'ByteDance Seed': 'ByteDance', 'Moonshot AI': 'Moonshot',
       'Meta Superintelligence Labs': 'Meta', 'Mistral AI': 'Mistral'}
def clean_name(n):
    n = re.sub(r'\s*\((Adaptive Reasoning[^)]*|Reasoning|Non-reasoning|high|xhigh|max|medium|low|Preview|Thinking)\)', '', n)
    n = re.sub(r'\s*\(.*?\)$', '', n)
    return n.strip()

data = [[d, ii, clean_name(n), CRE.get(c, c), o] for d, ii, n, c, o in rows]
# ayni gun + ayni temiz ad -> en yuksek varyant
best = {}
for d, ii, n, c, o in data:
    k = (d, n)
    if k not in best or ii > best[k][1]: best[k] = [d, ii, n, c, o]
data = sorted(best.values())
print('temizlenmis kayit:', len(data))

TEMPLATE = '''# -*- coding: utf-8 -*-
# Yapay zeka basarimi tek sayiyla: Artificial Analysis Zeka Endeksi (Intelligence Index)
# Veri kaynagi: artificialanalysis.ai  ({N} model, {LO} - {HI})
# Bu dosya scratchpad/make_ii_scripts.py ile uretildi; veri asagida GOMULU.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

# [tarih, endeks, model, sirket]
DATA = {DATA}

L = {LANG}

COLORS = {"Anthropic": "#d97757", "OpenAI": "#10a37f", "Google": "#4285F4", "xAI": "#1da1f2",
          "Meta": "#0668E1", "DeepSeek": "#ef4444", "Alibaba": "#7C3AED", "Moonshot": "#14B8A6",
          "Z.ai": "#BE185D", "MiniMax": "#C77DFF", "Mistral": "#fa8005", "ByteDance": "#22D3EE",
          "Microsoft": "#F25022", "Amazon": "#ff9900", "NVIDIA": "#76b900"}
OTHER = "#4a5160"

df = pd.DataFrame(DATA, columns=["date", "ii", "name", "comp", "open"])
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

# --- acik agirlik siniri ---
fo, best_o = [], -1
for _, r in df[df["open"] == 1].iterrows():
    if r["ii"] > best_o:
        best_o = r["ii"]
        if fo and fo[-1]["Date"] == r["Date"]: fo[-1] = r
        else: fo.append(r)
fro = pd.DataFrame(fo).reset_index(drop=True)
OPEN_C = "#3fb950"

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(26, 14))
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")

# arka plan bulutu
ax.scatter(df["Date"], df["ii"], s=34, c="#262c36", alpha=.85, edgecolors="none", zorder=1)

# sinir merdiveni
ax.step(fr["Date"], fr["ii"], where="post", color="#ffd166", lw=3.0, zorder=3, alpha=.95)
ax.fill_between(fr["Date"], fr["ii"], step="post", color="#ffd166", alpha=.05, zorder=2)
ax.step(fro["Date"], fro["ii"], where="post", color=OPEN_C, lw=2.6, zorder=3, alpha=.95, linestyle=(0, (6, 2)))
for _, r in fr.iterrows():
    ax.scatter([r["Date"]], [r["ii"]], s=230, c=COLORS.get(r["comp"], OTHER),
               edgecolors="white", linewidths=2.0, zorder=5)
for _, r in fro.iterrows():
    ax.scatter([r["Date"]], [r["ii"]], s=150, c=COLORS.get(r["comp"], OTHER),
               edgecolors=OPEN_C, linewidths=2.4, zorder=4)

# sinir etiketleri: PIKSEL uzayinda 2 boyutlu cakisma kontrolu
# (kademe farki tek basina yetmiyor: noktalarin kendi yuksekligi de degisiyor)
PX_DAY = (26 * 105 * 0.93) / max(1, (df["Date"].max() - df["Date"].min()).days)
YLIM = fr["ii"].max() * 1.20
PX_UNIT = (14 * 105 * 0.78) / YLIM
TIERS = [26, -34, 66, -74, 106, -114, 146, -154, 186, -194, 226, -234]
x0 = df["Date"].min().toordinal()
boxes = []
_seen = set()
_items = [(r, False) for _, r in fr.iterrows()] + [(r, True) for _, r in fro.iterrows()]
_items = [(r, o) for r, o in _items if not ((r["Date"], r["name"]) in _seen or _seen.add((r["Date"], r["name"])))]
_items.sort(key=lambda x: x[0]["Date"])
for r, _is_open in _items:
    cx = (r["Date"].toordinal() - x0) * PX_DAY
    hw = len(r["name"]) * 4.5 + 17
    tier, en_iyi = None, (-1, TIERS[-1])
    for t in TIERS:
        cy = r["ii"] * PX_UNIT + t
        if all(abs(cx - bx) > (hw + bw) or abs(cy - by) > 58 for bx, by, bw in boxes):
            tier = t; break
        # yer yoksa: en az cakisan kademeyi akilda tut
        pay = min(((abs(cx - bx) - (hw + bw)) if abs(cy - by) <= 58 else 9999) for bx, by, bw in boxes)
        if pay > en_iyi[0]: en_iyi = (pay, t)
    if tier is None: tier = en_iyi[1]
    boxes.append((cx, r["ii"] * PX_UNIT + tier, hw))
    ax.annotate(r["name"], (r["Date"], r["ii"]), xytext=(0, tier), textcoords="offset points",
                fontsize=12, color="#e6edf3", fontweight="bold", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.30", fc="#161b22",
                          ec=(OPEN_C if _is_open else COLORS.get(r["comp"], OTHER)), lw=1.6, alpha=.96),
                arrowprops=dict(arrowstyle="-", color=(OPEN_C if _is_open else COLORS.get(r["comp"], OTHER)),
                                lw=1.1, alpha=.55, shrinkA=2, shrinkB=6))

ax.set_ylabel(L["ylabel"], fontsize=17, color="#8b949e", labelpad=16)
ax.set_ylim(0, YLIM)
ax.grid(True, axis="y", color="#21262d", lw=1.0)
ax.grid(True, axis="x", color="#161b22", lw=.7)
for s in ax.spines.values(): s.set_color("#30363d")
ax.tick_params(colors="#8b949e", labelsize=14)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
# Ay adlari dile bagli. Onceden her iki cizelgede de %b kullaniliyordu,
# Turkce cizelgenin ekseni ve alt basligi "Nov 2022 - Sep 2026" diyordu.
if L.get("months"):
    _AY = L["months"]
    def _ay(ts): return "%s %d" % (_AY[ts.month - 1], ts.year)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, pos: _ay(mdates.num2date(x))))
else:
    def _ay(ts): return ts.strftime("%b %Y")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

# baslik + alt baslik
son = fr.iloc[-1]
bir_yil = fr[fr["Date"] <= son["Date"] - pd.Timedelta(days=365)]["ii"].max()
plt.title(L["title"], fontsize=30, color="white", pad=54, fontweight="bold")
ax.text(0.5, 1.045, L["sub"].format(n=len(df), lo=_ay(df["Date"].min()),
                                    hi=_ay(df["Date"].max())),
        transform=ax.transAxes, ha="center", fontsize=15, color="#8b949e", style="italic")

# buyume kutusu
son_o = fro.iloc[-1]
ax.text(0.015, 0.965, L["growth"].format(a=bir_yil, b=son["ii"], k=son["ii"] / bir_yil,
                                         ao=son_o["ii"], fark=son["ii"] - son_o["ii"]),
        transform=ax.transAxes, ha="left", va="top", fontsize=17, color="#ffd166", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", fc="#161b22", ec="#ffd166", lw=1.8, alpha=.95))

# lejant (sinirdaki sirketler)
import matplotlib.lines as mlines
comps = list(dict.fromkeys(fr["comp"]))
handles = [mlines.Line2D([], [], marker="o", linestyle="", markersize=13, markerfacecolor=COLORS.get(c, OTHER),
                         markeredgecolor="white", label=c) for c in comps]
handles.append(mlines.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="#262c36",
                             markeredgecolor="none", label=L["cloud"]))
handles.insert(0, mlines.Line2D([], [], color="#ffd166", lw=3, label=L["front_all"]))
handles.insert(1, mlines.Line2D([], [], color=OPEN_C, lw=2.6, linestyle=(0, (6, 2)), label=L["front_open"]))
ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="#161b22", edgecolor="#30363d",
          fontsize=14, labelcolor="#c9d1d9", ncol=2)

ax.text(0.995, -0.115, L["credit"], transform=ax.transAxes, ha="right", fontsize=13,
        color="#6e7681", style="italic")
plt.tight_layout()
plt.savefig("{PNG}", dpi=105, facecolor="#0d1117", bbox_inches="tight")
print("kaydedildi: {PNG}  |  model:", len(df), " sinir:", len(fr))
'''

LANG_TR = dict(
    months=['Oca','Şub','Mar','Nis','May','Haz','Tem','Ağu','Eyl','Eki','Kas','Ara'],
    ylabel='Zekâ Endeksi (Artificial Analysis)',
    title='Tek Sayıyla Yapay Zeka Modellerinin Başarımı',
    sub='{n} modelin bağımsız ölçümü ({lo} – {hi})  ·  sarı: o güne kadarki en iyi  ·  yeşil kesikli: en iyi açık ağırlıklı',
    growth='Son 12 ayda\n{a:.0f} → {b:.0f}  ({k:.1f}×)\nAçık ağırlık en iyi: {ao:.0f}  (makas {fark:.1f})',
    cloud='diğer ölçülen modeller',
    front_all='o güne kadarki en iyi',
    front_open='en iyi açık ağırlıklı',
    credit='Kaynak: artificialanalysis.ai  ·  Derleyen: Prof. Dr. Oğuz Ergin')
LANG_EN = dict(
    months=None,
    ylabel='Intelligence Index (Artificial Analysis)',
    title='AI Model Capability Over Time — A Single Number',
    sub='independent measurement of {n} models ({lo} – {hi})  ·  yellow: best available  ·  green dashed: best open-weights',
    growth='Last 12 months\n{a:.0f} → {b:.0f}  ({k:.1f}x)\nBest open-weights: {ao:.0f}  (gap {fark:.1f})',
    cloud='other measured models',
    front_all='best at the time',
    front_open='best open-weights',
    credit='Source: artificialanalysis.ai  ·  Compiled by Prof. Dr. Oğuz Ergin')

def emit(fname, lang, png):
    body = TEMPLATE.replace('{DATA}', json.dumps(data, ensure_ascii=False, indent=0).replace('], [', '],\n ['))
    body = body.replace('{LANG}', repr(lang)).replace('{PNG}', png)
    body = body.replace('{N}', str(len(data))).replace('{LO}', data[0][0]).replace('{HI}', data[-1][0])
    open(os.path.join(OUT, fname), 'w', encoding='utf-8').write(body)
    print('yazildi:', fname, len(body), 'char')

emit('intelligence_index_tr.py', LANG_TR, 'intelligence_index_tr.png')
emit('intelligence_index.py', LANG_EN, 'intelligence_index.png')

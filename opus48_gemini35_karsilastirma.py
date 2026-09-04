import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# --- Sabitler ---
FIG_W, FIG_H = 8.5, 13
DPI = 150
BG_COLOR    = '#0f1419'
PANEL_COLOR = '#1a1f2e'
HL_OPUS_BG  = '#3a2a1e'    # Opus 4.8 sütun arka planı
HL_GEM_BG   = '#1e3a5f'    # Gemini 3.5 Flash sütun arka planı
HL_OPUS_HDR = '#ffd0b0'    # Opus 4.8 başlık rengi
HL_GEM_HDR  = '#7cc0ff'    # Gemini 3.5 Flash başlık rengi
BEST_COLOR  = '#ffd166'    # Altın — en iyi hücre yazı
BEST_BG     = '#2a2500'    # En iyi hücre hafif arka plan
CAT_BG      = '#0d2240'    # Kategori şeridi
WHITE       = '#ffffff'
LIGHT_GRAY  = '#aab4be'
MID_GRAY    = '#6b7280'
DARK_BORDER = '#2d3748'

# --- Sütun tanımları ---
# (görüntü adı, tarih, vurgu_türü)  vurgu_türü: 'opus' | 'gem' | None
COLS = [
    ('Claude\nOpus 4.8',     '28 May 26', 'opus'),
    ('Gemini\n3.5 Flash',    '19 May 26', 'gem'),
    ('Claude\nOpus 4.7',     '16 Nis 26', None),
    ('GPT-5.5',              '23 Nis 26', None),
    ('Gemini\n3.1 Pro',      '19 Şub 26', None),
]
N_COLS = len(COLS)

# --- Satır verileri ---
# Yapı: (tip, benzmark_adi, [5 değer], is_elo, is_dollar)
# Değer None → "—"; is_elo=True → Elo sayısı göster; is_dollar=True → $ formatla
ROWS = [
    # KATEGORİ BAŞLIKLARI: ('CAT', 'ETİKET')
    ('CAT', 'KODLAMA'),
    ('DATA', 'SWE-bench Verified',     [88.6, None,  87.6, None,  80.6], False, False),
    ('DATA', 'SWE-bench Pro',          [69.2, 55.1,  64.3, 58.6,  54.2], False, False),
    ('DATA', 'Terminal-Bench 2.1',     [74.6, 76.2,  66.1, 78.2,  70.3], False, False),

    ('CAT', 'AKIL YÜRÜTME'),
    ('DATA', 'GPQA Diamond',           [93.6, None,  94.2, 93.6,  94.3], False, False),
    ('DATA', "Humanity's Last Exam",   [49.8, 40.2,  46.9, 41.4,  44.4], False, False),
    ('DATA', 'ARC-AGI-2',              [None, 72.1,  75.8, 85.0,  77.1], False, False),

    ('CAT', 'ÇOK BİÇİMLİ'),
    ('DATA', 'MMMU-Pro',               [None, 83.6,  None, 81.2,  80.5], False, False),
    ('DATA', 'CharXiv Reasoning',      [None, 84.2,  82.1, None,  None], False, False),

    ('CAT', 'EYLEMCİ / ARAÇ KULLANIMI'),
    ('DATA', 'GDPval-AA (Elo)',        [1890, 1656,  1753, 1769,  1314], True,  False),
    ('DATA', 'MCP-Atlas',              [82.2, 83.6,  79.1, 75.3,  78.2], False, False),
    ('DATA', 'Toolathlon',             [None, 56.5,  None, 55.6,  48.8], False, False),
    ('DATA', 'Finance Agent v2',       [53.9, 57.9,  51.5, 51.8,  43.0], False, False),
    ('DATA', 'OSWorld-Verified',       [83.4, 78.4,  82.8, 78.7,  76.2], False, False),
    ('DATA', 'Automation Bench',       [15.5, 14.5,   9.9, 12.9,   9.6], False, False),

    ('CAT', 'UZUN BAĞLAM'),
    ('DATA', 'GraphWalks BFS 256K',    [85.9, None,  76.9, 73.7,  None], False, False),
    ('DATA', 'MRCR 1M',                [None, 26.6,  None, None,  76.3], False, False),

    ('CAT', 'GERÇEK DÜNYA'),
    # Vending-Bench: $ değeri, EN BÜYÜK sayı en iyi (Opus 4.7 = 10.937 en yüksek)
    ('DATA', 'Vending-Bench 2 ($) *',  [5.787, 5.396, 10.937, 7.524, 911.0], False, True),
]

def best_col(values, is_dollar):
    """En iyi sütunu döndür. is_dollar=True → en yüksek $ en iyi."""
    candidates = [(i, v) for i, v in enumerate(values) if v is not None]
    if not candidates:
        return -1
    return max(candidates, key=lambda x: x[1])[0]

def fmt_val(v, is_elo, is_dollar):
    if v is None:
        return '—'          # em-dash: —
    if is_elo:
        return f'{int(v):,}'.replace(',', '.')   # binlik nokta ayraç
    if is_dollar:
        # v < 100 → 3 ondalık (örn. $5.787); v >= 100 → tam sayı (örn. $911)
        if v >= 100:
            s = f'{int(v):,}'.replace(',', '.')
            return f'${s}'
        else:
            s = f'{v:,.3f}'.replace(',', '.')
            return f'${s}'
    return f'{v:.1f}%'

# --- Çizim ---
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor(BG_COLOR)

ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_facecolor(BG_COLOR)

# --- Alan tanımları ---
TOP_MARGIN   = 0.96
HEADER_H     = 0.065     # başlık bloğu yüksekliği
TABLE_TOP    = TOP_MARGIN - HEADER_H - 0.01
BOTTOM_MARGIN = 0.038
FOOTNOTE_H   = 0.038     # dipnot + alt bilgi
TABLE_BOTTOM = BOTTOM_MARGIN + FOOTNOTE_H
TABLE_H      = TABLE_TOP - TABLE_BOTTOM
LEFT         = 0.01
RIGHT        = 0.99
TABLE_W      = RIGHT - LEFT

# Sütun genişlikleri
BENCH_COL_W = 0.26
DATA_COL_W  = (TABLE_W - BENCH_COL_W) / N_COLS

def col_x(ci):
    return LEFT + BENCH_COL_W + ci * DATA_COL_W

# Satır yüksekliği hesabı
N_DATA = sum(1 for r in ROWS if r[0] == 'DATA')
N_CAT  = sum(1 for r in ROWS if r[0] == 'CAT')
HEADER_ROW_H = 0.055
CAT_ROW_H    = 0.023
DATA_ROW_H   = (TABLE_H - HEADER_ROW_H - N_CAT * CAT_ROW_H) / N_DATA

# --- Başlık ---
ax.text(0.5, TOP_MARGIN - 0.008,
        'Bu Haftanın İki Yeni Modeli',
        ha='center', va='top', fontsize=24, fontweight='bold',
        color=WHITE, fontfamily='DejaVu Sans')
ax.text(0.5, TOP_MARGIN - 0.038,
        'Claude Opus 4.8 & Gemini 3.5 Flash · Üst Düzey Rakiplerle',
        ha='center', va='top', fontsize=13.5, color=LIGHT_GRAY,
        fontfamily='DejaVu Sans')
ax.text(RIGHT - 0.005, TOP_MARGIN - 0.008,
        'Mayıs 2026',
        ha='right', va='top', fontsize=8, color=MID_GRAY,
        fontfamily='DejaVu Sans')

# --- Tablo arka planı ---
table_rect = FancyBboxPatch((LEFT, TABLE_BOTTOM), TABLE_W, TABLE_H,
                             boxstyle='round,pad=0.002',
                             linewidth=0, facecolor=PANEL_COLOR)
ax.add_patch(table_rect)

# --- Dikey vurgu şeritleri (tüm tablo boyunca) ---
# Opus 4.8 → ci=0
opus_x = col_x(0)
opus_rect = FancyBboxPatch((opus_x, TABLE_BOTTOM), DATA_COL_W, TABLE_H,
                            boxstyle='round,pad=0',
                            linewidth=0, facecolor=HL_OPUS_BG, zorder=1)
ax.add_patch(opus_rect)

# Gemini 3.5 Flash → ci=1
gem_x = col_x(1)
gem_rect = FancyBboxPatch((gem_x, TABLE_BOTTOM), DATA_COL_W, TABLE_H,
                           boxstyle='round,pad=0',
                           linewidth=0, facecolor=HL_GEM_BG, zorder=1)
ax.add_patch(gem_rect)

# --- Sütun başlıkları ---
cur_y = TABLE_TOP
hdr_bottom = cur_y - HEADER_ROW_H

# Benchmark sütunu başlık
ax.text(LEFT + 0.006, cur_y - HEADER_ROW_H / 2, 'Benchmark',
        ha='left', va='center', fontsize=8.5, color=LIGHT_GRAY,
        fontweight='bold', fontfamily='DejaVu Sans')

for ci, (name, date, hl) in enumerate(COLS):
    cx   = col_x(ci) + DATA_COL_W / 2
    cy_m = cur_y - HEADER_ROW_H / 2
    if hl == 'opus':
        name_color = HL_OPUS_HDR
        date_color = HL_OPUS_HDR
        fw = 'bold'
    elif hl == 'gem':
        name_color = HL_GEM_HDR
        date_color = HL_GEM_HDR
        fw = 'bold'
    else:
        name_color = LIGHT_GRAY
        date_color = MID_GRAY
        fw = 'normal'

    ax.text(cx, cy_m + 0.010, name,
            ha='center', va='center', fontsize=7.5,
            color=name_color, fontweight=fw,
            fontfamily='DejaVu Sans')
    ax.text(cx, cy_m - 0.012, date,
            ha='center', va='center', fontsize=6,
            color=date_color, fontfamily='DejaVu Sans')

# Başlık alt çizgisi
ax.plot([LEFT, RIGHT], [hdr_bottom, hdr_bottom],
        color=DARK_BORDER, lw=0.8, zorder=5)
cur_y = hdr_bottom

# --- Satırlar ---
for row in ROWS:
    rtype = row[0]

    if rtype == 'CAT':
        rh = CAT_ROW_H
        ry = cur_y - rh
        cat_rect = plt.Rectangle((LEFT, ry), TABLE_W, rh,
                                   facecolor=CAT_BG, linewidth=0, zorder=2)
        ax.add_patch(cat_rect)
        ax.text(LEFT + 0.008, ry + rh / 2, row[1],
                ha='left', va='center', fontsize=7.8,
                color=WHITE, fontweight='bold',
                fontfamily='DejaVu Sans')
        cur_y = ry

    else:  # DATA
        _, bench, values, is_elo, is_dollar = row
        rh = DATA_ROW_H
        ry = cur_y - rh
        best_ci = best_col(values, is_dollar)

        # Satır arası çizgi
        ax.plot([LEFT, RIGHT], [ry, ry],
                color=DARK_BORDER, lw=0.3, zorder=5)

        # Benchmark adı
        ax.text(LEFT + 0.006, ry + rh / 2, bench,
                ha='left', va='center', fontsize=7.2, color=WHITE,
                fontfamily='DejaVu Sans')

        # Değer hücreleri
        for ci, v in enumerate(values):
            cx     = col_x(ci) + DATA_COL_W / 2
            is_best = (ci == best_ci and v is not None)
            txt    = fmt_val(v, is_elo, is_dollar)

            # En iyi hücre arka planı (vurgulu sütunlarda daha belirgin)
            if is_best:
                best_rect = plt.Rectangle(
                    (col_x(ci) + 0.002, ry + 0.003),
                    DATA_COL_W - 0.004, rh - 0.006,
                    facecolor=BEST_BG, linewidth=0, zorder=3)
                ax.add_patch(best_rect)

            # Metin rengi: en iyi → altın; vurgulu sütun → beyaz/açık renkli; normal → açık gri
            if is_best:
                txt_color = BEST_COLOR
            elif ci == 0:   # Opus 4.8 sütunu
                txt_color = '#ffded0'
            elif ci == 1:   # Gemini 3.5 Flash sütunu
                txt_color = '#b8daff'
            else:
                txt_color = LIGHT_GRAY

            ax.text(cx, ry + rh / 2, txt,
                    ha='center', va='center',
                    fontsize=7.8 if is_best else 7.4,
                    fontweight='bold' if is_best else 'normal',
                    color=txt_color,
                    fontfamily='DejaVu Sans',
                    zorder=4)

        cur_y = ry

# Tablo dış çerçeve
ax.plot([LEFT, RIGHT, RIGHT, LEFT, LEFT],
        [TABLE_BOTTOM, TABLE_BOTTOM, TABLE_TOP, TABLE_TOP, TABLE_BOTTOM],
        color=DARK_BORDER, lw=0.8, zorder=6)

# --- Dipnot ---
footnote_y = TABLE_BOTTOM - 0.006
ax.text(LEFT + 0.005, footnote_y,
        '* Vending-Bench: Opus 4.7 lider, ancak fiyat anlaşması ve yalan gibi '
        'hizaya aykırı taktiklerle. Opus 4.8 bu davranışları '
        'bıraktığı için skoru düştü (Kaynak: Andon Labs).',
        ha='left', va='top', fontsize=6.0, color=MID_GRAY,
        fontfamily='DejaVu Sans', wrap=True)

# --- Alt bilgi ---
footer_y = BOTTOM_MARGIN - 0.002
ax.text(LEFT + 0.005, footer_y,
        'Hazırlayan: Prof. Dr. Oğuz Ergin · yapayzeka.oguzergin.net',
        ha='left', va='top', fontsize=6.5, color=MID_GRAY,
        fontfamily='DejaVu Sans')
ax.text(RIGHT - 0.005, footer_y,
        'Veri: ilgili model system card’ları + andonlabs.com',
        ha='right', va='top', fontsize=6.5, color=MID_GRAY,
        fontfamily='DejaVu Sans')

# --- Kaydet ---
OUT = 'G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi/opus48_gemini35_karsilastirma.png'
plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print(f'PNG kaydedildi: {OUT}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# --- Sabitler ---
FIG_W, FIG_H = 7.2, 9.6
DPI = 150
BG_COLOR = '#0f1419'
PANEL_COLOR = '#1a1f2e'
HIGHLIGHT_COL = '#1e3a5f'      # Gemini 3.5 Flash sütun arka planı
BEST_COLOR = '#ffd166'          # Altın — en iyi hücre yazı
BEST_BG = '#2a2500'             # En iyi hücre hafif arka plan
CAT_BG = '#0d2240'              # Kategori şeridi
WHITE = '#ffffff'
LIGHT_GRAY = '#aab4be'
MID_GRAY = '#6b7280'
DARK_BORDER = '#2d3748'

# --- Sütun tanımları ---
COLS = [
    ('Gemini\n3.5 Flash', '19 May 26', True),
    ('Gemini\n3 Flash',   '17 Ara 25', False),
    ('Gemini\n3.1 Pro',   '19 Şub 26', False),
    ('Claude\nSonnet 4.6','17 Şub 26', False),
    ('Claude\nOpus 4.7', '16 Nis 26', False),
    ('GPT-5.5',          '23 Nis 26', False),
]
N_COLS = len(COLS)

# --- Satır verileri ---
# Yapı: (kategori, bench_adı, açıklama, [6 değer], is_elo)
# Değer None → "—"
ROWS = [
    # KATEGORİ
    ('CAT', 'KODLAMA', '', [], False),
    ('DATA', 'Terminal-bench 2.1',    '',           [76.2, 58.0, 70.3, None, 66.1, 78.2], False),
    ('DATA', 'SWE-Bench Pro (Public)', '',          [55.1, 49.6, 54.2, None, 64.3, 58.6], False),

    ('CAT', 'EYLEMCİ GÖREVLER', '', [], False),
    ('DATA', 'MCP Atlas',   '',                     [83.6, 62.0, 78.2, 69.5, 79.1, 75.3], False),
    ('DATA', 'Toolathlon',  '',                     [56.5, 49.4, None, None, None, 55.6], False),

    ('CAT', 'ARAYÜZ DENETİMİ', '', [], False),
    ('DATA', 'OSWorld-Verified', '',                [78.4, 65.1, 76.2, 72.5, 78.0, 78.7], False),

    ('CAT', 'UZMAN GÖREVLERİ', '', [], False),
    ('DATA', 'Finance Agent v2',  'Finansal analiz',[57.9, 42.6, 43.0, 51.0, 51.5, 51.8], False),
    ('DATA', 'GDPval-AA',         'Elo',            [1656, 1204, 1314, 1676, 1753, 1769], True),

    ('CAT', 'ÇOK BİÇİMLİ', 'metin + görsel + ses + video', [], False),
    ('DATA', 'CharXiv Reasoning', 'Karmaşık grafikler', [84.2, 80.3, 83.3, 72.4, 82.1, 84.1], False),
    ('DATA', 'MMMU-Pro',           '',              [83.6, 81.2, 80.5, 74.5, 75.2, 81.2], False),
    ('DATA', 'Blueprint-Bench 2', 'Mekansal akıl', [33.6, 0.0,  26.5,  6.7, 24.5, 36.2], False),

    ('CAT', 'UZUN BAĞLAM', '', [], False),
    ('DATA', 'MRCR v2 128k', '8-iğne ort.',        [77.3, 67.2, 84.9, 84.9, 59.3, 94.8], False),
    ('DATA', 'MRCR v2 1M',   'Noktasal',           [26.6, 22.1, 26.3, None, None, None],  False),

    ('CAT', 'AKIL YÜRÜTME', '', [], False),
    ('DATA', "Humanity's Last Exam", 'Akademik',   [40.2, 33.7, 44.4, 33.2, 46.9, 41.4], False),
    ('DATA', 'ARC-AGI-2',    'Soyut akıl yürütme', [72.1, 33.6, 77.1, 58.3, 75.8, 84.6], False),
]

# En iyi hücreyi bul (None hariç, maksimum değer)
def best_col(values):
    candidates = [(i, v) for i, v in enumerate(values) if v is not None]
    if not candidates:
        return -1
    return max(candidates, key=lambda x: x[1])[0]

# Değeri metin olarak formatla
def fmt_val(v, is_elo):
    if v is None:
        return '—'
    if is_elo:
        return f'{int(v)}'
    return f'{v:.1f}%'

# --- Çizim ---
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor(BG_COLOR)

# Tüm alan tek axes üzerinde manuel çizim
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_facecolor(BG_COLOR)

# --- Alan tanımları (normalize 0..1) ---
TOP_MARGIN = 0.95
HEADER_H = 0.08      # başlık bloğu
TABLE_TOP = TOP_MARGIN - HEADER_H - 0.01
BOTTOM_MARGIN = 0.04
TABLE_H = TABLE_TOP - BOTTOM_MARGIN
LEFT = 0.01
RIGHT = 0.99
TABLE_W = RIGHT - LEFT

# Sütun genişlikleri: 1. sütun (bench adı) daha geniş
BENCH_COL_W = 0.24
DATA_COL_W = (TABLE_W - BENCH_COL_W) / N_COLS

def col_x(ci):
    """Veri sütunu ci (0..5) sol kenarı"""
    return LEFT + BENCH_COL_W + ci * DATA_COL_W

# Satır yüksekliği hesabı
N_DATA = sum(1 for r in ROWS if r[0] == 'DATA')
N_CAT  = sum(1 for r in ROWS if r[0] == 'CAT')
HEADER_ROW_H = 0.048   # sütun başlık satırı
CAT_ROW_H    = 0.024
DATA_ROW_H   = (TABLE_H - HEADER_ROW_H - N_CAT * CAT_ROW_H) / N_DATA

# --- Başlık ---
ax.text(0.5, TOP_MARGIN - 0.010, 'Gemini 3.5 Flash',
        ha='center', va='top', fontsize=22, fontweight='bold',
        color=WHITE, fontfamily='DejaVu Sans')
ax.text(0.5, TOP_MARGIN - 0.038, 'Üst Düzey Modellerle Karşılaştırma',
        ha='center', va='top', fontsize=16, color=LIGHT_GRAY, fontfamily='DejaVu Sans')
ax.text(RIGHT - 0.005, TOP_MARGIN - 0.010,
        '20 Mayıs 2026 · Google I/O',
        ha='right', va='top', fontsize=7.5, color=MID_GRAY, fontfamily='DejaVu Sans')

# --- Tablo arka planı ---
table_rect = FancyBboxPatch((LEFT, BOTTOM_MARGIN), TABLE_W, TABLE_H,
                             boxstyle='round,pad=0.002',
                             linewidth=0, facecolor=PANEL_COLOR)
ax.add_patch(table_rect)

# --- Gemini 3.5 Flash sütun vurgulama (tüm tablo boyunca) ---
hl_x = col_x(0)
hl_rect = FancyBboxPatch((hl_x, BOTTOM_MARGIN), DATA_COL_W, TABLE_H,
                          boxstyle='round,pad=0',
                          linewidth=0, facecolor=HIGHLIGHT_COL, zorder=1)
ax.add_patch(hl_rect)

# --- Sütun başlıkları ---
cur_y = TABLE_TOP
hdr_bottom = cur_y - HEADER_ROW_H

# Bench adı sütunu başlık
ax.text(LEFT + 0.005, cur_y - HEADER_ROW_H / 2, 'Benchmark',
        ha='left', va='center', fontsize=8, color=LIGHT_GRAY,
        fontweight='bold', fontfamily='DejaVu Sans')

for ci, (name, date, highlight) in enumerate(COLS):
    cx = col_x(ci) + DATA_COL_W / 2
    cy_mid = cur_y - HEADER_ROW_H / 2
    # Model adı
    ax.text(cx, cy_mid + 0.008, name,
            ha='center', va='center', fontsize=7.5,
            color=WHITE if highlight else LIGHT_GRAY,
            fontweight='bold' if highlight else 'normal',
            fontfamily='DejaVu Sans')
    # Tarih
    ax.text(cx, cy_mid - 0.010, date,
            ha='center', va='center', fontsize=6,
            color=BEST_COLOR if highlight else MID_GRAY,
            fontfamily='DejaVu Sans')

# Başlık alt çizgisi
ax.plot([LEFT, RIGHT], [hdr_bottom, hdr_bottom], color=DARK_BORDER, lw=0.7, zorder=5)
cur_y = hdr_bottom

# --- Satırlar ---
for row in ROWS:
    rtype = row[0]

    if rtype == 'CAT':
        rh = CAT_ROW_H
        ry = cur_y - rh
        # Kategori şerit arka planı
        cat_rect = plt.Rectangle((LEFT, ry), TABLE_W, rh,
                                  facecolor=CAT_BG, linewidth=0, zorder=2)
        ax.add_patch(cat_rect)
        # Başlık + opsiyonel açıklama (3. eleman doluysa)
        cat_label = row[1]
        if len(row) > 2 and row[2]:
            cat_label = f"{row[1]}   ·   {row[2]}"
        ax.text(LEFT + 0.008, ry + rh / 2, cat_label,
                ha='left', va='center', fontsize=7.5,
                color=WHITE, fontweight='bold',
                fontfamily='DejaVu Sans')
        cur_y = ry

    else:  # DATA
        _, bench, desc, values, is_elo = row
        rh = DATA_ROW_H
        ry = cur_y - rh
        best_ci = best_col(values)

        # Satır arası çizgi
        ax.plot([LEFT, RIGHT], [ry, ry], color=DARK_BORDER, lw=0.3, zorder=5)

        # Benchmark adı
        name_x = LEFT + 0.005
        if desc:
            ax.text(name_x, ry + rh * 0.62, bench,
                    ha='left', va='center', fontsize=7.2, color=WHITE,
                    fontfamily='DejaVu Sans')
            ax.text(name_x, ry + rh * 0.25, desc,
                    ha='left', va='center', fontsize=5.5, color=MID_GRAY,
                    fontfamily='DejaVu Sans')
        else:
            ax.text(name_x, ry + rh / 2, bench,
                    ha='left', va='center', fontsize=7.2, color=WHITE,
                    fontfamily='DejaVu Sans')

        # Değer hücreleri
        for ci, v in enumerate(values):
            cx = col_x(ci) + DATA_COL_W / 2
            is_best = (ci == best_ci and v is not None)
            txt = fmt_val(v, is_elo)

            # En iyi hücre arka planı
            if is_best:
                best_rect = plt.Rectangle((col_x(ci) + 0.002, ry + 0.003),
                                          DATA_COL_W - 0.004, rh - 0.006,
                                          facecolor=BEST_BG, linewidth=0, zorder=3)
                ax.add_patch(best_rect)

            ax.text(cx, ry + rh / 2, txt,
                    ha='center', va='center',
                    fontsize=7.8 if is_best else 7.5,
                    fontweight='bold' if is_best else 'normal',
                    color=BEST_COLOR if is_best else (WHITE if ci == 0 else LIGHT_GRAY),
                    fontfamily='DejaVu Sans',
                    zorder=4)

        cur_y = ry

# Tablo dış çerçeve
for spine_color in [DARK_BORDER]:
    ax.plot([LEFT, RIGHT, RIGHT, LEFT, LEFT],
            [BOTTOM_MARGIN, BOTTOM_MARGIN, TABLE_TOP, TABLE_TOP, BOTTOM_MARGIN],
            color=spine_color, lw=0.8, zorder=6)

# --- Alt yazı ---
ax.text(LEFT + 0.005, BOTTOM_MARGIN - 0.012,
        'Hazırlayan: Prof. Dr. Oğuz Ergin · yapayzeka.oguzergin.net',
        ha='left', va='top', fontsize=6.5, color=MID_GRAY, fontfamily='DejaVu Sans')
ax.text(RIGHT - 0.005, BOTTOM_MARGIN - 0.012,
        'Kaynak: deepmind.google/models/gemini/flash',
        ha='right', va='top', fontsize=6.5, color=MID_GRAY, fontfamily='DejaVu Sans')

# --- Kaydet ---
OUT = 'G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi/gemini_35_flash_karsilastirma.png'
plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print(f'PNG kaydedildi: {OUT}')

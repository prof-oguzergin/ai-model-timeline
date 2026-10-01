# -*- coding: utf-8 -*-
"""Karsilastirma tablosu butunluk denetimi. HER YAYINDAN ONCE KOS.

Neden var: 1 Eki 2026'da bir takipci "tablo kaymis" diye yazdi. Uc sutun
eklenirken (deepseek41, opus55, grok47) thead'in UST satirindaki marka
bantlarinin colspan'i guncellenmemisti; eski denetim yalniz govdedeki hucre
sayisina bakiyordu ve bunu gormedi. Bu betik tabloyu butun yonleriyle denetler,
herhangi bir kusurda sifirdan farkli cikis koduyla durur.

Kullanim:  python _kod/tablo_denetim.py
"""
import re, sys, os
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKA = [('opus55', 'ANTHROPIC'), ('fable', 'ANTHROPIC'), ('mythos', 'ANTHROPIC'),
         ('opus', 'ANTHROPIC'), ('sonnet', 'ANTHROPIC'), ('gpt', 'OPENAI'),
         ('gemini', 'GOOGLE'), ('kimi', 'MOONSHOT'), ('glm', 'Z.AI'), ('qwen', 'ALIBABA'),
         ('deepseek', 'DEEPSEEK'), ('minimax', 'MINIMAX'), ('grok', 'xAI'), ('muse', 'META')]


def bant(c):
    for o, a in MARKA:
        if c.startswith(o):
            return a
    return None


kusur = []


def k(m):
    kusur.append(m)
    print('   KUSUR:', m)


def flag(s):
    m = re.search(r"var FLAG=\{([^}]*)\}", s)
    return set(re.findall(r"'(col-[^']+)':1", m.group(1))) if m else None


sutunlar = {}
for dosya in ['model_comparison_tr.html', 'model_comparison_en.html']:
    print('=== %s' % dosya)
    s = open(os.path.join(BASE, dosya), encoding='utf-8').read()
    hd = s[s.find('<thead'):s.find('</thead>')]
    trs = re.findall(r'<tr[^>]*>(.*?)</tr>', hd, re.S)
    if len(trs) != 2:
        k('%s: thead %d satir, 2 bekleniyordu' % (dosya, len(trs)))
        continue
    cols = [re.search(r'col-([A-Za-z0-9_-]+)', a).group(1)
            for a, v in re.findall(r'<th([^>]*)>(.*?)</th>', trs[1], re.S) if 'col-' in a]
    sutunlar[dosya] = cols
    N = len(cols) + 1
    print('   sutun: %d (hucre/satir %d)' % (len(cols), N))

    # 1) ust satir: marka bantlari
    grup = []
    for a, v in re.findall(r'<th([^>]*)>(.*?)</th>', trs[0], re.S):
        cs = int(re.search(r'colspan="(\d+)"', a).group(1)) if 'colspan' in a else 1
        grup.append((re.sub(r'<[^>]+>', '', v).strip(), cs))
    if sum(c for _, c in grup) != N:
        k('%s: marka bandi toplami %d, %d olmali' % (dosya, sum(c for _, c in grup), N))
    i = 0
    for ad, cs in grup:
        if not ad:
            continue
        for c in cols[i:i + cs]:
            if bant(c) != ad:
                k('%s: %s sutunu %s bandinin altinda, %s olmali' % (dosya, c, ad, bant(c)))
        i += cs

    # 1b) SINIF ESLESMESI: sayfanin JS'i bant genisligini m-X sinifli GORUNUR
    # sutunlari sayarak hesaplar. m-X hicbir company-X ile eslesmezse sutun
    # hicbir banda sayilmaz ve bantlar kayar (1 Eki 2026: m-opus, m-gpt,
    # m-gemini, m-grok, m-sonnet tahminle yazilmisti).
    bant_sinif = set(re.findall(r'company-([\w-]+)', trs[0]))
    for a, v in re.findall(r'<th([^>]*)>(.*?)</th>', trs[1], re.S):
        c = re.search(r'col-([A-Za-z0-9_-]+)', a)
        if not c:
            continue
        m = re.search(r'\bm-([\w-]+)', a)
        if not m:
            k('%s: %s sutununda m- sinifi yok' % (dosya, c.group(1)))
        elif m.group(1) not in bant_sinif:
            k('%s: %s sutununun sinifi m-%s, hicbir bant company-%s degil (gecerli: %s)'
              % (dosya, c.group(1), m.group(1), m.group(1), sorted(bant_sinif)))

    # 2) govde
    body = s[s.find('<tbody'):s.find('</tbody>')]
    nveri = 0
    for r in re.findall(r'<tr[^>]*>(.*?)</tr>', body, re.S):
        if 'colspan' in r:
            n = int(re.search(r'colspan="(\d+)"', r).group(1))
            if n != N:
                k('%s: bolum basligi colspan %d, %d olmali' % (dosya, n, N))
            continue
        tds = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, re.S)
        nveri += 1
        ad = re.sub(r'<[^>]+>', '', tds[0]).strip()[:28]
        if len(tds) != N:
            k('%s: "%s" satiri %d hucre, %d olmali' % (dosya, ad, len(tds), N))
        nbest = len(re.findall(r'class="best', r))
        if nbest > 1:
            k('%s: "%s" satirinda %d en-iyi isareti' % (dosya, ad, nbest))
    print('   veri satiri: %d' % nveri)

    # 3) FLAG: tanimli sutunlara isaret ediyor mu
    f = flag(s)
    if f is None:
        k('%s: FLAG bulunamadi' % dosya)
    else:
        yok = sorted(x for x in f if x[4:] not in cols)
        if yok:
            k('%s: FLAG olmayan sutunlara isaret ediyor: %s' % (dosya, yok))

# 4) TR ile EN ayni sutunlara mi sahip
if len(sutunlar) == 2:
    a, b = sutunlar.values()
    if a != b:
        k('TR ve EN sutun listesi farkli')

# 5) hub'daki FLAG kopyasi tablolarinkiyle ayni mi
hub = flag(open(os.path.join(BASE, 'index.html'), encoding='utf-8').read())
tr = flag(open(os.path.join(BASE, 'model_comparison_tr.html'), encoding='utf-8').read())
if hub != tr:
    k('index.html FLAG tablodan farkli: eksik %s fazla %s' % (sorted(tr - hub), sorted(hub - tr)))

print()
if kusur:
    print('SONUC: %d KUSUR, YAYIMLAMA' % len(kusur))
    sys.exit(1)
print('SONUC: tablo saglam')

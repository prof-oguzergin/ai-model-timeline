# -*- coding: utf-8 -*-
"""Karsilastirma tablosunun MARKA GRUBU basligini sutunlardan TURETIR.

HATA (1 Eki 2026, takipci bildirdi): thead'de IKI satir var. Alttaki satir
sutunlari, ustteki satir marka bantlarini (ANTHROPIC, OPENAI, ...) colspan
ile gosteriyor. Sutun eklerken (deepseek41 10 Eyl, opus55 + grok47 22 Eyl)
yalnizca alt satir ve govde guncellendi, UST SATIR UNUTULDU. Colspan toplami
42'de kaldi, tablo 45 hucreye cikti; bantlar kaydi, sonnet46 OPENAI'nin
altinda gorundu, sondaki uc Meta sutunu bantsiz kaldi.

COZUM: colspan elle yazilmaz, sutun listesinden hesaplanir. Yeni sutun
eklenince bu betik yeniden kosulur ve bant kendiliginden duzelir.
"""
import re, sys, os
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# sutun oneki -> marka bandi (sira onemli: uzun onek once)
MARKA = [
    ('opus55', 'ANTHROPIC'), ('fable51', 'ANTHROPIC'), ('mythospre', 'ANTHROPIC'),
    ('mythos', 'ANTHROPIC'), ('opus', 'ANTHROPIC'), ('sonnet', 'ANTHROPIC'),
    ('gpt', 'OPENAI'),
    ('gemini', 'GOOGLE'),
    ('kimi', 'MOONSHOT'),
    ('glm', 'Z.AI'),
    ('qwen', 'ALIBABA'),
    ('deepseek', 'DEEPSEEK'),
    ('minimax', 'MINIMAX'),
    ('grok', 'xAI'),
    ('muse', 'META'),
]


def bant(col):
    for onek, ad in MARKA:
        if col.startswith(onek):
            return ad
    raise ValueError('bant bulunamadi: ' + col)


for dosya in ['model_comparison_tr.html', 'model_comparison_en.html']:
    p = os.path.join(BASE, dosya)
    s = open(p, encoding='utf-8').read()
    hb, hs = s.find('<thead'), s.find('</thead>')
    hd = s[hb:hs]
    trs = list(re.finditer(r'<tr[^>]*>.*?</tr>', hd, re.S))
    assert len(trs) == 2, '%s: thead %d satir, 2 bekleniyordu' % (dosya, len(trs))
    grup_tr, sutun_tr = trs[0].group(0), trs[1].group(0)

    cols = [re.search(r'col-([A-Za-z0-9_-]+)', a).group(1)
            for a, v in re.findall(r'<th([^>]*)>(.*?)</th>', sutun_tr, re.S) if 'col-' in a]

    # sutun sirasina gore ardisik bantlar
    beklenen = []
    for c in cols:
        b = bant(c)
        if beklenen and beklenen[-1][0] == b:
            beklenen[-1][1] += 1
        else:
            beklenen.append([b, 1])

    mevcut = []
    for a, v in re.findall(r'<th([^>]*)>(.*?)</th>', grup_tr, re.S):
        cs = int(re.search(r'colspan="(\d+)"', a).group(1)) if 'colspan' in a else 1
        mevcut.append((re.sub(r'<[^>]+>', '', v).strip(), cs, a, v))

    # ilk hucre etiket sutunu (bos), onu atla
    bantli = [m for m in mevcut if m[0]]
    assert len(bantli) == len(beklenen), \
        '%s: bant sayisi %d, beklenen %d -> %s vs %s' % (
            dosya, len(bantli), len(beklenen), [m[0] for m in bantli], [b[0] for b in beklenen])

    yeni_tr = grup_tr
    degisen = []
    for (ad, cs, a, v), (bad, bcs) in zip(bantli, beklenen):
        assert ad == bad, '%s: bant sirasi bozuk %s != %s' % (dosya, ad, bad)
        if cs == bcs:
            continue
        eski_th = '<th%s>%s</th>' % (a, v)
        yeni_a = re.sub(r'colspan="\d+"', 'colspan="%d"' % bcs, a) if 'colspan' in a \
            else a + ' colspan="%d"' % bcs
        yeni_tr = yeni_tr.replace(eski_th, '<th%s>%s</th>' % (yeni_a, v), 1)
        degisen.append((ad, cs, bcs))

    toplam = 1 + sum(b[1] for b in beklenen)
    assert toplam == len(cols) + 1, 'toplam %d != %d' % (toplam, len(cols) + 1)
    s = s[:hb] + hd.replace(grup_tr, yeni_tr, 1) + s[hs:]
    open(p, 'w', encoding='utf-8').write(s)
    print('%s: %d sutun | bant toplami %d' % (dosya, len(cols), toplam))
    for ad, e, y in degisen:
        print('   %-12s colspan %d -> %d' % (ad, e, y))
    if not degisen:
        print('   bantlar zaten dogru')

# -*- coding: utf-8 -*-
"""Karsilastirma sayfasinin ELLE YAZILMIS ve bayatlayan parcalarini tablodan uretir.

1) KAZANC SAYISI kartlari: her satirdaki birincilik isaretinden (.best) sayilir.
   Kartin kendi kurali: Mythos Preview haric; onun birinci oldugu satirda
   oteki modellerin en iyisi sayilir. 3 ve uzeri birincilik gosterilir.
   (1 Eki 2026'ya kadar elle yaziliydi; Fable 5'i 21 gosteriyordu, gercegi 20.
   Fable 5.1 ve DeepSeek V4-Pro 3'e cikmisti, kartta yoktular.)
2) Rozet ve sekme basligindaki tarih bugune cekilir (9 Haziran / Nisan'da kalmisti).

Kullanim: tabloya satir/sutun ekledikten sonra  python _kod/sayfa_guncelle.py
"""
import re, sys, os, datetime
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESIK = 3

AY_TR = ['OCAK', 'ŞUBAT', 'MART', 'NİSAN', 'MAYIS', 'HAZİRAN', 'TEMMUZ', 'AĞUSTOS', 'EYLÜL', 'EKİM', 'KASIM', 'ARALIK']
AY_TR_K = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
AY_EN = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']

RENK = [('fable', '#a78bfa'), ('mythos', '#a78bfa'), ('opus', '#f5b98c'), ('sonnet', '#d4866a'),
        ('gpt', '#6ee7a0'), ('gemini', '#7db8ff'), ('deepseek', '#ef4444'), ('grok', '#5cb8ff'),
        ('kimi', '#2DD4BF'), ('glm', '#EC4899'), ('qwen', '#A78BFA'), ('minimax', '#C77DFF'),
        ('muse', '#4a9eff')]


def renk(c):
    for o, r in RENK:
        if c.startswith(o):
            return r
    return '#cccccc'


def sayi(x):
    x = re.sub(r'<[^>]+>', '', x).replace('&mdash;', '').strip()
    if not x or x in '—–-':
        return None
    para = '$' in x
    x = x.replace('%', '').replace('$', '').replace(' ', '')
    x = x.replace(',', '').replace('.', '') if para else x.replace(',', '')
    try:
        return float(x)
    except ValueError:
        return None


def kazanc(s):
    hd = s[s.find('<thead'):s.find('</thead>')]
    trs = re.findall(r'<tr[^>]*>(.*?)</tr>', hd, re.S)
    cols, ad = [], {}
    for a, v in re.findall(r'<th([^>]*)>(.*?)</th>', trs[1], re.S):
        c = re.search(r'col-([\w-]+)', a)
        if not c:
            continue
        c = c.group(1)
        cols.append(c)
        t = re.sub(r'<span.*?</span>', '', v)
        t = re.sub(r'\s+', ' ', re.sub(r'<br\s*/?>', ' ', t)).replace('&nbsp;', ' ').strip()
        t = t.replace('GPT 6', 'GPT-6').replace('GPT 5', 'GPT-5')
        if c.startswith('deepseek'):
            t = 'DeepSeek ' + t
        ad[c] = t
    body = s[s.find('<tbody'):s.find('</tbody>')]
    k = Counter()
    for r in re.findall(r'<tr[^>]*>(.*?)</tr>', body, re.S):
        if 'colspan' in r:
            continue
        tds = re.findall(r'<td([^>]*)>(.*?)</td>', r, re.S)
        if len(tds) != len(cols) + 1:
            continue
        best = [cols[i] for i, (a, v) in enumerate(tds[1:]) if 'best' in a]
        if not best:
            continue
        w = best[0]
        if w == 'mythospre':
            vals = [(sayi(v), cols[i]) for i, (a, v) in enumerate(tds[1:])
                    if cols[i] != 'mythospre' and sayi(v) is not None]
            if not vals:
                continue
            w = max(vals)[1]
        k[w] += 1
    return [(c, n, ad[c]) for c, n in k.most_common() if n >= ESIK]


bugun = datetime.date.today()
for dosya, dil in [('model_comparison_tr.html', 'tr'), ('model_comparison_en.html', 'en')]:
    p = os.path.join(BASE, dosya)
    s = open(p, encoding='utf-8').read()
    liste = kazanc(s)
    kart = '  <div class="win-count">\n' + ''.join(
        '    <div class="win-item">\n'
        '      <div class="num" style="color:%s;">%d</div>\n'
        '      <div class="label">%s</div>\n'
        '    </div>\n' % (renk(c), n, a) for c, n, a in liste) + '  </div>\n'
    s2, n1 = re.subn(r'  <div class="win-count">\n.*?\n  </div>\n', lambda m: kart, s, count=1, flags=re.S)
    assert n1 == 1, dosya + ': win-count blogu bulunamadi'
    if dil == 'tr':
        rozet = 'ÖNCÜ LLM KARŞILAŞTIRMA &middot; %d %s %d' % (bugun.day, AY_TR[bugun.month - 1], bugun.year)
        baslik = 'Öncü LLM Karşılaştırma · %s %d' % (AY_TR_K[bugun.month - 1], bugun.year)
    else:
        rozet = 'FRONTIER LLM COMPARISON &middot; %s %d, %d' % (AY_EN[bugun.month - 1], bugun.day, bugun.year)
        baslik = 'Frontier LLM Comparison · %s %d' % (AY_EN[bugun.month - 1].capitalize(), bugun.year)
    s2, n2 = re.subn(r'(<div class="badge">)[^<]*(</div>)', lambda m: m.group(1) + rozet + m.group(2), s2, count=1)
    s2, n3 = re.subn(r'<title>[^<]*</title>', '<title>%s</title>' % baslik, s2, count=1)
    open(p, 'w', encoding='utf-8').write(s2)
    print('%s: %d kart | rozet %s | baslik %s' % (dosya, len(liste), 'tamam' if n2 else 'YOK', 'tamam' if n3 else 'YOK'))
    for c, n, a in liste:
        print('      %2d  %s' % (n, a))

# -*- coding: utf-8 -*-
"""Zeka endeksi verisini AA yukunden tazeler + yeni modelleri ekler.

Kullanim:  python _kod/ii_tazele.py <aa_cek.py ciktisi: aa_okt.json>
Oturum klasorunde duruyordu ve her oturumda siliniyordu; depoya alindi
(15 Eki 2026).

TAKMA AD: AA bazi modelleri yeniden adlandiriyor. Ornek, bizdeki
"Mistral Large 4" AA'da "Mistral Large 4 Preview" diye duruyor; eslesme
kurulmayinca hem bizim kayit tazelenmiyor hem de yeni model sanilip ikinci
kez ekleniyordu (12 ve 15 Eki 2026'da iki kez elle silindi).
"""
import re, sys, json
sys.stdout.reconfigure(encoding='utf-8')
aa = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'aa_okt.json', encoding='utf-8'))
biz = json.load(open(r'G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi/_kod/ii_data_open.json',
                     encoding='utf-8'))
CRE = {'SpaceXAI': 'xAI', 'Z AI': 'Z.ai', 'ByteDance Seed': 'ByteDance', 'Moonshot AI': 'Moonshot',
       'Meta Superintelligence Labs': 'Meta', 'Mistral AI': 'Mistral'}


TAKMA = {'Mistral Large 4 Preview': 'Mistral Large 4'}


def temiz(n):
    n = re.sub(r'\s*\((Adaptive Reasoning[^)]*|Reasoning|Non-reasoning|High|Low|Max|Medium|Xhigh|high|xhigh|max|medium|low|Preview|Thinking)\)', '', n)
    return re.sub(r'\s*\(.*?\)$', '', n).strip()


def anahtar(n):
    return re.sub(r'[^a-z0-9]', '', temiz(n).lower())


# ad -> (ii, tarih, uretici, acik)
aa_ad = {}
for slug, v in aa.items():
    if v['ii'] is None:
        continue
    ad = TAKMA.get(v['ad'], v['ad'])
    if ad not in aa_ad or v['ii'] > aa_ad[ad][0]:
        aa_ad[ad] = (round(v['ii'], 1), v['tarih'], CRE.get(v['uretici'], v['uretici']), v['acik'])
print('AA puanli: %d' % len(aa_ad))

# --- 1) tazele
yeni, tazelenen = [], 0
for tar, ii, ad, sir, ack in biz:
    if ad in aa_ad and abs(aa_ad[ad][0] - ii) >= 0.051:
        ii = aa_ad[ad][0]
        tazelenen += 1
    yeni.append([tar, ii, ad, sir, ack])
print('tazelenen: %d / %d' % (tazelenen, len(biz)))

# --- 2) yeni modeller (Haz 2026 sonrasi)
var = {anahtar(x[2]) for x in biz}
ekle = {}
for ad, (ii, rd, cr, ow) in aa_ad.items():
    if not rd or rd < '2026-06-01':
        continue
    if anahtar(ad) in var:
        continue
    ekle[ad] = [rd, ii, ad, cr, 1 if ow else 0]
print('eklenen: %d kayit / %d ayri model' % (len(ekle), len({anahtar(a) for a in ekle})))
for ad in sorted(ekle, key=lambda a: -ekle[a][1]):
    r = ekle[ad]
    print('   %s  %-44s %5.1f  %-14s %s' % (r[0], ad[:44], r[1], (r[3] or '?')[:14],
                                            'acik' if r[4] else 'kapali'))
yeni.extend(ekle.values())
yeni.sort()
json.dump(yeni, open(r'G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi/_kod/ii_data_open.json',
                     'w', encoding='utf-8'), ensure_ascii=False)
print()
print('ham kayit: %d -> %d' % (len(biz), len(yeni)))

best = {}
for tar, ii, ad, sir, ack in yeni:
    k = (tar, temiz(ad))
    if k not in best or ii > best[k][1]:
        best[k] = [tar, ii, temiz(ad), sir, ack]
D = sorted(best.values())


def merd(f=None):
    d = sorted([r for r in D if f is None or r[4] == f])
    en, a = -1, []
    for r in d:
        if r[1] > en:
            en = r[1]
            a.append(r)
    return a


m, ac = merd(), merd(1)
print('temizlenmis: %d' % len(D))
print('ZIRVE      : %s %.1f %s' % (m[-1][0], m[-1][1], m[-1][2]))
print('ACIK ZIRVE : %s %.1f %s' % (ac[-1][0], ac[-1][1], ac[-1][2]))
print('MAKAS      : %.1f' % (m[-1][1] - ac[-1][1]))

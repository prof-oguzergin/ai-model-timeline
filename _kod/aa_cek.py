# -*- coding: utf-8 -*-
"""AA yukunden model kaydi cikarir.

Kullanim:  python _kod/aa_cek.py <indirilen leaderboards/models sayfasi> [cikti.json]

AA bicimi sik degisiyor; desenler ESNEK ve her adimda alt sinir denetimi var.
Yuk iki diziye bolunmus, slug ile birlestiriliyor:
  A) "slug","name","deprecated","releaseDate","creator":{"name"}
  B) "slug","name","shortName",...,"intelligenceIndex","isOpenWeights"
SABIT PENCERE KULLANMA: alan sinirlari obj_span ile dengeli parantezden.

24 Eki 2026: AA sayfaya gomdugu puanli kayit sayisini 701'den 274'e dusurdu
(tarihli kayit 509'da kaldi). Eski alt sinir 600'du ve boru hatti kiriliyordu.
Sinir 200'e cekildi; bu sayi bir daha duserse bicim yine degismis demektir,
ham yuku gozle incele.
"""
import re, sys, json
sys.stdout.reconfigure(encoding='utf-8')
DOSYA = sys.argv[1]
CIKTI = sys.argv[2] if len(sys.argv) > 2 else 'aa_okt.json'
t = open(DOSYA, encoding='utf-8', errors='replace').read().replace('\\"', '"')


def obj_span(s, p):
    d, i, instr = 0, p, False
    while i > 0:
        i -= 1
        c = s[i]
        if c == '"' and s[i - 1] != chr(92):
            instr = not instr
            continue
        if instr:
            continue
        if c == '}':
            d += 1
        elif c == '{':
            if d == 0:
                break
            d -= 1
    bas = i
    d, j, instr = 0, bas, False
    while j < len(s):
        c = s[j]
        if c == '"' and s[j - 1] != chr(92):
            instr = not instr
            j += 1
            continue
        if not instr:
            if c == '{':
                d += 1
            elif c == '}':
                d -= 1
                if d == 0:
                    return bas, j + 1
        j += 1
    return bas, min(len(s), bas + 6000)


# --- B) puan + alanlar (shortName capali)
puan = {}
for m in re.finditer(r'"slug":"([^"]+)","name":"([^"]*)","shortName":"([^"]+)"', t):
    slug, ad, kisa = m.groups()
    a, b = obj_span(t, m.start())
    o = t[a:b]
    alan = {k: (None if v == 'null' else float(v))
            for k, v in re.findall(r'"([a-zA-Z][A-Za-z0-9_]*)":(-?[0-9.]+|null)', o)}
    alan['_acik'] = '"isOpenWeights":true' in o
    alan['_ad'] = ad or kisa
    alan['_kisa'] = kisa
    ii = alan.get('intelligenceIndex') or -1
    if slug not in puan or (puan[slug].get('intelligenceIndex') or -1) < ii:
        puan[slug] = alan
print('B) puanli kayit: %d' % len(puan))
assert len(puan) > 200, 'B kirik: %d kayit' % len(puan)

# --- A) tarih (release blogu bazen yok; esnek desen)
tarih = {}
for m in re.finditer(r'"slug":"([^"]+)","name":"([^"]*)"[^{}]{0,200}?"releaseDate":"(\d{4}-\d\d-\d\d)"', t):
    slug, ad, rd = m.groups()
    tarih.setdefault(slug, rd)
# ikinci gecis: releaseDate'i iceren nesneden slug'i geri bul
for m in re.finditer(r'"releaseDate":"(\d{4}-\d\d-\d\d)"', t):
    a, b = obj_span(t, m.start())
    o = t[a:b]
    sm = re.search(r'"slug":"([^"]+)"', o)
    if sm:
        tarih.setdefault(sm.group(1), m.group(1))
print('A) tarihli kayit: %d' % len(tarih))
assert len(tarih) > 300, 'A kirik: %d kayit' % len(tarih)

# --- uretici
uretici = {}
for slug, alan in puan.items():
    a = re.search(r'"slug":"%s","name":"[^"]*","shortName":"[^"]*"' % re.escape(slug), t)
    if not a:
        continue
    o = t[a.start():a.start() + 1200]
    cm = re.search(r'"modelCreatorName":"([^"]+)"', o)
    if cm:
        uretici[slug] = cm.group(1)

cik = {}
for slug, alan in puan.items():
    cik[slug] = {'ad': alan['_ad'], 'kisa': alan['_kisa'],
                 'ii': alan.get('intelligenceIndex'),
                 'tarih': tarih.get(slug), 'uretici': uretici.get(slug),
                 'acik': alan['_acik'],
                 'hle': alan.get('hle'), 'scicode': alan.get('scicode'),
                 'critpt': alan.get('critpt'), 'gpqa': alan.get('gpqa'),
                 'ifbench': alan.get('ifbench'),
                 'terminalbenchHard': alan.get('terminalbenchHard'),
                 'terminalBench40': alan.get('terminalBench40')}
json.dump(cik, open(CIKTI, 'w', encoding='utf-8'), ensure_ascii=False)
print('kaydedildi: %d model, tarihi olan %d' % (len(cik), sum(1 for v in cik.values() if v['tarih'])))

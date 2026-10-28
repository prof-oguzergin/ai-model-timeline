# -*- coding: utf-8 -*-
"""onayli_*.json anahtarlarinin su anki satir duzeniyle tuttugunu denetler.

NEDEN: anahtar "etiket|satir_y" ve satir y'si company_order'dan turuyor
(y = (N-1-i)*3.0). Yeni bir AILE eklenince y degerleri kayiyor, anahtarlar
tutmuyor ve cozucu butun etiketleri bastan yerlestiriyor. 24 Eki 2026'da
NVIDIA eklendiginde bu oldu ve koruma dort gun sessizce olu kaldi;
belirtisi "company_overrides'a yazdigim yatay kayma cizime yansimiyor".

Kullanim:  python _kod/onayli_denetim.py     -> kusurda cikis 1
Cizim GEREKTIRMEZ, saniyeler surer. Aile ekledikten sonra calistir.
"""
import json, re, sys, os
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kusur = 0

s = open(os.path.join(BASE, 'ai_timeline_final_tr.py'), encoding='utf-8').read()
m = re.search(r'company_order\s*=\s*\[(.*?)\]', s, re.S)
assert m, 'company_order bulunamadi'
sira = re.findall(r'"([^"]+)"', m.group(1))
gecerli = {round((len(sira) - 1 - i) * 3.0, 4): c for i, c in enumerate(sira)}
print('aile: %d, gecerli satir y: %s ... %s'
      % (len(sira), min(gecerli), max(gecerli)))

for dosya in ['_kod/onayli_final.json', '_kod/onayli_2yil.json']:
    yol = os.path.join(BASE, dosya)
    if not os.path.exists(yol):
        print('%-26s YOK' % dosya)
        continue
    d = json.load(open(yol, encoding='utf-8'))
    tutan, tutmayan = 0, {}
    for k in d:
        y = round(float(k.rpartition('|')[2]), 4)
        if y in gecerli:
            tutan += 1
        else:
            tutmayan[y] = tutmayan.get(y, 0) + 1
    if tutmayan:
        kusur += 1
        print('%-26s KUSUR: %d/%d anahtar tutmuyor, bayat y: %s'
              % (dosya, sum(tutmayan.values()), len(d),
                 sorted(tutmayan.items())[:6]))
    else:
        print('%-26s %d anahtarin hepsi tutuyor' % (dosya, tutan))

print()
print('SONUC: ' + ('%d DOSYA BAYAT -> python _kod/onayli_uret.py' % kusur
                   if kusur else 'onayli duzen anahtarlari saglam'))
sys.exit(1 if kusur else 0)

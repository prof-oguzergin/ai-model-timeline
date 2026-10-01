# -*- coding: utf-8 -*-
"""Onayli etiket duzenini su anki cizelge betiklerinden yeniden uretir.

Ne zaman: yeni bir yerlesim YAYIMLANDIKTAN sonra (ya da Oguz bir konumu tarif
edip duzeltme yayimlandiktan sonra). Uretilen dosyalar cozucunun sabit kabul
ettigi konumlardir; bir sonraki guncellemede yalniz yeni etiketler oynar.

  _kod/onayli_final.json   tam cizelge   (ai_timeline_final_tr.py)
  _kod/onayli_2yil.json    iki yillik    (ai_timeline_2yil_tr.py)
Anahtar "etiket|satir_y", deger [dx, dy] (nokta).
Kullanim:  python _kod/onayli_uret.py
"""
import os, sys, subprocess, tempfile, json
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DUMP = '''
fig.canvas.draw()
import json as _js
_d = {}
for _t in ax.texts:
    _p = getattr(_t, "xyann", None)
    if not isinstance(_p, tuple) or abs(_p[1]) < 20: continue
    _d["%s|%s" % (_t.get_text(), round(float(_t.xy[1]), 4))] = [_p[0], _p[1]]
open(__CIKIS__, "w", encoding="utf-8").write(_js.dumps(_d, ensure_ascii=False, indent=0))
print("@@@", len(_d))
raise SystemExit
'''

for kaynak, cikis in [('ai_timeline_final_tr.py', 'onayli_final.json'),
                      ('ai_timeline_2yil_tr.py', 'onayli_2yil.json')]:
    s = open(os.path.join(BASE, kaynak), encoding='utf-8').read()
    hedef = os.path.join(BASE, '_kod', cikis)
    s = s.replace('plt.savefig(', DUMP.replace('__CIKIS__', repr(hedef)) + '\nplt.savefig(', 1)
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(s)
        gecici = f.name
    r = subprocess.run([sys.executable, gecici], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    os.unlink(gecici)
    n = [l for l in r.stdout.splitlines() if l.startswith('@@@')]
    if not n:
        print(kaynak, 'HATA:', r.stderr[-600:]); sys.exit(1)
    print('%-26s -> _kod/%s (%s etiket)' % (kaynak, cikis, n[0][4:]))

"""LLM zaman cizelgesindeki etiket kutularinin GERCEK cakismasini olcer.
Ayrica kac etiketin YATAY kaymasi (egik cizgi) oldugunu sayar.
HER KOSUDA YENIDEN URETILIR: uretilen llm_check_run.py bayat kalirsa
eksen duzeltmelerinden onceki sayilari okurdun (bir kez oldu)."""
import re, sys, subprocess, os
sys.stdout.reconfigure(encoding='utf-8')
SRC = sys.argv[1] if len(sys.argv) > 1 else r'G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi/ai_timeline_final_tr.py'
CIK = sys.argv[2] if len(sys.argv) > 2 else 'llm_check_run.py'
s = open(SRC, encoding='utf-8').read()

CHECK = '''
# ---- OLCUM (gecici) ----
fig.canvas.draw(); _r = fig.canvas.get_renderer()
_PAY = 8   # piksel: bu kadar yakin etiketler de kusur sayilir
_kutu = []
for _t in ax.texts:
    _p = getattr(_t, "xyann", None)
    try: _bb = _t.get_window_extent(_r)
    except Exception: continue
    if _bb.width < 2 or _bb.height < 2: continue
    # Sol etiket blogu (sirket adi / ulke / bayrak) cizim alaninin
    # DISINDA ve kendi nokta kaymasi var; model etiketi degil, sayilmaz.
    # Sayilinca "egik" sayisi 24 -> 82 gorunuyordu.
    if _bb.x1 <= ax.get_window_extent(_r).x0: continue
    _kutu.append((_t.get_text(), _bb, _p if isinstance(_p, tuple) else (0, 0)))
_cak = []
for _i in range(len(_kutu)):
    for _j in range(_i+1, len(_kutu)):
        _a, _ba, _ = _kutu[_i]; _b, _bb2, _ = _kutu[_j]
        _ox = min(_ba.x1, _bb2.x1) - max(_ba.x0, _bb2.x0)
        _oy = min(_ba.y1, _bb2.y1) - max(_ba.y0, _bb2.y0)
        # PAY: kutular birbirine DEGIYOR ama ust uste binmiyorsa eski olcut
        # 0 cakisma diyordu; gozle bakinca etiketler yapisik gorunuyordu
        # (Opus 5 | Fable 5.1 | Opus 5.5). Bosluk payi eklendi.
        if _ox > -_PAY and _oy > 1: _cak.append((_a, _b, _ox, _oy))
_egik = [(t, p) for t, b, p in _kutu if abs(p[0]) > 0.5]
print("=" * 60)
print("ETIKET: %d | CAKISMA/YAPISIK: %d | YATAY KAYMALI (egik): %d" % (len(_kutu), len(_cak), len(_egik)))
for _a, _b, _ox, _oy in _cak[:18]:
    print("   CAKISMA  %-22s <-> %-22s %.0fx%.0f" % (_a[:22], _b[:22], _ox, _oy))
for _t, _p in _egik[:22]:
    print("   EGIK     %-22s dx=%s dy=%s" % (_t[:22], _p[0], _p[1]))
print("=" * 60)
'''
s = s.replace('plt.savefig(', CHECK + '\nplt.savefig(', 1)
s = re.sub(r'plt\.savefig\(f?"[^"]+"', 'plt.savefig("llm_test.png"', s)
open(CIK, 'w', encoding='utf-8').write(s)
r = subprocess.run([sys.executable, CIK], capture_output=True, text=True, encoding='utf-8', errors='replace')
out = r.stdout
i = out.find('=' * 60)
print(out[i:i + 3000] if i >= 0 else (out[-1500:] + '\n--- STDERR ---\n' + r.stderr[-1500:]))

"""LLM zaman cizelgesindeki etiketlerin GORUNEN cakismasini olcer.

Kullanim: python _kod/llm_check.py [cizelge.py] [gecici_calistirma.py]

OLCUM (cozucuyle AYNI, 1 Eki 2026):
  etiket kutusu = cizilen yuvarlak kutu (get_bbox_patch). Annotation.get_window_extent
                  KULLANMA: baglanti cizgisini de katiyor, egik etiketlerde kutu
                  noktadan etikete uzanan dikdortgen oluyor, sahte cakisma uretiyor.
  baglanti      = noktadan kutu merkezine dogru parcasi
  kusur         = iki kutu 8 px paydan yakin VEYA bir baglanti baska kutunun icinden geciyor
Yalniz cizim alanindaki model etiketleri sayilir (sol sirket blogu disarida).
HER KOSUDA YENIDEN URETILIR; uretilen calistirma dosyasini tekrar kullanma.
"""
import re, sys, subprocess
sys.stdout.reconfigure(encoding='utf-8')
SRC = sys.argv[1] if len(sys.argv) > 1 else r'G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi/ai_timeline_final_tr.py'
CIK = sys.argv[2] if len(sys.argv) > 2 else 'llm_check_run.py'
s = open(SRC, encoding='utf-8').read()

CHECK = '''
# ---- OLCUM (gecici) ----
import matplotlib.dates as _mdx
from matplotlib.text import Text as _TextX
fig.canvas.draw(); _rr = fig.canvas.get_renderer()
_PAYX = 8
_sol = ax.get_window_extent(_rr).x0
def _kx(_t):
    _bp = _t.get_bbox_patch()
    return _bp.get_window_extent(_rr) if _bp is not None else _TextX.get_window_extent(_t, _rr)
def _nx(_t):
    _x, _y = _t.xy
    if not isinstance(_x, (int, float)): _x = _mdx.date2num(_x)
    return ax.transData.transform((_x, _y))
def _ix(_p0, _p1, _b):
    for _k in range(1, 40):
        _u = _k / 40.0
        _x = _p0[0] + (_p1[0] - _p0[0]) * _u; _y = _p0[1] + (_p1[1] - _p0[1]) * _u
        if _b.x0 + 2 < _x < _b.x1 - 2 and _b.y0 + 2 < _y < _b.y1 - 2: return True
    return False
_L = []
for _t in ax.texts:
    _p = getattr(_t, "xyann", None)
    if not isinstance(_p, tuple) or abs(_p[1]) < 20: continue
    _b = _kx(_t)
    if _b.x1 <= _sol: continue
    _L.append((_t.get_text(), _b, _nx(_t), _p))
_cak = []
for _i in range(len(_L)):
    for _j in range(_i + 1, len(_L)):
        _a, _ba, _na, _ = _L[_i]; _c, _bc, _nc, _ = _L[_j]
        _ox = min(_ba.x1, _bc.x1) - max(_ba.x0, _bc.x0)
        _oy = min(_ba.y1, _bc.y1) - max(_ba.y0, _bc.y0)
        _ma = ((_ba.x0 + _ba.x1) / 2, (_ba.y0 + _ba.y1) / 2)
        _mc = ((_bc.x0 + _bc.x1) / 2, (_bc.y0 + _bc.y1) / 2)
        if _ox > -_PAYX and _oy > 1:
            _cak.append(("KUTU", _a, _c, _ox))
        elif _ix(_na, _ma, _bc):
            _cak.append(("CIZGI", _a, _c, 0))
        elif _ix(_nc, _mc, _ba):
            _cak.append(("CIZGI", _c, _a, 0))
_egik = [(t, p) for t, b, n, p in _L if abs(p[0]) > 0.5]
print("=" * 60)
print("ETIKET: %d | CAKISMA/YAPISIK: %d | YATAY KAYMALI (egik): %d" % (len(_L), len(_cak), len(_egik)))
for _tip, _a, _c, _ox in _cak[:18]:
    if _tip == "KUTU":
        print("   CAKISMA  %-22s <-> %-22s %.0f px" % (_a[:22], _c[:22], _ox))
    else:
        print("   CIZGI    %-22s cizgisi %-22s kutusundan geciyor" % (_a[:22], _c[:22]))
for _t, _p in _egik[:30]:
    print("   EGIK     %-22s dx=%s dy=%s" % (_t[:22], _p[0], _p[1]))
print("=" * 60)
'''
s = s.replace('plt.savefig(', CHECK + '\nplt.savefig(', 1)
s = re.sub(r'plt\.savefig\(f?"[^"]+"', 'plt.savefig("llm_test.png"', s)
open(CIK, 'w', encoding='utf-8').write(s)
r = subprocess.run([sys.executable, CIK], capture_output=True, text=True, encoding='utf-8', errors='replace')
out = r.stdout
i = out.find('=' * 60)
print(out[i:i + 4000] if i >= 0 else (out[-1500:] + '\n--- STDERR ---\n' + r.stderr[-1500:]))

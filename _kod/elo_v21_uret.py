# -*- coding: utf-8 -*-
"""AA v2.1 olcegindeki yeni Elo cizelgelerini uretir (gorsel + video, TR + EN).

Girdi: _kod/aa_video.json, _kod/aa_image.json — AA leaderboard sayfalarindan
cikarilan ham kayitlar (10 Kas 2026). Yenilemek icin sayfayi indirip tablo
satirlarini ayni bicimde cikar: model, uretici, elo, cikis ayi.
Cikti: video_elo_v21*.py, image_elo_v21*.py (sablon: eski elo betikleri).
Kullanim:  python _kod/elo_v21_uret.py
"""
import json, re, ast, os, sys, datetime
sys.stdout.reconfigure(encoding='utf-8')
BASE = r'G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi'
SCR = os.path.dirname(os.path.abspath(__file__))   # _kod: ham AA goruntuleri burada
AY = {m: i + 1 for i, m in enumerate(
    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}

URETICI = {'ByteDance Seed': 'ByteDance', 'Bytedance': 'ByteDance', 'SpaceXAI': 'xAI',
           'KlingAI': 'Kling', 'Black Forest Labs': 'FLUX', 'Luma Labs': 'Luma',
           'Microsoft AI': 'Microsoft', 'Z AI': 'Z.ai', 'Pika Art': 'Pika',
           'Utopai Studios': 'Utopai', 'Leonardo.Ai': 'Leonardo.Ai'}

# cizelgemizdeki bilinen kesin tarihler (AA yalniz ay veriyor)
KESIN_EK = {
    'GPT Image 2.6 Sunburst (max)': '2026-11-04', 'GPT Image 2.6 Flare (max)': '2026-11-04',
    'Nano Banana 2.1': '2026-10-06', 'FLUX 3 Image': '2026-10-01',
    'Qwen-Image-2.1': '2026-09-20', 'HunyuanImage 3.5 (Preview)': '2026-09-22',
    'Recraft V4.1 Flash': '2026-09-23', 'Ideogram 4.5 (High)': '2026-09-30',
    'Fibo 2.0': '2026-10-20', 'MAI-Image-2.7': '2026-10-27',
    'Kling 4.0': '2026-10-15', 'Pika H3 Flex': '2026-10-29',
    'Utopai X': '2026-09-30', 'HeyGen Video 1': '2026-09-30',
    'Gemini Omni Flash 1.1': '2026-08-27',
}


def eski_veri(dosya):
    s = open(os.path.join(BASE, dosya), encoding='utf-8').read()
    i = s.index('DATA = ['); j = s.index('\nL = ', i)
    return {r[2]: r[0] for r in ast.literal_eval(s[i + 7:j].rstrip())}


def tarih_bul(ad, ay_metin, eski):
    yil = int(ay_metin.split()[-1]); ay = AY[ay_metin.split()[0]]
    ayin_ilki = datetime.date(yil, ay, 1)
    if ad in KESIN_EK:
        return KESIN_EK[ad]
    if ad in eski:
        d = datetime.date.fromisoformat(eski[ad])
        if abs((d - ayin_ilki).days) <= 45:
            return eski[ad]
    return ayin_ilki.isoformat()


def uret(kaynak_json, sablon, cikis, eski_dosya, L_tr, L_en, png_tr, png_en):
    ham = json.load(open(os.path.join(SCR, kaynak_json), encoding='utf-8'))
    eski = eski_veri(eski_dosya)
    veri = []
    for v in ham:
        if v['elo'] is None:
            continue
        veri.append([tarih_bul(v['model'], v['cikis'], eski), float(v['elo']),
                     v['model'], URETICI.get(v['uretici'], v['uretici'])])
    veri.sort(key=lambda r: (r[0], r[1]))
    blok = 'DATA = [\n' + ',\n'.join(json.dumps(r, ensure_ascii=False) for r in veri) + '\n]'

    s = open(os.path.join(BASE, sablon), encoding='utf-8').read()
    i = s.index('DATA = ['); j = s.index('\nL = ', i)
    s = s[:i] + blok + s[j:]
    for hedef, L, png, eski_png in [(cikis[0], L_tr, png_tr, eski_dosya.replace('.py', '.png')),
                                    (cikis[1], L_en, png_en, None)]:
        m = re.search(r'^L = \{.*?\}$', s, flags=re.M | re.S)
        t = s[:m.start()] + 'L = ' + repr(L) + s[m.end():]
        t = re.sub(r'plt\.savefig\("[^"]+"', lambda _m: 'plt.savefig("%s"' % png, t)
        t = re.sub(r'print\("kaydedildi: [^"]+"', lambda _m: 'print("kaydedildi: %s"' % png, t)
        # 12 aylik geriye bakis yerine ilk noktadan bugune
        t = t.replace('bir_yil = fr[fr["Date"] <= son["Date"] - pd.Timedelta(days=365)]["ii"].max()',
                      'bir_yil = fr.iloc[0]["ii"]')
        open(os.path.join(BASE, hedef), 'w', encoding='utf-8').write(t)
        print(hedef, len(veri), 'kayit')
    return len(veri), veri


SUB = ('{n} modelin körlemesine insan oylamasıyla Elo puanı ({lo} – {hi})  ·  '
       'sarı merdiven: o güne kadarki en iyi  ·  ölçek: %s, 10 Kasım 2026 görüntüsü')
SUB_EN = ('Elo from blind human votes for {n} models ({lo} – {hi})  ·  '
          'yellow staircase: best at each point in time  ·  scale: %s, snapshot of 10 November 2026')
BUY = '{a:.0f} → {b:.0f}\n(+{d:.0f} puan)'
BUY_EN = '{a:.0f} → {b:.0f}\n(+{d:.0f} points)'

n_v, _ = uret(
    'aa_video.json', 'video_elo_tr.py', ('video_elo_v21_tr.py', 'video_elo_v21.py'), 'video_elo_tr.py',
    {'ylabel': 'Video Kalite Elo (AA-Video-T2V v2.1)',
     'title': 'Video Üretme Modellerinin Başarımı — Yeni Ölçek',
     'sub': SUB % 'AA-Video-T2V v2.1', 'growth': BUY, 'cloud': 'diğer ölçülen modeller',
     'credit': 'Kaynak: artificialanalysis.ai (Video Arena, v2.1)  ·  Derleyen: Prof. Dr. Oğuz Ergin'},
    {'ylabel': 'Video Quality Elo (AA-Video-T2V v2.1)',
     'title': 'Video Generation Capability — New Scale',
     'sub': SUB_EN % 'AA-Video-T2V v2.1', 'growth': BUY_EN, 'cloud': 'other measured models',
     'credit': 'Source: artificialanalysis.ai (Video Arena, v2.1)  ·  Compiled by Prof. Dr. Oğuz Ergin'},
    'video_elo_v21_tr.png', 'video_elo_v21.png')

n_i, _ = uret(
    'aa_image.json', 'image_elo_tr.py', ('image_elo_v21_tr.py', 'image_elo_v21.py'), 'image_elo_tr.py',
    {'ylabel': 'Görsel Kalite Elo (AA-Image-T2I v2.1)',
     'title': 'Görsel Üretme Modellerinin Başarımı — Yeni Ölçek',
     'sub': SUB % 'AA-Image-T2I v2.1', 'growth': BUY, 'cloud': 'diğer ölçülen modeller',
     'credit': 'Kaynak: artificialanalysis.ai (Image Arena, v2.1)  ·  Derleyen: Prof. Dr. Oğuz Ergin'},
    {'ylabel': 'Image Quality Elo (AA-Image-T2I v2.1)',
     'title': 'Image Generation Capability — New Scale',
     'sub': SUB_EN % 'AA-Image-T2I v2.1', 'growth': BUY_EN, 'cloud': 'other measured models',
     'credit': 'Source: artificialanalysis.ai (Image Arena, v2.1)  ·  Compiled by Prof. Dr. Oğuz Ergin'},
    'image_elo_v21_tr.png', 'image_elo_v21.png')
print('video', n_v, 'gorsel', n_i)

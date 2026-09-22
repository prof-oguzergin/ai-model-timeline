"""Tam LLM zaman cizelgesinden 'son iki yil' (2025-2026) surumunu turetir.
Veri tek kaynakta kalir: ai_timeline_final_*.py duzenlenince bu da yenilenir.

NOT: bu betik once oturumun gecici klasorundeydi ve her oturumda siliniyordu.
Depoya alindi (22 Eyl 2026) ki kaybolmasin.
Kullanim:  python _kod/make_2yil.py
"""
import re, sys, os
sys.stdout.reconfigure(encoding='utf-8')
BASE = r'G:/My Drive/Claude Code/YZ Model Zaman Cizelgesi'
ESIK = '2025-01-01'
NL = chr(10)

ISLER = [
    ('ai_timeline_final_tr.py', 'ai_timeline_2yil_tr.py', 'ai_timeline_2yil_tr.png',
     'Yapay Zeka Model Yayınlanma Zaman Çizelgesi', 'Son İki Yıl · Yapay Zeka Modelleri'),
    ('ai_timeline_final.py', 'ai_timeline_2yil.py', 'ai_timeline_2yil.png',
     'AI Model Release Timeline', 'The Last Two Years · AI Models'),
]

# Kaymalar sifirlandiktan sonra kalan gercek cakismalar icin hedefli deger
IKI_YIL = [('OpenAI', '4.1', '(-80, -55)'), ('OpenAI', 'o4-mini', '(80, -55)'),
           ('Anthropic', 'Fable 5.1', '(0, 55)')]   # 2 gun arayla, ayni tarafta: yatay sart

for src, hedef, png, eski_baslik, yeni_baslik in ISLER:
    s = open(os.path.join(BASE, src), encoding='utf-8').read()

    anc = 'df["Date"] = pd.to_datetime(df["Date"])'
    assert anc in s, src + ': df anchor'
    s = s.replace(anc, anc + NL +
                  '# --- SON IKI YIL SURUMU: veri esikten filtrelenir ---' + NL +
                  'df = df[df["Date"] >= "%s"].reset_index(drop=True)' % ESIK, 1)

    s = s.replace('for year in [2023, 2024, 2025, 2026]:', 'for year in [2025, 2026]:', 1)
    # eksenin sol siniri tam cizelgede sabit yazilmis (2022-11-01); veriye baglanir
    s = re.sub(r'mdates\.date2num\(pd\.Timestamp\("2022-11-01"\)\)',
               'mdates.date2num(df["Date"].min() - pd.Timedelta(days=20))', s)
    s = s.replace('"%s"' % eski_baslik, '"%s"' % yeni_baslik, 1)
    # DIKKAT: TR betigi duz dizge, EN betigi f-string kullaniyor
    # (f"{output_dir}/ai_timeline_final.png"). Yalniz duz dizgeyi arayan eski
    # desen EN'de eslesmedi ve iki yillik betik ANA INGILIZCE PNG'yi EZIYORDU.
    for eski_ad, yeni_ad in [('ai_timeline_final_tr_linkedin.png', png.replace('.png', '_linkedin.png')),
                             ('ai_timeline_final_linkedin.png', png.replace('.png', '_linkedin.png')),
                             ('ai_timeline_final_tr.png', png),
                             ('ai_timeline_final.png', png)]:
        s = s.replace(eski_ad, yeni_ad)
    kalan = re.findall(r'savefig\([^)]*?(ai_timeline_final[a-z_]*\.png)', s)
    assert not kalan, hedef + ': hedef PNG hala ana cizelgeyi gosteriyor -> %s' % kalan

    # yer bollastigi icin tum yatay kaymalari sifirla -> cozucu dik yerlestirir
    n = [0]

    def sifirla(m):
        if int(m.group(3)) == 0:
            return m.group(0)
        n[0] += 1
        return '%s%s: (0, %s)' % (m.group(1), m.group(2), m.group(4))

    s = re.sub(r'(\("[^"]+",\s*"[^"]+"\)|"[^"]+")(\s*):\s*\(\s*(-?\d+),\s*(-?\d+)\)', sifirla, s)

    for sr, ad, ofs in IKI_YIL:
        pat = re.compile(r'\("%s",\s*"%s"\)(\s*):\s*\([^)]*\)' % (re.escape(sr), re.escape(ad)))
        if pat.search(s):
            s = pat.sub(lambda m: '("%s", "%s")%s: %s' % (sr, ad, m.group(1), ofs), s)
        else:
            m2 = re.search(r'(company_overrides\s*=\s*\{' + NL + ')', s)
            s = s[:m2.end()] + '    ("%s", "%s"): %s,%s' % (sr, ad, ofs, NL) + s[m2.end():]

    open(os.path.join(BASE, hedef), 'w', encoding='utf-8').write(s)
    print('%-26s -> %-24s (%d yatay kayma sifirlandi, %d hedefli)' % (src, hedef, n[0], len(IKI_YIL)))

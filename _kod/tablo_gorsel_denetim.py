# -*- coding: utf-8 -*-
"""Karsilastirma tablosunu GERCEK TARAYICIDA cizip marka bantlarinin hizasini olcer.

Neden: bantlarin genisligini sayfadaki JavaScript CALISIRKEN hesapliyor. Her
bant (company-X) yalniz sinifi m-X olan GORUNUR modelleri sayar. 1 Eki 2026'da
eklenen bes sutuna yanlis sinif verilmisti (m-opus, m-gpt, m-gemini, m-grok,
m-sonnet); hicbir banda sayilmadilar, bantlar kaydi. HTML'e bakan denetim
(tablo_denetim.py) bunu goremedi, cunku colspan'ler cizimde yeniden yaziliyor.

Olcum: her gorunur bant icin sol/sag kenar, altindaki gorunur sutunlarin ilk
sol ve son sag kenariyla 2 px icinde olmali; hicbir gorunur sutun bantsiz
kalmamali. Hem acilis gorunumu hem "Hepsi" gorunumu denetlenir.
Kullanim:  python _kod/tablo_gorsel_denetim.py [dosya_ya_da_url]   -> kusurda 1
"""
import sys, os, pathlib
sys.stdout.reconfigure(encoding='utf-8')
from playwright.sync_api import sync_playwright

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hedefler = sys.argv[1:] or [os.path.join(BASE, 'model_comparison_tr.html'),
                            os.path.join(BASE, 'model_comparison_en.html')]
CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'

OLC = r'''() => {
  const t = document.querySelector('.tbl-scroll table');
  const comp = t.tHead.rows[0], mod = t.tHead.rows[1];
  const gor = el => el.offsetParent !== null && getComputedStyle(el).display !== 'none';
  const ths = [...mod.cells].slice(1).filter(gor);
  const bantlar = [...comp.cells].filter(c => /company-/.test(c.className) && gor(c));
  const kusur = [];
  const bantli = new Set();
  for (const b of bantlar) {
    const x = b.className.match(/company-([\w-]+)/)[1];
    const alt = ths.filter(th => th.classList.contains('m-' + x));
    const r = b.getBoundingClientRect();
    if (!alt.length) { kusur.push('bant ' + x + ' gorunur ama altinda sutun yok'); continue; }
    alt.forEach(th => bantli.add(th));
    const L = alt[0].getBoundingClientRect().left, R = alt[alt.length - 1].getBoundingClientRect().right;
    if (Math.abs(r.left - L) > 2 || Math.abs(r.right - R) > 2)
      kusur.push('bant ' + x + ' [' + r.left.toFixed(0) + ',' + r.right.toFixed(0) + '] sutunlar [' + L.toFixed(0) + ',' + R.toFixed(0) + ']');
  }
  for (const th of ths) if (!bantli.has(th))
    kusur.push('sutun ' + th.textContent.replace(/\s+/g, ' ').trim() + ' hicbir bandin altinda degil');
  return {sutun: ths.length, bant: bantlar.length, kusur};
}'''

toplam = 0
with sync_playwright() as p:
    tarayici = p.chromium.launch(executable_path=CHROME, headless=True)
    sayfa = tarayici.new_page(viewport={'width': 1500, 'height': 900})
    for h in hedefler:
        url = h if h.startswith('http') else pathlib.Path(h).as_uri()
        sayfa.goto(url)
        sayfa.wait_for_timeout(600)
        for gorunum in ['acilis', 'hepsi']:
            if gorunum == 'hepsi':
                dugme = [b for b in sayfa.query_selector_all('.msel-btn') if b.inner_text().strip() in ('Hepsi', 'All')]
                if not dugme:
                    print('   "Hepsi" dugmesi bulunamadi'); toplam += 1; continue
                dugme[0].click(); sayfa.wait_for_timeout(400)
            r = sayfa.evaluate(OLC)
            ad = os.path.basename(h.split('?')[0])
            print('%-26s %-7s %2d sutun, %2d bant | kusur %d' % (ad, gorunum, r['sutun'], r['bant'], len(r['kusur'])))
            for k in r['kusur'][:8]:
                print('      ', k)
            toplam += len(r['kusur'])
        sayfa.screenshot(path=os.path.join(os.environ.get('TEMP', '.'), 'tablo_' + os.path.basename(h.split('?')[0]) + '.png'),
                         clip={'x': 0, 'y': 250, 'width': 1500, 'height': 260})
    # TELEFON (390 px): sayfa ekrani asmamali (tablo kendi kutusunda kaymali),
    # etiket sutunu dar olmali, bantlar hizali olmali. 1 Eki 2026'ya kadar tablo
    # sayfayi ~1000 px'e genisletiyor, ekranda yalniz kiyaslama adlari kaliyordu.
    tel = tarayici.new_context(viewport={'width': 390, 'height': 844}, device_scale_factor=2,
                               is_mobile=True, has_touch=True)
    sp = tel.new_page()
    for h in hedefler:
        url = h if h.startswith('http') else pathlib.Path(h).as_uri()
        sp.goto(url)
        sp.wait_for_timeout(800)
        r = sp.evaluate(OLC)
        b = sp.evaluate('''() => ({sayfa: document.documentElement.scrollWidth, ekran: innerWidth,
            etiket: document.querySelector('.tbl-scroll tbody tr:nth-child(2) td').getBoundingClientRect().width})''')
        ad = os.path.basename(h.split('?')[0])
        k = list(r['kusur'])
        # innerWidth'e guvenme: tasan sayfada telefon tarayicisi gorunum alanini da genisletiyor
        if b['sayfa'] > 391 or b['ekran'] > 391:
            k.append('telefonda sayfa %d px, ekran %d px: tablo sayfayi genisletiyor' % (b['sayfa'], b['ekran']))
        if b['etiket'] > 140:
            k.append('telefonda etiket sutunu %.0f px, ekranin cogunu kapliyor' % b['etiket'])
        print('%-26s %-7s %2d sutun, sayfa %d/%d px, etiket %.0f px | kusur %d'
              % (ad, 'telefon', r['sutun'], b['sayfa'], b['ekran'], b['etiket'], len(k)))
        for x in k[:6]:
            print('      ', x)
        toplam += len(k)
    tel.close()
    tarayici.close()
print()
print('SONUC: %s' % ('%d KUSUR, YAYIMLAMA' % toplam if toplam else 'bantlar sutunlarla hizali'))
sys.exit(1 if toplam else 0)

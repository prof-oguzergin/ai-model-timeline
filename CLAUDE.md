# YZ Model Zaman Cizelgesi

## Proje Yapisi
- **Ana dosya (guncel):** `ai_timeline_final_tr.py` — Turkce etiketli, en son guncellenen versiyon
- `ai_timeline_final.py` — Eski Ingilizce versiyon (guncellenmeyebilir)
- `ai_timeline_vertical.py` — Dikey layout denemesi
- `ai_timeline_microsoft_test.py` — Microsoft test versiyonu

## Onemli
- Yeni model eklerken veya kontrol yaparken **her zaman `ai_timeline_final_tr.py` dosyasini kullan**
- Diger `.py` dosyalari eski/test versiyonlaridir, referans alinmamali

## Yeni model ekleme + yayinlama
1. `ai_timeline_final_tr.py`: `data` listesine `("Model", "Sirket", "YYYY-MM-DD", milestone)` ekle; `short_labels`'a kisa etiket ekle. Seri tutarliligi: ara surumler `False` (kucuk nokta), yeni nesil/major `True` (donum noktasi). Cakisma olursa `company_overrides` / `manual_overrides` ile +/-55 konumla.
2. `python ai_timeline_final_tr.py` calistir → `ai_timeline_final_tr.png` uretilir. PNG'yi kirpip gozle dogrula (cakisma + okunabilirlik).
3. `index.html`: `ASSET_V` sabitini bump et (PNG cache busting) + "Son guncelleme" tarihini 3 yerde yenile (statik HTML satiri + JS `tr` + JS `en`).
4. `git add ai_timeline_final_tr.py ai_timeline_final_tr.png index.html` + commit + push. Repo `prof-oguzergin/ai-model-timeline` → yapayzeka.oguzergin.net (GitHub Pages).

## Notlar
- Bayrak gorselleri absolute path `C:/Users/Z GAMES/flags/`'tan okunur (repo'da degil ama PNG uretiminde gerekli).
- EN sekmesi (`ai_timeline_final.png`) eski `ai_timeline_final.py`'den uretilir; TR ile senkron DEGIL, son modeller (GLM-5.2 dahil) EN'de eksik olabilir.

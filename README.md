# GABRELA IDDAA TAHMIN SISTEMI

iddaa odakli futbol tahmin, kupon takip, backtest ve model analiz paneli.

Bu repo su anda:
- Streamlit tabanli koyu temali panel
- iddaa veri guncelleme akisi
- model egitimi / tahmin / backtest
- manuel kupon kaydi ve kupon performans takibi
- market bazli ve lig x market bazli analiz

sunuyor.

Durust durum notu:
- sistem calisan bir uygulamadir
- ama "garanti kazandiran" bir sistem degildir
- en kritik gelistirme alani hala bahis stratejisi kalibrasyonudur

Detayli kullanim icin:
- [KULLANIM_KILAVUZU.md](./KULLANIM_KILAVUZU.md)

## Hizli Baslangic

Windows'ta:

```bat
baslat.bat
```

Bu komut:
1. uygun Python ortamını bulur
2. gerekirse `.venv` olusturur
3. paketleri kurar
4. `.env` yoksa `.env.example` dosyasindan uretir
5. bos bir port bulup Streamlit uygulamasini acmaya calisir

Varsayilan adres genelde:
- `http://127.0.0.1:8502`

Port doluysa `8503-8512` araligindan bos port secer.

## Kurulum Sonrasi Kontrol

Kurulumdan sonra hizli kontrol icin:

```bat
kontrol_et.bat
```

Bu script:
- Python / venv durumunu kontrol eder
- temel importlari dener
- ana uygulama dosyalarini `py_compile` ile dogrular

## Gereksinimler

- Windows 10/11
- Python 3.11+
- internet erisimi (ilk paket kurulumu ve veri guncelleme icin)

## Manuel Kurulum

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py --server.port 8502
```

## Temel Kullanim Akisi

1. `baslat.bat` ile paneli ac
2. `Veri Islemleri` bolumunden iddaa verisini cek
3. gerekirse `Model Islemleri > Modeli egit`
4. `Model Islemleri > Yaklasan maclari tahmin et`
5. `Tahminler > Bugun Oyna` ekraninda adaylari incele
6. satirlardan `Kupona ekle` ile kupon olustur
7. `Kuponlarim` sekmesinden sonucu kaydet ve performansi takip et

## Proje Yapisi

```text
C:\football_analytics
|-- football_analytics\
|   |-- __init__.py
|   |-- audit.py
|   |-- backtest.py
|   |-- config.py
|   |-- data_ingest.py
|   |-- db.py
|   |-- explain.py
|   |-- features.py
|   |-- model_compare.py
|   |-- normalize.py
|   |-- predict.py
|   |-- recommend.py
|   |-- streamlit_app.py
|   `-- train.py
|-- scripts\
|-- app.py
|-- baslat.bat
|-- kontrol_et.bat
|-- requirements.txt
|-- .env.example
|-- Dockerfile
|-- docker-compose.yml
`-- KULLANIM_KILAVUZU.md
```

## Bilinen Sinirlar

- bazi marketlerde veri kaynagina bagli olarak her macta odds gelmeyebilir
- model dogrulugu gelismektedir; sistem "tamamlandi, artik eksigi yok" seviyesinde degildir
- backtestte negatif ROI veren segmentler otomatik ayiklansa da bahis stratejisi hala kalibrasyon ister

## Onerilen Sonraki Gelistirmeler

- odds movement history
- market anomaly rozetleri
- best market selector
- kisisellestirilmis lig/market modeli

# GABRELA IDDAA TAHMIN SISTEMI

iddaa odakli futbol tahmin, kupon takip, backtest ve model analiz paneli.

Bu repo su anda:
- Streamlit tabanli koyu temali panel
- iddaa veri guncelleme akisi
- model egitimi / tahmin / backtest
- manuel kupon kaydi ve kupon performans takibi
- market bazli ve lig x market bazli analiz

sunuyor.

Bu repo **calisan bir uygulama** durumundadir, ancak durust not olarak:
- sistem gelismeye aciktir
- "garanti kazandiran" bir sistem degildir
- en kritik gelistirme alani hala bahis stratejisi kalibrasyonudur

Detayli kullanim icin:
- [KULLANIM_KILAVUZU.md](./KULLANIM_KILAVUZU.md)

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
|-- data\
|-- models\
|-- scripts\
|-- app.py
|-- baslat.bat
|-- requirements.txt
|-- .env.example
|-- Dockerfile
|-- docker-compose.yml
`-- KULLANIM_KILAVUZU.md
```

## Hızlı Başlangıç

Windows'ta:

```bat
baslat.bat
```

Bu komut:
1. uygun Python ortamını bulur
2. gerekirse `.venv` oluşturur
3. paketleri kurar
4. `.env` yoksa `.env.example`'dan üretir
5. boş bir port bulup Streamlit uygulamasını açar

Varsayılan adres genelde:
- `http://127.0.0.1:8502`

Port doluysa `8503-8512` aralığından boş port seçer.

## Gereksinimler

- Windows 10/11
- Python 3.11+
- internet erişimi (iddaa veri akışı ve pip kurulumu için)

## Manuel Kurulum

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py --server.port 8502
```

## Temel Kullanım Akışı

1. `baslat.bat` ile paneli aç
2. `Veri Islemleri` bölümünden iddaa verisini çek
3. gerekirse `Model Islemleri > Modeli egit`
4. `Model Islemleri > Yaklasan maclari tahmin et`
5. `Tahminler > Bugun Oyna` ekranında adayları incele
6. satırlardan `Kupona ekle` ile kupon oluştur
7. `Kuponlarim` sekmesinden sonucu kaydet ve performansı takip et

## Bilinen Sınırlar

- bazı marketlerde veri kaynağına bağlı olarak her maçta odds gelmeyebilir
- model doğruluğu gelişmektedir; sistem "tamamlandı, artık eksiği yok" seviyesinde değildir
- backtestte negatif ROI veren segmentler otomatik ayıklansa da bahis stratejisi hâlâ kalibrasyon ister

## Önerilen Sonraki Geliştirmeler

- odds movement history
- market anomaly rozetleri
- best market selector
- kişiselleştirilmiş lig/market modeli

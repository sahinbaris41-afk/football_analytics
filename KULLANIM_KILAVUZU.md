# KULLANIM KILAVUZU

Bu dokuman, projeyi baska bir bilgisayarda calistirmak ve ilk kullanim akisini dogru sirayla izlemek icin hazirlandi.

## 1. Baska Bilgisayarda Kurulum

### Gerekenler
- Windows 10/11
- Python 3.11 veya uzeri
- internet erisimi

### Adimlar
1. Repoyu indir:
   - GitHub uzerinden ZIP indirerek
   - veya `git clone` ile
2. Klasore gir:
   - `C:\football_analytics`
3. `baslat.bat` dosyasini calistir

Ilk acilista sistem:
- `.venv` olusturur
- kutuphaneleri kurar
- `.env` dosyasini uretir
- paneli tarayicida acmaya calisir

Kurulumdan sonra kontrol icin:
- `kontrol_et.bat`

## 2. Ilk Acilistan Sonra Ne Yapacagim?

Soldaki akis mantigi:

### [01] Kontrol Merkezi
- model tipi
- edge esigi
- bankroll
- flat stake
- kupon secim adedi
- kupon modu

### [02] Egitim Ayarlari
- tarihsel lig secimi
- sezon derinligi

### [03] Veri Islemleri
Genelde su sirayi kullan:
1. `iddaa sportsbook verisini cek`
2. gerekirse `Sahte scheduled/live kayitlari temizle`
3. gerekirse `Supheli aktif 1X2 oranlarini temizle`

### [05] Model Islemleri
Genelde su sirayi kullan:
1. `Modeli egit`
2. `Yaklasan maclari tahmin et`
3. daha sonra istersen:
   - `Calistir'de geri test`
   - `Flat vs Kelly Karsilastir`
   - `Model aciklamalarini hazirla`

## 3. Gunluk Kullanim Akisi

### Tahmin Uretme
1. `Model Islemleri > Yaklasan maclari tahmin et`
2. `Tahminler > Bugun Oyna`

Burada:
- solda aday maclar
- sagda `Gunun Kuponu`
- altta manuel kupon alani

### Listeden Dogrudan Kupon Yapma
Her satirda:
- `Kupona ekle`
- tekrar basinca `Cikar`

mantigi vardir.

Yani:
1. soldaki listeden maclari sec
2. alttaki manuel kupon alanina otomatik dussun
3. iddaa toplam oranini ve kupon tutarini gir
4. `Kuponlarima kaydet`

## 4. Kupon Sonucunu Sisteme Islemek

`Kuponlarim` sekmesinde:
- kupon gecmisini gorursun
- kuponu `Acik / Kazandi / Kaybetti` olarak isaretleyebilirsin
- boylece kisisel performans ozetin olusur

Bu bolum sunlari takip etmek icin onemlidir:
- toplam stake
- kar / zarar
- ROI
- hangi liglerde daha basarili oldugun
- hangi marketlerde daha basarili oldugun

## 5. Kullanilabilen Marketler

Veri geldigi olcude sistem su marketleri degerlendirebilir:
- `Mac Sonucu`
- `2.5 Alt`
- `2.5 Ust`
- `KG Var / KG Yok`
- `Cifte Sans`
- `Handikapli Mac Sonucu 1:0`

Not:
- her macta tum marketler gelmeyebilir
- bu tamamen iddaa veri kaynaginda o marketin o anda bulunmasina baglidir

## 6. Ekranlar Ne Ise Yariyor?

### Genel Durum
- sistem saglik raporu
- veri kaynak ozeti
- KPI kartlari

### Mac Merkezi
- hizli bulten gorunumu
- market rozetleri

### Tahminler
- gunluk adaylar
- gunun kuponu
- manuel kupon

### Geri Test
- gecmiste stratejinin nasil calistigini gosterir
- en durust ekranlardan biridir

### Model Laboratuvari
- hata analizi
- market bazli analiz
- lig x market segment analizi

### Kuponlarim
- kendi oynadigin kuponlari saklar
- performans takibi saglar

## 7. Bilinen Sinirlar

Bu sistem su anda:
- kurulabilir
- calistirilabilir
- tahmin uretir
- kupon kaydeder
- geri test yapar
- performans takibi saglar

Ama durust sinirlar:
- sistem hala gelisen bir yapidir
- tahmin kalitesi anlamlidir ama kusursuz degildir
- en kritik acik alan hala bahis stratejisi optimizasyonudur
- bazi liglerde ve bazi marketlerde veri eksikligi olabilir

## 8. Sorun Olursa

### Uygulama Acilmiyorsa
- `baslat.bat` ile tekrar dene
- farkli port aciyorsa, ekranda yazan adresi kullan
- `kontrol_et.bat` ile temel saglik kontrolu yap

### Tahmin Uretilmiyorsa
Su sirayi dene:
1. `iddaa sportsbook verisini cek`
2. `Supheli aktif 1X2 oranlarini temizle`
3. `Modeli egit`
4. `Yaklasan maclari tahmin et`

### Marketler Gorunmuyorsa
- o macta veri kaynaginda market oddsi yok olabilir
- ozellikle `Cifte Sans` ve `Handikap` icin bu normaldir

## 9. Onerilen Baslangic Ayari

En guvenli baslangic:
- kupon modu: `Temkinli`
- kupon kaynagi: `Sadece IDDAA`
- guclu segment filtresi: once `Kapali`, sonra test ederek ac
- flat stake: sabit kucuk miktarla basla

## 10. Son Soz

Bu sistemin gucu tek basina modelden gelmez.
Asil guc:
- duzenli veri
- dogru filtreleme
- disiplinli kupon secimi
- geri testten ogrenen strateji

Bu nedenle en iyi sonuc, sistemi sabirli ve olculu kullaninca gelir.

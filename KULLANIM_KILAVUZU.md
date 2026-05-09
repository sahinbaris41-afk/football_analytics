# KULLANIM KILAVUZU

Bu doküman, projeyi başka bir bilgisayarda çalıştırmak ve ilk kullanımı doğru sırayla yapmak için hazırlandı.

## 1. Başka Bilgisayarda Kurulum

### Gerekenler
- Windows 10/11
- Python 3.11 veya üzeri
- internet erişimi

### Adımlar
1. Bu klasörü indir:
   - GitHub üzerinden zip indirerek
   - veya `git clone` ile
2. Klasöre gir:
   - `C:\football_analytics`
3. Çift tıkla:
   - `baslat.bat`

İlk açılışta sistem:
- `.venv` oluşturur
- kütüphaneleri kurar
- `.env` dosyasını üretir
- paneli tarayıcıda açar

## 2. İlk Açılıştan Sonra Ne Yapacağım?

Soldaki akış mantığı:

### [01] Kontrol Merkezi
- model tipi
- edge eşiği
- bankroll
- flat stake
- kupon seçim adedi
- kupon modu

### [02] Eğitim Ayarları
- tarihsel lig seçimi
- sezon derinliği

### [03] Veri İşlemleri
Genelde şu sırayı kullan:
1. `iddaa sportsbook verisini cek`
2. gerekirse `Sahte scheduled/live kayitlari temizle`
3. gerekirse `Supheli aktif 1X2 oranlarini temizle`

### [05] Model İşlemleri
Genelde şu sırayı kullan:
1. `Modeli egit`
2. `Yaklasan maclari tahmin et`
3. daha sonra istersen:
   - `Calistir'de geri test`
   - `Flat vs Kelly Karsilastir`
   - `Model aciklamalarini hazirla`

## 3. Günlük Kullanım Akışı

### Tahmin Üretme
1. `Model Islemleri > Yaklasan maclari tahmin et`
2. `Tahminler > Bugun Oyna`

Burada:
- solda aday maçlar
- sağda `Gunun Kuponu`
- altta manuel kupon alanı

### Listeden Doğrudan Kupon Yapma
Artık her satırda:
- `Kupona ekle`
- tekrar basınca `Cikar`

mantığı var.

Yani:
1. soldaki listeden maçları seç
2. alttaki manuel kupon alanına otomatik düşsün
3. iddaa toplam oranını ve kupon tutarını gir
4. `Kuponlarima kaydet`

## 4. Kupon Sonucunu Sisteme İşlemek

`Kuponlarim` sekmesinde:
- kupon geçmişini görürsün
- kuponu `Acik / Kazandi / Kaybetti` olarak işaretleyebilirsin
- böylece kişisel performans özetin oluşur

Bu bölüm şunları takip etmek için önemlidir:
- toplam stake
- kar / zarar
- ROI
- hangi liglerde daha başarılı olduğun
- hangi marketlerde daha başarılı olduğun

## 5. Kullanılabilen Marketler

Veri geldiği ölçüde sistem şu marketleri değerlendirebilir:
- `Mac Sonucu`
- `2.5 Alt`
- `2.5 Ust`
- `KG Var / KG Yok`
- `Cifte Sans`
- `Handikapli Mac Sonucu 1:0`

Not:
- her maçta tüm marketler gelmeyebilir
- bu tamamen iddaa veri kaynağında o marketin o anda bulunmasına bağlıdır

## 6. Model Ekranları Ne İşe Yarıyor?

### Genel Durum
- sistem sağlık raporu
- veri kaynak özeti
- KPI kartları

### Maç Merkezi
- hızlı bülten görünümü
- market rozetleri

### Tahminler
- günlük adaylar
- günün kuponu
- manuel kupon

### Geri Test
- geçmişte stratejinin nasıl çalıştığını gösterir
- en dürüst ekranlardan biridir

### Model Laboratuvarı
- hata analizi
- market bazlı analiz
- lig x market segment analizi

### Kuponlarım
- kendi oynadığın kuponları saklar
- performans takibi sağlar

## 7. Bilinen Sınırlar

Bu sistemi çalıştırabilir, kullanabilir ve başka bilgisayarda ayağa kaldırabilirsin.  
Ama dürüst sınırlar:

- sistem hâlâ gelişen bir yapıdır
- tahmin kalitesi anlamlıdır ama kusursuz değildir
- en kritik açık alan hâlâ bahis stratejisi optimizasyonudur
- bazı liglerde ve bazı marketlerde veri eksikliği olabilir

## 8. Sorun Olursa

### Uygulama Açılmıyorsa
- `baslat.bat` ile tekrar dene
- farklı port açıyorsa, ekranda yazan adresi kullan

### Tahmin Üretilmiyorsa
Şu sırayı dene:
1. `iddaa sportsbook verisini cek`
2. `Supheli aktif 1X2 oranlarini temizle`
3. `Modeli egit`
4. `Yaklasan maclari tahmin et`

### Marketler Görünmüyorsa
- o maçta veri kaynağında market odds’ı yok olabilir
- özellikle `Cifte Sans` ve `Handikap` için bu normaldir

## 9. Önerilen Kullanım Şekli

En güvenli başlangıç:
- kupon modu: `Temkinli`
- kupon kaynağı: `Sadece IDDAA`
- güçlü segment filtresi: önce `Kapali`, sonra test ederek aç
- flat stake: sabit küçük miktarla başla

## 10. Son Söz

Bu sistem şu an:
- kurulabilir
- çalıştırılabilir
- tahmin üretir
- kupon kaydeder
- geri test yapar
- performans takibi sağlar

Ama onu “mükemmel” yapan şey tek bir model değil; düzenli veri, doğru filtreleme ve disiplinli kullanım olur.


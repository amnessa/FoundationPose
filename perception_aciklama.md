# Algılama hattı: laptop → sunucu → laptop

Bu belge, robotun kamerasından alınan **tek bir RGB-D karenin** laptop'tan çıkıp
masaüstü sunucuda işlenmesini ve sonucun laptop'a geri dönmesini baştan sona
anlatır. Her adımda verinin **şekli, birimi ve hangi koordinat çerçevesinde
olduğu** yazılıdır. Kullanılan yöntemlerin (nokta bulutu, PPF, FoundationPose)
nasıl çalıştığı da ayrı bölümlerde açıklanır.

Sunucu tarafının İngilizce teknik referansı (ayarlar, uç noktalar, hata
belirtileri) için: [readme_perception.md](readme_perception.md).
PPF ayrıntıları için: [docs/PPF.md](docs/PPF.md).

---

## İçindekiler

1. [Genel bakış](#1-genel-bakış)
2. [Kamera: Intel RealSense D435i](#2-kamera-intel-realsense-d435i)
3. [Laptop: kareyi yakalayıp göndermek](#3-laptop-kareyi-yakalayıp-göndermek)
4. [CAD dosyaları](#4-cad-dosyaları)
5. [Sunucu: veriyi almak](#5-sunucu-veriyi-almak)
6. [Segmentasyon: SAM2](#6-segmentasyon-sam2)
7. [Nokta bulutuna dönüştürme](#7-nokta-bulutuna-dönüştürme)
8. [PPF: hangi parça?](#8-ppf-hangi-parça)
9. [FoundationPose: parça nerede, nasıl duruyor?](#9-foundationpose-parça-nerede-nasıl-duruyor)
10. [Sunucunun cevabı](#10-sunucunun-cevabı)
11. [Laptop: cevabı almak ve yayınlamak](#11-laptop-cevabı-almak-ve-yayınlamak)
12. [Her aşamada verinin özeti](#12-her-aşamada-verinin-özeti)
13. [Süreler](#13-süreler)
14. [Kontrol edilecekler ve açık sorular](#14-kontrol-edilecekler-ve-açık-sorular)

---

## 1. Genel bakış

```
┌──────────────────────── LAPTOP (robot tarafı, ROS 2) ────────────────────────┐
│                                                                              │
│  D435i ──▶ /camera/color/image_raw   (RGB)                                   │
│        ──▶ /camera/depth/...         (depth)                                 │
│                    │                                                         │
│                    ▼                                                         │
│        foundationpose_bridge_node                                            │
│          • RGB ve depth zaman olarak eşleşmiş mi? (≤ 0.25 s)                 │
│          • PNG'ye çevir, camera.json'u hazırla                               │
│          • HTTP POST  ──────────────────────────────────────────┐            │
│                                                                 │            │
└─────────────────────────────────────────────────────────────────┼────────────┘
                                                                  │  rgb.png
                                                                  │  depth.png
                                                                  │  camera.json
┌──────────────────── MASAÜSTÜ (GPU, Docker, fp_server.py :5000) ─┼────────────┐
│                                                                 ▼            │
│   [5] veriyi çöz:  RGB 720×1280×3,  depth 720×1280 (metre),  K 3×3           │
│                                     │                                        │
│   [6] SAM2:        tık + RGB  ──▶  maske 720×1280 (0/1)                      │
│                                     │                                        │
│   [7] nokta bulutu: maske ⊙ depth, K ile geri izdüşüm ──▶ N×6, N ≤ 2000      │
│                                     │                                        │
│   [8] PPF:         sahne bulutu vs. CAD kütüphanesi ──▶ "test_objv2_ear"     │
│                                     │                                        │
│   [9] FoundationPose: RGB + depth + maske + test_objv2_ear.ply               │
│                                     ──▶ T (4×4), kamera çerçevesi, metre     │
│                                     │                                        │
│  [10] cevap: JSON (isim, poz, skor tablosu, dosyalar)  ─────────┐            │
└─────────────────────────────────────────────────────────────────┼────────────┘
                                                                  │
┌──────────────────────── LAPTOP ─────────────────────────────────┼────────────┐
│                                                                 ▼            │
│  [11] foundationpose_bridge_node                                             │
│         • birim metre mi? kontrol et                                         │
│         • dosyaları foundationpose_results/ klasörüne yaz                    │
│         • /perception/detections   (Detection3DArray, poz + quaternion)      │
│         • /perception/object_name  ("test_objv2_ear", latched)               │
│                    │                                                         │
│                    ▼                                                         │
│        ICP düğümü: aynı isimli .ply'yi yükler, T'ye oturtur, takip eder      │
└──────────────────────────────────────────────────────────────────────────────┘
```

Özetle laptop sunucuya **iki görüntü ve bir kamera dosyası** gönderiyor. Sunucu
ona **bir parça adı ve bir 4×4 poz** döndürüyor. Arada olan her şey (maske, nokta
bulutu, sınıflandırma, poz) sunucuda yapılıyor.

İki bilgisayarın aynı parçadan bahsedebilmesi için tek bir ortak anahtar var:
**CAD dosyasının adı.** Bu yüzden iki taraftaki CAD klasörü aynı dosyaları aynı
adlarla içeriyor (bkz. [Bölüm 4](#4-cad-dosyaları)).

---

## 2. Kamera: Intel RealSense D435i

D435i bir **stereo derinlik kamerası**. İki kızılötesi kamera ve sahneye desen
düşüren bir projektör ile derinlik ölçüyor. Ayrıca ayrı bir RGB kamerası var.
"i" harfi dahili IMU anlamına geliyor. Bu hatta IMU kullanılmıyor.

| akış | biçim | bu projede |
|---|---|---|
| RGB | 8 bit, 3 kanal | 1280×720 |
| Depth | 16 bit tek kanal, her piksel **Z değeri** (optik eksen boyunca uzaklık), ham birim = 1 mm | 1280×720 |

**Kamera iç parametreleri (K)**, renk kamerasınınki (`Data/Input/camera.json`):

```
        ┌ fx   0  cx ┐   ┌ 919.47     0    650.85 ┐
    K = │  0  fy  cy │ = │    0    918.94   350.62 │
        └  0   0   1 ┘   └    0       0       1   ┘
```

- `fx, fy`: piksel cinsinden odak uzaklığı. Bir metre ötedeki 1 mm'lik bir şey
  yaklaşık 0.92 piksel tutar.
- `cx, cy`: optik eksenin görüntüyü deldiği piksel (görüntü merkezine yakın).

**Kamera koordinat çerçevesi** (`camera_color_optical_frame`): orijin renk
kamerasının optik merkezi. **X sağa, Y aşağıya, Z ileriye** (sahneye doğru)
bakıyor. Bu belgedeki bütün 3B noktalar ve pozlar bu çerçevede. Robot tabanına
dönüşüm bu hattın dışında, laptop'ta yapılıyor.

**Hizalama (alignment).** Derinlik ve renk farklı kameralardan geliyor. Bir
`depth[v,u]` pikselinin `rgb[v,u]` ile aynı noktaya bakması için depth'in renk
kamerasına **hizalanması** gerekiyor. RealSense sürücüsü bunu
`/camera/aligned_depth_to_color/image_raw` topic'inde yapıyor. Hizalanmamış bir
depth aynı boyutta olsa bile pikseller kayık olur. Maske nesnenin yanındaki
derinlikleri alır. Bkz. [Bölüm 14](#14-kontrol-edilecekler-ve-açık-sorular).

---

## 3. Laptop: kareyi yakalayıp göndermek

Bu işi `foundationpose_bridge_node.py` (ROS 2 düğümü) yapıyor.

1. **Abone olur:** RGB topic'i (`/camera/color/image_raw`) ve depth topic'i.
   Her topic'in son mesajını tutar.
2. **Tetiklenir:** `ros2 service call /foundationpose_bridge/trigger std_srvs/srv/Trigger`
   ile, ya da `auto_trigger=true` ise ilk eşleşen çiftte bir kez.
3. **Senkron kontrolü:** Son RGB ile son depth mesajının zaman damgaları arasındaki
   fark **0.25 s'den büyükse** gönderilmez. Farklı anlara ait bir RGB ve depth
   eşleştirilmiş olmasın diye.
4. **Kodlar:**
   - RGB → `rgb.png` (8 bit, 3 kanal),
   - depth → `depth.png` (16 bit, mm),
   - `camera.json` → `{"cam_K": [9 sayı], "depth_scale": 1.0}`.
     `depth_scale`, ham depth değerini mm'ye çeviren çarpandır.
5. **Gönderir:** Üç dosya tek bir HTTP `multipart/form-data` isteğiyle
   `http://<masaüstü>:5000/predict_pose` adresine POST edilir. Aynı dosyalar
   laptop'taki aktarım klasörüne de yazılır.
6. **Bekler:** Zaman aşımı **300 s**. Sunucuda operatörün nesneye tıklaması
   gerekebileceği için uzun tutuldu.

---

## 4. CAD dosyaları

### 4.1 Nerede duruyorlar?

| | konum |
|---|---|
| Sunucu | `Data/Input/*.ply` (`CAD_DIR`). PPF kütüphanesi bu klasördeki **bütün** `.ply` dosyalarından kurulur. |
| Laptop | ICP düğümünün CAD klasörü. **Aynı dosyalar, aynı adlarla.** |

Sunucu cevapta parçanın adını gönderiyor (`"test_objv2_ear"`). Laptop bu adın
sonuna `.ply` ekleyip kendi klasöründen yüklüyor. Bu nedenle:

- İki taraftaki dosya **adları** birebir aynı olmalı.
- İki taraftaki dosya **içerikleri** aynı olmalı. Sunucu pozu kendi dosyasına göre
  hesaplıyor. Laptop'taki dosyanın orijini veya ekseni farklıysa aynı poz başka bir
  yere oturur.
- Yeni bir parça iki tarafa birden eklenmeli. Sunucuya eklemek için
  `/add_model` uç noktası var (bkz. readme_perception.md).

### 4.2 Dosya biçimi

Hepsi **ikili (binary little-endian) PLY** ve hepsi **üçgen ağı (mesh)**:

- `vertex`: köşe noktaları, yalnızca `x, y, z` (float32). **Normal, renk veya
  doku yok.**
- `face`: her üçgenin hangi üç köşeden oluştuğu.
- **Birim: milimetre.** Sunucu yüklerken `MESH_SCALE = 0.001` ile metreye çevirir.
  Bu çarpan bütün dosyalara aynı uygulanır. Bu yüzden her dosyanın mm olarak dışa
  aktarılmış olması gerekir. FreeCAD'de dışa aktarma birimi belge başına bir ayar,
  dikkat edilmesi gerekiyor.

Mesh nokta bulutu değildir. Örneğin bir kutu 8 köşe ve 12 üçgenden oluşur. PPF
için yüzeyden nokta örneklenmesi gerekir ([Bölüm 8.2](#82-hazırlık-offline-cad--model-bulutu)).

### 4.3 Kütüphanedeki dosyalar

Boyutlar dosyalardan okundu (eksen hizalı kutu, mm):

| dosya | kaynak | köşe | üçgen | boyut (mm) | not |
|---|---|---:|---:|---|---|
| `test_objv2_ear.ply` | FreeCAD | 8 | 12 | 250 × 8 × 99 | ince plaka (kulak) |
| `test_objv2_base.ply` | FreeCAD | 8 | 12 | 249 × 8 × 256 | ince plaka (taban) |
| `test_objv2.ply` | FreeCAD | 24 | 44 | 256 × 250 × 108 | taban + kulak birleşik |
| `test_objv3.ply` | FreeCAD | 24 | 44 | 256 × 250 × 108 | `test_objv2` ile aynı dış boyut |
| `test_objv1_base.ply` | FreeCAD | 40 | 76 | 180 × 4 × 100 | ince plaka |
| `test_objv1_ear.ply` | FreeCAD | 16 | 28 | 54 × 4 × 100 | ince plaka |
| `test_objectv1.ply` | FreeCAD | 24 | 44 | 180 × 100 × 54 | v1 birleşik |
| `plate.ply` | FreeCAD | 8 | 12 | 150 × 4 × 100 | `test_objv1_base`'e çok benzer |
| `smallplate.ply` | FreeCAD | 8 | 12 | 100 × 50 × 4 | |
| `vplate.ply` | FreeCAD | 12 | 20 | 100 × 104 × 100 | V şeklinde |
| `Tblock.ply` | FreeCAD | 16 | 28 | 150 × 80 × 100 | T profil |
| `270circle.ply` | FreeCAD | 384 | 764 | 150 × 50 × 150 | 270°'lik ince halka, kavisli |
| `powerdrill.ply` | Artec 3B tarayıcı | 249 998 | 500 000 | 131 × 248 × 258 | taranmış gerçek nesne |
| `assembly_mesh.ply` | trimesh | 16 | 24 | 298 × 304 × 119 | program tarafından yazılmış montaj |

Bu tablodan çıkan sonuçlar:

- Parçaların çoğu **düzlemsel yüzlü, birkaç köşeli** basit katılar. Köşe sayısı
  geometrinin ne kadar ayrıntılı olduğunu değil, kaç düz yüzü olduğunu gösteriyor.
- **Ayırt etmesi zor çiftler:** `plate` ile `test_objv1_base` (iki ince dikdörtgen,
  tek kenarda 30 mm fark). `test_objv2` ile `test_objv3` (dış boyutları aynı,
  yalnızca şekil ayırıyor).
- `powerdrill` tarama verisi, diğerlerinden farklı türde. 20 MB, yüklemesi en yavaş
  olan dosya.
- `assembly_mesh.ply` FreeCAD'den değil, bir programdan geliyor ve `CAD_DIR`
  içinde durduğu için kütüphaneye de giriyor. Sınıflandırma adayı olması
  isteniyor mu, kontrol edilmeli.

---

## 5. Sunucu: veriyi almak

`fp_server.py`, `/predict_pose` isteğini aldığında (`_receive_frame`):

1. Üç dosyayı `Data/Input/` altına kaydeder (`rgb.png`, `depth.png`, `camera.json`).
2. `camera.json` → `K` (3×3) ve `depth_scale`.
3. `rgb.png` → 720×1280×3 dizi. OpenCV BGR okur, RGB'ye çevrilir.
4. `depth.png` → **metre**: `Z = ham × depth_scale / 1000`.
5. Geçersiz derinlikler 0 yapılır: < 1 mm (sensör ölçememiş, D435i'de bu değer 0
   gelir) veya ≥ 3 m (`ZFAR`).
6. RGB ve depth boyutları farklıysa istek reddedilir.

**Çıktı:** `rgb` (720×1280×3, uint8), `depth` (720×1280, float32, m), `K` (3×3).

Sunucu aynı anda tek istek işler. İkinci bir istek gelirse "meşgul" (HTTP 503)
döner. GPU tek, tıklama penceresi de tek.

---

## 6. Segmentasyon: SAM2

### 6.1 Ne yapıyor?

FoundationPose'un kendi nesne bulucusu yok. Hangi piksellerin nesneye ait olduğu
ona **söylenmeli.** Bunu SAM2 (Meta, *Segment Anything Model 2*) yapıyor.

- **Giriş:** yalnızca RGB görüntü ve bir veya birkaç **tık** (piksel koordinatı).
  Depth kullanılmıyor.
- **Çıktı:** 720×1280 ikili maske (nesne = 1, geri kalan = 0).

### 6.2 Nasıl çalışıyor?

SAM2 "istemle" (prompt) çalışan bir segmentasyon ağı. "Bu noktayı içeren nesne
hangisi?" sorusuna maske ile cevap veriyor. Sınıf adı bilmiyor ve nesneye özel
eğitim gerektirmiyor. Çok büyük bir maske veri setinde, "bir şeyin sınırı nerede"
sorusunu genel olarak öğrenmiş durumda. Üç parçadan oluşuyor:

```
 RGB ──▶ görüntü kodlayıcı (Hiera) ──▶ özellik haritası ─┐
                                                         ├──▶ maske çözücü ──▶ maske(ler) + skor
 tıklar ──▶ istem kodlayıcı ──────────▶ istem vektörleri ─┘
```

1. **Görüntü kodlayıcı:** Görüntüyü bir kez işleyip her bölgeyi anlatan bir özellik
   haritası çıkarır. En pahalı kısım budur.
2. **İstem kodlayıcı:** Her tıkın konumunu ve etiketini (1 = nesne, 0 = nesne değil)
   bir vektöre çevirir.
3. **Maske çözücü:** İkisini birleştirip maskeyi ve maskenin ne kadar iyi olduğuna
   dair bir tahmin (tahmini IoU) üretir.

Kullanılan model: `sam2.1_hiera_small`.

**Tek tık belirsizdir.** Bir parçanın ortasına tıklamak "parçanın tamamı", "bu yüz"
veya "bu delik" anlamına gelebilir. Tek tıkta SAM2'den **3 aday maske** istenir ve
tahmini skoru en yüksek olan alınır. Birden fazla tıkta tek maske istenir.

**Zemin nasıl dışarıda kalıyor?** Ayrı bir zemin çıkarma adımı yok. Tık parçanın
üstünde olduğu için SAM2 parçayı döndürüyor. Masa ayrı bir bölge. Maske masaya
taşarsa sağ tık (negatif nokta) ile oyuluyor.

### 6.3 Tık nereden geliyor?

Öncelik sırasıyla:

1. İstemci hazır bir `mask` dosyası gönderdiyse doğrudan o kullanılır, SAM2 atlanır.
2. İstemci `click` alanı gönderdiyse (`{"u":640,"v":360}` veya nokta listesi +
   `click_labels`) SAM2 bu noktalarla çalışır. Ekran gerekmez.
3. İkisi de yoksa sunucuda bir pencere açılır. Operatör sol tık (nesne) / sağ tık
   (nesne değil) ile işaretler, ENTER ile onaylar. Maske her tıktan sonra yeniden
   çizilir.

---

## 7. Nokta bulutuna dönüştürme

Kod: `scene_cloud_from_mask()` (`scripts/ppf_classifier.py`).

### 7.1 Kamera modeli: 3B nokta → piksel ve geri

Kamera, uzaydaki bir `(X, Y, Z)` noktasını şu pikselde görür (iğne deliği modeli):

```
u = fx · X / Z + cx
v = fy · Y / Z + cy
```

Depth görüntüsü bize her piksel için `Z`'yi veriyor. Denklemi ters çevirince o
pikselin uzaydaki noktası bulunuyor:

```
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
Z = depth[v, u]
```

**Örnek:** `u = 900, v = 400` pikselinde `Z = 0.50 m` olsun:
`X = (900 − 650.85) · 0.50 / 919.47 ≈ 0.135 m`,
`Y = (400 − 350.62) · 0.50 / 918.94 ≈ 0.027 m`.
Nokta kameranın 13.5 cm sağında, 2.7 cm aşağısında, 50 cm ilerisinde.

### 7.2 Adımlar

```
 maske (720×1280, 0/1)    depth (720×1280, m)
          └────────── ⊙ ──────────┘           eleman eleman çarpım
                      │
                      ▼
      maskeli depth (720×1280): maske dışı 0, içi gerçek Z     ← hâlâ bir görüntü
                      │
                      ▼  Z > 1 mm olan her (u,v) için geri izdüşüm
      N×3 nokta (X, Y, Z), metre, kamera çerçevesi              ← artık nokta bulutu
                      │
                      ▼  normal tahmini + temizlik + seyreltme
      N×6 (x, y, z, nx, ny, nz),  N ≤ 2000
```

1. **Maske 2 piksel aşındırılır (erode).** Nesnenin kenarındaki pikseller hem
   nesneyi hem arkasındaki masayı görür. Derinlikleri ikisinin karışımıdır ve
   nesnenin etrafında var olmayan bir "etek" oluşturur. Kenarı atmak bunu önler.
2. **Maske biraz genişletilmiş hâliyle (4 px dilate) geri izdüşülür.** Bu genişletme
   yalnızca normaller için yapılıyor (3. madde).
3. **Yüzey normalleri tahmin edilir.** Her noktanın en yakın 12 komşusuna bir düzlem
   oturtulur. Düzlemin dik vektörü o noktanın normalidir (OpenCV
   `computeNormalsPC3d`). Kenardaki bir noktanın komşuları tek taraftadır ve normali
   içe yatar. Genişletilmiş bölgede hesaplayıp kenarı sonra atınca kalan her noktanın
   tam bir komşuluğu olur.
4. **Normaller kameraya doğru çevrilir.** CAD'lerdeki yüzey normalleri dışa bakar.
   Sahnedekiler de dışa, yani kameraya bakmalı. Yoksa PPF'deki açılar tutmaz.
5. **Genişletilmiş kısım atılır**, yalnızca aşındırılmış maskenin içi kalır.
6. **Voksel ile seyreltilir.** Uzay küçük küplere bölünür ve her küpten bir nokta
   tutulur. Küp kenarı = max(nesne çapı / 60, 1.5 mm). Sonra en fazla **2000 nokta**
   kalacak şekilde rastgele seçilir.

### 7.3 Sonuç

**Sahne bulutu:** N×6, N ≤ 2000, metre, kamera çerçevesi.

**Neden 6 sütun?** Her satır bir noktadır. 6 sayı **konum + yüzey normali**dir,
konum + yönelim (roll-pitch-yaw) **değil**:

```
   x,  y,  z   → noktanın konumu (metre)
  nx, ny, nz   → o noktadaki yüzeyin dışa bakan dik doğrultusu (birim vektör, uzunluğu 1)
```

Bir noktanın kendi yönelimi olmaz. Normal, noktanın üzerinde bulunduğu **yüzeyin
hangi yöne baktığını** söyler. Uzunluğu hep 1 olduğu için 3 sayıyla yazılır ama
yalnızca 2 serbestlik derecesi taşır. Bir masanın üst yüzündeki her noktanın
normali yukarı bakar. Bir kutunun yan yüzündeki noktalarınki yana bakar.

Örneğin `(0.135, 0.027, 0.500, 0, 0, −1)` satırı şunu söyler: nokta kameranın 50 cm
ilerisinde ve bu noktada yüzey dümdüz kameraya bakıyor. Kamera +Z yönüne baktığı
için kameraya dönük bir yüzeyin normali −Z olur.

Bu bulut **kısmi**: kamera nesneye tek yönden baktığı için yalnızca görünen yüzler
var. Altı ve arkası yok. PPF ve FoundationPose bunu hesaba katacak şekilde
çalışıyor (bkz. 8.6 ve 9.2).

---

## 8. PPF: hangi parça?

Kod: `scripts/ppf_classifier.py`, OpenCV `cv2.ppf_match_3d` modülü üzerine.
Yöntem: Drost vd., *Model Globally, Match Locally* (CVPR 2010). Doğrulama skoru:
Birdal & Ilic (3DV 2015).

**Giriş:** sahne bulutu (N×6) ve kütüphanedeki her CAD'in model bulutu.
**Çıktı:** yalnızca bir **isim** (ör. `"test_objv2_ear"`) ve tüm modellerin skor
tablosu. PPF bu hatta **sınıflandırıcı** olarak kullanılıyor, poz vermiyor.

### 8.1 Temel fikir: Point Pair Feature

Tek bir nokta tek başına bir şey anlatmaz. Ama **iki nokta ve normalleri**, aradaki
geometri hakkında bilgi verir. İki nokta `m1`, `m2`, normalleri `n1`, `n2`,
aralarındaki vektör `d = m2 − m1` olsun:

```
                n1                     n2
                ↑                    ↗
                │       d           /
               m1 ─────────────▶ m2

F(m1, m2) = ( ‖d‖ ,      ∠(n1, d) ,  ∠(n2, d) ,  ∠(n1, n2) )
              uzaklık    açı         açı         iki normal arası açı
```

Bu **4 sayı bir nokta çiftini** tanımlar. Tek bir noktayı tanımlamaz. Nesneyi nasıl
döndürür ya da kaydırırsanız kaydırın bu sayılar **değişmez.** Uzaklıklar ve açılar
katı harekette korunur. Bu sayede PPF, nesnenin pozunu bilmeden sahnedeki çiftlerle
modeldeki çiftleri karşılaştırabilir.

`‖d‖` metre cinsinden. Yani PPF **ölçekten bağımsız değil.** İki katı büyüklükte
aynı şekildeki bir parçanın çiftleri farklı uzaklıklar üretir. Depth kamerası
metrik ölçtüğü için nesneye uzaktan bakmak bulutu küçültmez. Sadece nokta sayısı
azalır, gürültü artar. Yani çözülmesi gereken bir ölçek belirsizliği yok.

### 8.2 Hazırlık (offline): CAD → model bulutu

Her CAD bir kez işlenir (`build_ppf_library.py`, `.npz` yoksa sunucu açılışta da
yapar):

1. Mesh yüklenir, mm → m ölçeklenir.
2. Yüzeyden **alanla orantılı 60 000 rastgele nokta** örneklenir. Her nokta, üzerine
   düştüğü üçgenin **kesin normalini** alır (CAD'in normalleri sahneninkinin aksine
   gürültüsüz).
3. Voksel ile **~600 noktaya** seyreltilir. Her vokselden ortalama değil, bir
   temsilci alınır. Ortalama köşeleri yuvarlar ve iki yüz arasında hiçbir yüze ait
   olmayan normaller üretir.
4. Model bulutu (**~600×6**), çapı ve boyutları `Data/ppf_library.npz` dosyasına
   kaydedilir.

**Bu 600×6 hangi çerçevede?** Sahne bulutuyla aynı anlam (konum + normal), ama
**hiçbir yere dönüştürülmemiş** hâlde: **CAD'in kendi çerçevesinde.** Orijin ve
eksenler FreeCAD'de parça çizilirken neredeyse orada. Tek değişiklik mm → m
ölçeklemesi.

Yani elimizde iki ayrı çerçevede iki bulut var:

```
  model bulutu  →  CAD çerçevesi    (offline, CAD'den)
  sahne bulutu  →  kamera çerçevesi (canlı, depth görüntüsünden)

  ikisini bağlayan dönüşüm  =  T  =  zaten bulmaya çalıştığımız poz
```

PPF'nin işe yaramasının sebebi de bu. 8.1'deki özellikler (uzaklık ve açılar) hangi
çerçevede hesaplanırsa hesaplansın aynı çıkar. Bu yüzden T'yi bilmeden iki bulutu
karşılaştırabiliyoruz.

### 8.3 PPF3DDetector'ın iki fonksiyonu

OpenCV'nin `PPF3DDetector` sınıfının dışarıya açık iki fonksiyonu var. İkisi de
aynı özelliği (`F`) hesaplıyor ama farklı bulutlarda ve farklı amaçla:

```
  trainModel(model bulutu)     offline / açılışta   CAD bulutundan hash tablosu kurar
  match(sahne bulutu)          her istekte          depth'ten gelen bulutun çiftlerini
                                                    tabloda arar, oy toplar, poz önerir
```

Yani PPF **iki tarafta da** çalışıyor. Model tarafında sonuçlar tabloya **yazılıyor**,
sahne tarafında (kameranın depth görüntüsünden üretilen bulutta) tabloda
**aranıyor.** Kütüphanedeki her CAD'in kendi dedektörü, yani kendi tablosu var.
Sahne bulutu sırayla her birinde aranıyor.

### 8.4 Eğitim (`trainModel`): hash tablosu

1. **OpenCV kendi seyreltmesini yapar.** Modelin sınırlayıcı kutusu her eksende 25
   parçaya bölünür (adım 0.04 = 1/25). Aynı hücreye düşen noktaların konumları ve
   normalleri ortalanır. 600 noktalık bulut bu adımdan sonra kütüphanemizde
   **339–591 noktaya** iniyor (ör. `test_objv2_ear`: 600 → ~543).
2. **Bütün sıralı çiftler** için `F` hesaplanır. Bu adımda kalan nokta sayısı
   M olsun. Her nokta kendisi dışındaki her noktayla eşleşir:

   ```
   çift sayısı = M × (M − 1)          ← faktöriyel değil
   ```

   `test_objv2_ear` için 543 × 542 ≈ **294 000 çift.** Kütüphanede model başına
   115 000 ile 349 000 arasında. Çiftler **sıralıdır**, yani (a, b) ile (b, a)
   ayrı çiftlerdir. Çünkü `∠(n1, d)` ile `∠(n2, d)` yer değiştirince farklı bir
   özellik çıkar. Sırasız sayılsaydı M(M−1)/2 olurdu. Faktöriyel (M!) ise bütün
   noktaların dizilme sayısıdır, burada geçmiyor.
3. Her çift için ayrıca bir **α_m açısı** saklanır. α_m şu demek: `m1` orijine
   taşınıp normali x eksenine çevrildiğinde, `m2` x ekseni etrafında hangi açıda
   duruyor? Bu açı eşleştirmede pozu bulmak için gerekli (8.5, 4. adım).
4. `F` nicelenir: uzaklık, çapın %4'ü genişliğinde kutulara, her açı 12°'lik
   kutulara bölünür (360° / 30). Yakın değerler aynı kutuya düştüğü için gürültüye
   dayanıklı olur.
5. Nicelenmiş `F` bir **anahtar** olur. Hash tablosuna
   `anahtar → [(m1, m2, α_m), …]` yazılır: "bu özellik modelde şu çiftlerde
   görülüyor".

Sonuçta elde bir matris yok, **"özellik → model çiftleri" sözlüğü** var. OpenCV
bu tabloyu diske yazamadığı için sunucu her açılışta yeniden eğitiyor (model başına
~1–2 s). `.npz` dosyasında tablo değil, model bulutları duruyor.

### 8.5 Eşleştirme (`match`): oylama

İki bulutun nokta sayısı ve sırası farklı. Hangi sahne noktasının hangi model
noktasına karşılık geldiği bilinmiyor. PPF bunu **oylama** ile buluyor.

**Rastgelelik yok, her şey sistematik.** Test karesinde (`test_objv2_ear`)
ölçülen sayılarla:

1. **Sahne de seyreltilir.** 2000 noktalık sahne bulutu 8.4'teki aynı 25×25×25
   ızgarayla **~490 noktaya** iner. (2000 noktanın hepsi doğrulamada, 8.6'da,
   kullanılıyor.)
2. **Referans noktalar:** Bu ~490 noktanın listesinde **her 5. nokta** (0., 5.,
   10., …) referans nokta `s_r` olur: **~98 referans nokta.**
3. **Her `s_r`, kalan bütün sahne noktalarıyla** (`s_i`, ~489 nokta) eşleştirilir.
   Seçme yok, komşuluk sınırı yok. Model başına ~98 × 489 ≈ **48 000 sahne çifti.**
4. Her `(s_r, s_i)` çifti için:
   - `F(s_r, s_i)` hesaplanıp tabloda aranır,
   - sahne için de α_s açısı hesaplanır (8.4, 3. adımdaki gibi),
   - tabloda bulunan her model çifti `(m_r, m_i, α_m)` bir öneri yapar:
     "`s_r` aslında modeldeki `m_r` noktası. İki çifti çakıştırmak için normal
     etrafında `α = α_m − α_s` kadar döndürmek gerekiyor",
   - öneri, **o `s_r`'ye ait** oy tablosunun `[m_r, α]` hücresine **1 oy** ekler.
     Tablonun boyutu: model nokta sayısı × 30 açı kutusu.
5. `s_r`'nin bütün çiftleri bitince **oy tablosundaki en yüksek hücre** seçilir.
   `s_r ↔ m_r` eşleşmesi ve `α` birlikte **tek bir tam poz** (4×4) verir:
   - `s_r`'yi orijine taşıyıp normalini x eksenine çevir,
   - x ekseni etrafında `α` kadar döndür,
   - `m_r`'nin aynı işleminin tersini uygula.

   Bu pozun **oy sayısı** o hücredeki oy sayısıdır.

Yanlış eşleşmeler oylarını rastgele hücrelere dağıtır. Doğru eşleşme hep aynı
hücreye oy yığar (Hough dönüşümü mantığı). Noktaların sırası ve sayısı bu yüzden
önemli değil.

### 8.6 Doğrulama: 12 poz hipotezi ve hangi modelin kazandığı

**12 hipotez nereden geliyor?**

```
 ~98 referans nokta ──▶ her biri 1 poz ──▶ ~98 ham poz
                                              │  birbirine yakın olanlar birleştirilir
                                              ▼  (konum ve açı olarak yakın; oylar toplanır,
                                                  pozların ortalaması alınır)
                                         55 küme ── oy toplamına göre sıralı
                                              │
                                              ▼  bizim kod: ilk 12'yi al
                                         12 poz hipotezi
```

- 5. adımda **her referans nokta bir poz önerir.** Referans nokta nesnenin doğru
  bir yerindeyse bu poz doğru pozdur. Birçok referans nokta aynı doğru pozu bulur.
- OpenCV birbirine yakın pozları **kümeler** ve oylarını toplar. Test karesinde
  `test_objv2_ear` için ~98 ham poz **55 kümeye** indi. Kümelerin oyları
  884, 583, 459, 334, … şeklinde sıralandı.
- Bizim kod bu listenin **ilk 12'sini** alıyor (`n_hypotheses = 12`). Sadece birinciyi
  almıyoruz, çünkü simetrik veya düz parçalarda en çok oyu yanlış bir yönelim
  alabiliyor. İlk birkaç hipotezden biri doğruysa doğrulama onu bulur.

**Model seçimi.** Oy sayısı **farklı modeller arasında karşılaştırılamaz.** Çok
noktalı, kendini tekrar eden yüzeyleri olan büyük bir model, sahnede olmasa bile
küçük bir modelden fazla oy toplar. Bu yüzden her aday model için:

1. Yukarıdaki **12 poz hipotezi** alınır.
2. Her biri **ICP** (Iterative Closest Point, 50 iterasyon) ile birkaç milimetre
   düzeltilir. ICP iyi bir başlangıç pozu ister, onu PPF veriyor. Ayrıca PPF'nin
   12°'lik açı kutuları pozda küçük bir hata bırakıyor, ICP onu kapatıyor.
3. Model bu pozda sahneye yerleştirilir. Kameradan **görülemeyecek** noktalar
   atılır: kameraya sırtını dönen yüzler ve modelin kendi kendini örttüğü kısımlar.
   Sahne kısmi olduğu için modelin de yalnızca görünen kısmıyla karşılaştırılması
   gerekiyor.
4. İki yönlü skor, `τ = 8 mm` (sensör gürültüsü payı):

```
coverage  = (modele τ'dan yakın sahne noktası) / (tüm sahne noktaları)
            → "sahnenin ne kadarını bu model açıklıyor?"
explained = (sahneye τ'dan yakın görünür model noktası) / (tüm görünür model noktaları)
            → "modelin görünmesi gereken kısmının ne kadarı sahnede var?"
skor      = coverage × explained          (0 ile 1 arası)
```

Yalnızca `coverage` kullanılsa büyük bir model sahneyi örterek kazanır. Yalnızca
`explained` kullanılsa küçük bir model sahnenin içine saklanarak kazanır. Çarpım
ikisini de cezalandırır.

5. Her modelin skoru en iyi hipotezininkidir. **En yüksek skor kazanır.**
   Birinci ile ikinci arasındaki fark (**margin**) kararın ne kadar güvenli
   olduğunu gösterir.

### 8.7 Boyut ön-filtresi

Asıl maliyet doğrulamada (12 hipotez × ICP × skor), bu yüzden önce çok ucuz bir
eleme yapılır. Sahne bulutunun
çapı (noktalar arası uzaklıkların %99.5'lik değeri) her modelin çapıyla
karşılaştırılır:

- Sahne modelden **%15'ten fazla büyükse** model elenir. Kısmi bir görüntü gerçek
  nesneden büyük olamaz.
- Sahne modelin **%45'inden küçükse** model elenir. Örtülme görünen kısmı
  küçültebilir, bu yüzden bu sınır gevşek tutuldu.

Gerçek test karesinde 13 modelden 8'i burada elendi. Hepsi elenirse filtre yok
sayılır ve bütün modeller puanlanır.

### 8.8 PPF'nin pozu neden kullanılmıyor?

8.5–8.6'da PPF bir poz hesaplıyor. Ama bu poz yalnızca skor için var: bir CAD'in
bulutu açıklayıp açıklamadığını sormak için onu bir yere koymak gerekiyor. Bu poz
sınıflandırıcıdan **dışarı çıkmıyor.** FoundationPose pozu sıfırdan hesaplıyor.
FoundationPose'un daha doğru olduğu **varsayıldı, ölçülmedi** (bkz. Bölüm 14).

### 8.9 Sınırlar

- Ayırt edici özellik görünmüyorsa (ör. alt yüzdeki bir kabartma) tek görüşten
  ayırt edilemez. Margin küçülür. Bu bir hata değil, bilginin olmadığını gösteren
  bir sinyal.
- `plate` ile `test_objv1_base` tek görüşten çok zor ayrılıyor.
- İnce kavisli parçalar (`270circle`) daha ince bir örnekleme adımı istiyor.

---

## 9. FoundationPose: parça nerede, nasıl duruyor?

Kod: `estimater.py` (NVIDIA FoundationPose, CVPR 2024), sunucuda `EST.register()`.

**Giriş:** `K`, `rgb` (720×1280×3), `depth` (m), `mask`, ve PPF'nin seçtiği `.ply`
(metreye ölçeklenmiş).
**Çıktı:** `T` (4×4), **nesnenin kamera çerçevesindeki pozu**:

```
        ┌ r11 r12 r13 | tx ┐     R (3×3): nesnenin kameraya göre dönmesi
    T = │ r21 r22 r23 | ty │     t (3×1): nesnenin orijininin kameraya göre konumu (m)
        │ r31 r32 r33 | tz │
        └  0   0   0  |  1 ┘     CAD'deki bir nokta p  →  kamerada  R·p + t
```

### 9.1 Neden FoundationPose?

FoundationPose **yeni nesneler için yeniden eğitim gerektirmiyor.** CAD verilen her
nesnede doğrudan çalışıyor. Ağlar çok büyük bir sentetik veri setinde (binlerce
farklı nesne, render edilmiş sahneler) eğitilmiş. Öğrendiği şey belirli bir parça
değil, "bir CAD'in render'ı ile gerçek görüntü arasındaki farka bakıp pozu
düzeltmek". Bu yüzden kütüphaneye yeni bir parça eklemek yalnızca dosyayı vermek
demek.

### 9.2 Nasıl çalışıyor? "Render et ve karşılaştır"

```
  hipotez üret ──▶ her hipotezi iyileştir (×5) ──▶ hepsini puanla ──▶ en iyisi
  (~250 poz)        refiner ağı                     scorer ağı
```

1. **Depth ön işleme:** Aşındırma ve bilateral filtre ile gürültü azaltılır.
2. **Öteleme tahmini:** Maskenin sınırlayıcı kutusunun merkez pikseli, maskedeki
   derinliklerin **medyanı** ile geri izdüşülür (Bölüm 7.1'deki formül). Bu, nesnenin
   kabaca nerede olduğunu verir.
3. **Dönme hipotezleri:** Nesnenin etrafındaki bir küre üzerinde **42 bakış yönü**
   (ikosfer köşeleri) alınır. Her yön için **6 düzlem içi dönme** (60°'de bir)
   eklenir: **252 aday poz.** Birbirine 30°'den yakın olanlar ve nesnenin simetrisi
   yüzünden aynı görünenler birleştirilir. Hepsi 2. adımdaki ötelemeyle başlar.
4. **İyileştirme ağı (refiner),** her hipotez için 5 tur:
   - CAD o pozda **render edilir** (renk ve derinlik görüntüsü olarak),
   - gerçek RGB-D görüntünün aynı bölgesi kırpılır,
   - ağ iki görüntüyü karşılaştırıp **"pozu şu kadar döndür, şu kadar kaydır"**
     düzeltmesini tahmin eder,
   - düzeltme uygulanır, tekrar render edilir.
5. **Puanlama ağı (scorer):** İyileştirilmiş bütün hipotezler için render ile gerçek
   görüntünün ne kadar uyuştuğunu puanlar. Hipotezleri birbirleriyle de karşılaştırarak
   sıralar.
6. **En yüksek puanlı poz** döndürülür.

Notlar:

- Bizim CAD'lerimizde renk veya doku yok, render düz gri çıkıyor. Bu yüzden
  karşılaştırma ağırlıklı olarak **geometriye ve derinliğe** dayanıyor.
- Dönen `score` (son çalıştırmada 67.97) normalize edilmemiş bir sıralama puanı,
  olasılık değil.
- Ağlar parçadan bağımsız ve sunucu açılırken bir kez yükleniyor. Parça değiştirmek
  sadece `reset_object()` çağrısı, milisaniyeler sürüyor. PPF'nin her istekte farklı
  bir CAD seçebilmesi bu sayede mümkün.
- Simetrik parçalarda (düz plakalar gibi) birden fazla poz aynı görünür.
  `SYMMETRY_INFO` verilmezse sonuç bu eşdeğer pozlar arasında atlayabilir.
- FoundationPose'un bir **takip modu** da var (`track_one`): önceki pozdan başlayıp
  her yeni karede sadece iyileştirme yapıyor. Sunucu bunu **kullanmıyor.** Takibi
  laptop'taki ICP yapıyor.

**Son çalıştırmadan örnek** (`Data/Output/foundationpose_results/detection_pem.json`):
parça `test_objv2_ear`, öteleme ≈ (164, 45, 499) mm. Yani kameradan yaklaşık 50 cm
ileride, 16 cm sağda, 4.5 cm aşağıda.

---

## 10. Sunucunun cevabı

### 10.1 HTTP cevabı (JSON)

```jsonc
{
  "status": "success",
  "units": "m",
  "object_name": "test_objv2_ear",           // laptop bu adla .ply yükleyecek
  "object_file": "test_objv2_ear.ply",
  "pose": [[r11,r12,r13,tx], …, [0,0,0,1]],  // 4×4, metre, kamera çerçevesi
  "score": 67.97,                            // FoundationPose puanı
  "elapsed_sec": …,                          // register() süresi
  "classification": {                        // PPF'nin kararı
    "score": 0.588, "margin": 0.438, "runner_up": "test_objv2_base",
    "scores":   { "test_objv2_ear": 0.588, "test_objv2_base": 0.150, … },
    "rejected": { "Tblock": "scene 255mm exceeds CAD 181mm by 41%", … },
    "elapsed_sec": 1.4
  },
  "mask_source": "sam2(interactive click)",
  "artifacts": { …base64 dosyalar… }
}
```

### 10.2 Dosyalar (`artifacts`)

Cevaba base64 olarak gömülüyor ve sunucuda `Data/Output/foundationpose_results/`
klasörüne de yazılıyor:

| dosya | içerik | kim kullanıyor |
|---|---|---|
| `detection_pem.json` | poz, eski SAM-6D biçiminde: `R` (3×3) ve `t` **milimetre**, `obj_name` | laptop ICP düğümü |
| `detection_ism.npz` | `segmentation`: (1, 720, 1280) maske | laptop ICP düğümü (sahne bulutunu kırpmak için) |
| `object_name.txt` | parça adı, düz metin | dosya izleyen herhangi bir düğüm |
| `mask.png` | maske, görsel | insan |
| `vis_pose.png` | RGB üstüne çizilmiş poz kutusu ve eksenler | insan, ilk bakılacak dosya |

**Birim farkına dikkat:** HTTP cevabındaki `pose` **metre**, `detection_pem.json`
içindeki `t` **milimetre.** İkincisi eski ICP düğümünün beklediği biçim olduğu için
korundu.

---

## 11. Laptop: cevabı almak ve yayınlamak

`foundationpose_bridge_node.py` cevabı aldığında:

1. **Kontrol eder:** HTTP 200 mü, `status == "success"` mi, `units == "m"` mi?
   Birim metre değilse cevabı reddeder. Milimetre ile metre karışırsa poz 1000 kat
   yanlış olur.
2. **Dosyaları yazar:** `artifacts` içindekileri `foundationpose_results/`
   klasörüne çözer. ICP düğümünün `results_dir` parametresi bu klasörü göstermeli.
3. **PPF kararını loglar:** ilk üç skor ve margin. Belirsizse uyarı verir.
4. **`min_score` kontrolü** (varsayılan 0, yani kapalı).
5. **Yayınlar:**
   - `/perception/detections` (`vision_msgs/Detection3DArray`):
     - `header.frame_id` = RGB mesajının çerçevesi (`camera_color_optical_frame`),
     - `id` ve `class_id` = parça adı,
     - `pose.position` = `T`'nin öteleme sütunu (metre),
     - `pose.orientation` = `T`'nin 3×3'lük dönme kısmından çevrilen **quaternion**,
     - `score` = FoundationPose puanı.
   - `/perception/object_name` (`std_msgs/String`): parça adı.

   İkisi de **latched** (`TRANSIENT_LOCAL`). Sonradan başlayan bir düğüm de son
   sonucu alır.
6. **ICP düğümü:** Adı alır, **kendi CAD klasöründen aynı adlı `.ply`'yi**
   yükler, `T` pozuna oturtur ve kameradan gelen canlı bulut üzerinde ICP ile
   takibe başlar.

---

## 12. Her aşamada verinin özeti

| # | nerede | veri | şekil | birim | çerçeve |
|---|---|---|---|---|---|
| 1 | laptop | RGB | 720×1280×3 uint8 | – | renk kamerası pikselleri |
| 2 | laptop | depth | 720×1280 uint16 | mm | renk kamerası pikselleri (hizalıysa) |
| 3 | laptop → sunucu | `camera.json` | K 3×3 + `depth_scale` | piksel | – |
| 4 | sunucu | depth | 720×1280 float32 | m | piksel |
| 5 | sunucu | maske | 720×1280 bool | – | piksel |
| 6 | sunucu | sahne bulutu | N×6, N ≤ 2000 | m | kamera |
| 7 | sunucu (offline) | model bulutu | ~600×6 her CAD için | m | CAD'in kendi çerçevesi |
| 8 | sunucu | PPF sonucu | isim + skor tablosu | skor 0–1 | – |
| 9 | sunucu | poz `T` | 4×4 | m | kamera (CAD → kamera) |
| 10 | sunucu → laptop | JSON + dosyalar | `pose` 4×4, `t` (pem) | **m** (JSON) / **mm** (pem) | kamera |
| 11 | laptop | `Detection3DArray` | konum + quaternion | m | `camera_color_optical_frame` |

---

## 13. Süreler

| aşama | süre | durum |
|---|---|---|
| Sunucu açılışı: FoundationPose ağları + PPF eğitimi + SAM2 | PPF model başına ~1–2 s | bir kez |
| Tıklama | operatöre bağlı | `click` gönderilirse yok |
| Nokta bulutu + PPF sınıflandırma | ~1.4 s (13 model) | ölçüldü |
| └ bunun içinde `match` (oylama), model başına | ~0.07 s | ölçüldü (CPU, test karesi) |
| └ geri kalanı: 12 hipotez × ICP + skor, her aday model için | baskın kısım | ayrıca ölçülmedi |
| FoundationPose `register()` | cevapta `elapsed_sec` | ayrıca ölçülmedi |
| Tüm zincir | gözlemle < 10 s | adım adım ölçülmedi |
| (Eski SAM-6D hattı) | 1–1.5 dk | karşılaştırma için |

---

## 14. Kontrol edilecekler ve açık sorular

**Kontrol edilecekler:**

- [ ] **Depth hizalaması.** Bridge'in varsayılan depth topic'i
      `/camera/depth/image_rect_raw`. Bu topic renk kamerasına **hizalı değil.**
      Sunucu yalnızca boyutların eşit olup olmadığına bakıyor, hizalamayı
      denetlemiyor. Laptop'ta `depth_topic` parametresinin
      `/camera/aligned_depth_to_color/image_raw` olduğu (veya `normalize_depth_image`
      fonksiyonunun hizalama yaptığı) doğrulanmalı. `camera.json`'daki K renk
      kamerasına ait. Hizasız depth ile kullanılırsa noktalar kayar.
- [ ] **CAD eşliği.** Laptop'taki ve sunucudaki CAD klasörleri aynı dosyaları aynı
      adlarla ve aynı içerikle tutuyor mu? Bir sağlama toplamı (`md5sum`) karşılaştırması
      yeterli.
- [ ] **`assembly_mesh.ply`** kütüphanede olmalı mı? `CAD_DIR` içinde durduğu için
      şu an aday.

**Toplantıdan (1 Ekim 2026) gelen açık sorular:**

- [ ] PPF pozu ile FoundationPose pozunu 3–5 farklı sahnede karşılaştır (doğruluk +
      süre). FoundationPose kullanmanın gerekçesi bu tablo olmalı.
- [ ] FoundationPose takibi (`track_one`) ile ICP takibini karşılaştır.
- [ ] Süreleri adım adım ölç: SAM2, model başına PPF, FoundationPose.
- [ ] D435i'nin en doğru ölçtüğü mesafeyi bul. Minimum mesafe ~28 cm, ama doğruluk
      nerede en iyi? `PPF_TAU` (8 mm) buna göre ayarlanmalı.
- [ ] İki aşamalı bakış: uzaktan tanı, en iyi mesafeye yaklaşıp pozu yeniden hesapla.
- [ ] Parça yerine konduktan sonra pozu bir kez daha hesapla. Takipte biriken hata
      sıfırlansın.
- [ ] Margin küçükse başka bir açıdan bak (*next-best-view*).

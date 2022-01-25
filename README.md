# Hitters_Linear_Regression

![image](https://user-images.githubusercontent.com/84872652/151046087-e66ef7de-92ac-405a-b5de-b8ee114a67cb.png)

Kullanacağımız veri seti Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır. Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır. Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır. 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir. Salary yani maaş değişkeninini bu projede linear regression ile tahmin edeceğiz.

Veri setini daha yakından tanımak adına değişkenleri tanıyalım:

AtBat: 1986–1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı 

Hits: 1986–1987 sezonundaki isabet sayısı 

HmRun: 1986–1987 sezonundaki en değerli vuruş sayısı 

Runs: 1986–1987 sezonunda takımına kazandırdığı sayı 

RBI: Bir vurucunun vuruş yaptığında koşu yaptırdığı oyuncu sayısı 

Walks: Karşı oyuncuya yaptırılan hata sayısı 

Years: Oyuncunun major liginde oynama süresi (sene) 

CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı 

CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı 

CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli vuruş sayısı 

CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı 

CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı 

CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı 

League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör 

Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör 

PutOuts: Oyun icinde takım arkadaşınla yardımlaşma 

Assits: 1986–1987 sezonunda oyuncunun yaptığı asist sayısı 

Errors: 1986–1987 sezonundaki oyuncunun hata sayısı 

Salary: Oyuncunun 1986–1987 sezonunda aldığı maaş(bin uzerinden) 

NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

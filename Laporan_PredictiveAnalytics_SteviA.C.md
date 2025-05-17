# Laporan Proyek Machine Learning - Stevi Aprilianti Cahyani

## Domain Proyek

**Ford Motor Company** (umumnya dikenal sebagai **Ford**) adalah produsen mobil multinasional asal Amerika Serikat yang berkantor pusat di **Dearborn, Michigan**. Perusahaan ini didirikan oleh **Henry Ford** pada tanggal **16 Juni 1903**, dan telah menjadi salah satu pelopor dalam industri otomotif global. Ford menjual mobil dan kendaraan komersial dengan merek **Ford**, serta mobil mewah dengan merek **Lincoln**. Perusahaan ini juga terdaftar di Bursa Efek New York dengan simbol ticker satu huruf “F” dan dikendalikan oleh **keluarga Ford**. Meskipun hanya memiliki kepemilikan minoritas, keluarga Ford tetap memiliki hak suara mayoritas dalam pengambilan keputusan perusahaan \[1].

Sebagai produsen otomotif besar, Ford telah mengalami berbagai pasang surut dalam sejarah perusahaannya, termasuk masuk ke pasar Indonesia pada tahun 2000, sempat menghentikan operasinya pada 2016, dan kembali beroperasi di Indonesia sejak 2020.

Namun, di tengah pencapaiannya, Ford juga menghadapi tantangan serius dalam hal **kualitas produk**. Pada tahun 2023, Ford melakukan dua kali penarikan kembali (*recall*) terhadap lebih dari **148.000 unit kendaraan** akibat **kebocoran cairan rem** dan **gangguan modul kontrol powertrain**. Masalah ini berdampak pada model populer seperti **Ford F-150**, **Expedition**, **Lincoln Navigator**, dan **Ford Explorer**. Recall ini dilakukan karena potensi penurunan performa pengereman serta risiko mesin mati saat berkendara, yang membahayakan keselamatan pengguna \[2].

![Ford Car](https://thebrandhopper.com/wp-content/uploads/2023/03/ford-history.jpeg)
**Gambar 1. Ford Car**

Kejadian tersebut menegaskan bahwa **kualitas produk** sangat memengaruhi kepercayaan pelanggan, keselamatan pengguna, dan reputasi perusahaan. Di samping itu, **persaingan bisnis otomotif global** yang semakin ketat juga menuntut Ford untuk terus berinovasi, termasuk dalam **menentukan harga kendaraan yang kompetitif namun sebanding dengan kualitas**. Ford perlu mampu memprediksi harga produknya dengan tepat, tidak hanya berdasarkan permintaan pasar, tetapi juga dari spesifikasi teknis, fitur, performa, serta faktor historis lainnya.



## Business Understanding

### Problem Statements

1. Bagaimana cara menentukan model *machine learning* terbaik untuk memprediksi harga dari mobil Ford secara akurat dan efisien?

2. Algoritma *machine learning apa yang memberikan performa terbaik dalam hal error terkecil dan akurasi tertinggi ketika diterapkan pada data harga mobil Ford?

3. Bagaimana hasil evaluasi model dapat digunakan untuk mendukung pengambilan keputusan bisnis Ford, khususnya dalam strategi penetapan harga produk otomotif?
   

### Goals

1. Menghasilkan model *machine learning* yang mampu memprediksi harga mobil Ford secara akurat.
   Tujuan ini untuk memastikan bahwa perusahaan memiliki alat prediksi harga yang andal berdasarkan fitur-fitur kendaraan dan data historis penjualan.

2. Membandingkan performa beberapa algoritma *machine learning* menggunakan metrik evaluasi seperti Mean Squared Error (MSE) dan akurasi prediksi. Hal ini bertujuan untuk mengetahui model mana yang paling unggul dalam meminimalkan kesalahan dan memberikan prediksi harga terdekat dengan harga sebenarnya.

3. Memberikan insight kepada manajemen dalam proses penetapan harga yang lebih berbasis data (data-driven decision making).

### Solution Statements

1. Menerapkan dan membandingkan beberapa model *machine learning* populer seperti K-Nearest Neighbor, Random Forest, dan Boosting Algorithm.

2. Melakukan hyperparameter tuning pada model untuk meningkatkan performa prediksi.
Optimalisasi parameter seperti nilai k pada KNN, jumlah pohon pada Random Forest, dan learning rate pada Boosting dilakukan agar model tidak hanya cepat, tetapi juga akurat.

3. Menggunakan visualisasi dan evaluasi hasil prediksi untuk mengkomunikasikan model terbaik kepada tim non-teknis.
   Visualisasi seperti grafik MSE dan akurasi model akan membantu pihak manajemen memahami hasil dan manfaat dari penerapan *machine learning*.
   
   
## Data Understanding & Cleaning

Dataset yang digunakan dalam proyek ini adalah **Ford Car Price Prediction**, yang dapat diakses melalui [**Kaggle**](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction). Data tersebut tersedia dalam format **CSV** dengan nama file `ford.csv`.

Selanjutnya, dilakukan proses **Exploratory Data Analysis (EDA)** sebagai tahap awal untuk memahami data lebih dalam. EDA bertujuan untuk mengidentifikasi karakteristik data, menemukan pola atau tren, mendeteksi anomali, serta memverifikasi asumsi-asumsi awal terhadap data.

Setelah melalui proses EDA, diperoleh informasi bahwa dataset ini terdiri dari **17.966 baris data** yang merepresentasikan riwayat harga mobil **Ford**, dengan **9 kolom atau variabel** yang mencakup informasi sebagai berikut :

    - `model` :  Jenis mobil Ford (misalnya Fiesta, Focus, dll).
    - `year` : Tahun pembuatan mobil.
    - `price` : Harga mobil dalam satuan mata uang tertentu.
    - `transmission` : Tipe transmisi mobil (Manual, Automatic, dll).
    - `mileage` : Jarak tempuh mobil dalam satuan kilometer atau mil.
    - `fuelType` : Jenis bahan bakar yang digunakan (Bensin, Diesel, Hybrid, dll).
    - `tax` : Besarnya pajak tahunan kendaraan.
    - `mpg` : Konsumsi bahan bakar, menunjukkan efisiensi penggunaan bensin.
    - `engineSize` : Kapasitas mesin mobil dalam liter.

| **No.** | **Nama Kolom** | **Jumlah Data (Non-Null)** | **Tipe Data** |
| ------: | -------------- | -------------------------: | ------------- |
|       1 | `model`        |                     17.966 | object        |
|       2 | `year`         |                     17.966 | int64         |
|       3 | `price`        |                     17.966 | int64         |
|       4 | `transmission` |                     17.966 | object        |
|       5 | `mileage`      |                     17.966 | int64         |
|       6 | `fuelType`     |                     17.966 | object        |
|       7 | `tax`          |                     17.966 | int64         |
|       8 | `mpg`          |                     17.966 | float64       |
|       9 | `engineSize`   |                     17.966 | float64       |

**Tabel 1. Tipe Data**

Dari tabel di atas, dapat diketahui bahwa terdapat **6 data numerik** yakni: `year`, `price`, `mileage`, `tax` dengan tipe data *int64* dan `mpg`, `engineSize` dengan tipe data *float64*. Lalu, **3 data kategori** yakni: `model`, `transmission`, dan `fuelType` dengan tipe data *object*.

   
  - **Ringkasan Statistik**
          | **Statistik**    | **year** | **price** | **mileage** | **tax** | **mpg** | **engineSize** |
      | ---------------- | -------- | --------- | ----------- | ------- | ------: | -------------- |
      | **count**        | 17,966   | 17,966    | 17,966      | 17,966  |  17,966 | 17,966         |
      | **mean**         | 2016.87  | 12,279.53 | 23,362.61   | 113.33  |   57.91 | 1.35           |
      | **std**          | 2.05     | 4,741.34  | 19,472.05   | 62.01   |   10.13 | 0.43           |
      | **min**          | 1996     | 495       | 1           | 0       |   20.80 | 1.00           |
      | **25%**          | 2016     | 8,999     | 9,987       | 30      |   52.30 | 1.00           |
      | **50% (median)** | 2017     | 11,291    | 18,242.5    | 145     |   58.90 | 1.00           |
      | **75%**          | 2018     | 15,299    | 31,060      | 145     |   65.70 | 1.50           |
      | **max**          | 2060     | 54,995    | 177,644     | 580     |  201.80 | 5.00           |

    **Tabel 2. Deskripsi statistik**

     Berdasarkan tabel hasil ringkasan statistik, dapat diketahui bahwa dataset ini berisi 17.966 data mobil Ford. Rata- 
     rata tahun mobil adalah 2016, dengan tahun tertua 1996 dan tertinggi tercatat 2060 (kemungkinan data tidak valid). 
     Harga mobil rata-rata sekitar £12.280, dengan harga termurah £495 dan termahal £54.995. Jarak tempuh (mileage) rata- 
     rata adalah 23.362 mil, dengan maksimal hingga 177.644 mil.

      Besaran pajak (tax) bervariasi, dengan nilai rata-rata £113 dan maksimal £580. Konsumsi bahan bakar (mpg) rata-rata 
      57.9 mpg, menunjukkan banyak mobil hemat bahan bakar, namun ada nilai ekstrim hingga 201.8 mpg (kemungkinan error). 
      Ukuran mesin (engineSize) rata-rata 1.35L, dengan rentang dari 0.0L hingga 5.0L, menunjukkan adanya data yang 
      mungkin perlu dibersihkan atau diperiksa lebih lanjut.

        
- **Memeriksa dan Menangani Missing Value**
  
   Dalam dataset **tidak ditemukan missing value**, artinya semua kolom memiliki jumlah data yang lengkap tanpa ada nilai kosong. Namun, untuk memastikan kualitas data, perlu dilakukan pengecekan lebih lanjut terhadap **keberadaan nilai nol (0)** pada kolom-kolom numerik tertentu.
   
   Berdasarkan ringkasan statistik deskriptif, ditemukan nilai minimum **0.000000** pada fitur **`engineSize`**, yang secara logika tampak janggal. Hal ini karena setiap mobil seharusnya memiliki kapasitas mesin lebih dari nol; maka kemungkinan besar nilai 0 pada kolom ini bukan representasi yang valid.
   
   Setelah dilakukan pemeriksaan, ternyata terdapat **51 baris data** yang memiliki nilai **0 pada kolom `engineSize`**. Karena nilai tersebut tidak masuk akal dan berpotensi menurunkan kualitas analisis, maka langkah yang diambil adalah **menghapus seluruh baris tersebut**.
   
   Setelah proses pembersihan ini, jumlah data pada dataset **berkurang dari 17.966 menjadi 17.915 baris**, sementara jumlah kolom tetap sebanyak 9.
   
   Langkah ini bertujuan untuk memastikan bahwa data yang digunakan dalam proses analisis maupun pemodelan hanya terdiri dari informasi yang valid dan relevan, khususnya dalam hal ukuran mesin mobil. Dengan begitu, hasil analisis akan menjadi lebih akurat, terpercaya, dan representatif terhadap kondisi nyata.
  ![image](https://github.com/user-attachments/assets/d25aedd3-4dcb-4560-b018-e022462755b7)

- **Memeriksa dan Menangani Outliers**
  
   Pada tahapan ini dilakukan pemeriksaan dan penanganan outliers agar data lebih bersih dan representatif. Pemeriksaan outliers dilakukan dengan visualisasi boxplot dan perhitungan IQR untuk mengidentifikasi nilai-nilai yang ekstrem dan tidak wajar.
  
     ![image](https://github.com/user-attachments/assets/3f28b818-adcd-4494-9ada-6012f2631b2c)

   Boxplot menunjukkan adanya outlier di hampir semua fitur numerik dalam dataset mobil Ford. Outlier paling mencolok terlihat pada kolom mileage (jarak tempuh) dan price (harga mobil), di mana banyak nilai berada jauh di atas batas atas (upper whisker). Kolom year juga menunjukkan beberapa data tahun yang tidak wajar (misalnya tahun 2060), sedangkan engineSize memiliki nilai nol yang tampaknya tidak valid. Kolom tax dan mpg memiliki sedikit outlier, tapi masih dalam batas yang lebih terkendali. Secara keseluruhan, boxplot ini menunjukkan perlunya pembersihan data, terutama pada outlier yang ekstrem dan nilai yang tidak realistis.

  Selanjutnya, dilakukan pembersihan outlier menggunakan metode IQR pada seluruh kolom numerik dalam dataset. Proses ini dimulai dengan menghitung kuartil pertama (Q1), kuartil ketiga (Q3), dan rentang interkuartil (IQR). Data yang memiliki nilai di luar batas bawah (Q1 - 1.5 × IQR) dan batas atas (Q3 + 1.5 × IQR) dianggap sebagai outlier dan dihapus dari dataset. Langkah ini efektif dalam mengurangi nilai-nilai ekstrem yang berpotensi mengganggu hasil analisis, sehingga jumlah data tersisa menjadi 16.450 baris dan 9 kolom. Perubahan distribusi data setelah pembersihan dapat dilihat pada boxplot berikut:
    ![image](https://github.com/user-attachments/assets/e8c1fad8-e7e0-4ccd-b80d-a56e42af70cb)

  Boxplot menunjukkan bahwa data kini jauh lebih bersih dibanding sebelumnya. Nilai-nilai ekstrem yang sebelumnya mendominasi kolom price, mileage, dan year telah berhasil dikurangi. Distribusi data menjadi lebih seimbang, dan mayoritas nilai berada dalam rentang normal.
  
- **Univariate Analysis**

   Univariate analysis dilakukan untuk memahami karakteristik dari setiap fitur secara individual. Dalam dataset, fitur 
   dibagi menjadi dua tipe utama numerik dan kategorikal. Proses analisisnya pun berbeda tergantung pada tipe data.
   
   - **Categorical Features**
     
     - **Analisis Fitur Categorical - Model**
    
       ![image](https://github.com/user-attachments/assets/6c24a75f-2744-40cd-9521-878c7d5016cf)

       Distribusi data pada fitur model menunjukkan dominasi yang kuat oleh dua jenis mobil, yaitu Ford 
       Fiesta dan Ford Focus. Fiesta merupakan model yang paling sering muncul dengan total sekitar 6.200 
       sampel, diikuti oleh Focus dengan lebih dari 4.100 sampel. Setelah itu, jumlahnya menurun cukup 
       tajam. Kuga menempati posisi ketiga dengan sekitar 2.000 sampel, sementara EcoSport berada di 
       kisaran 1.100. Model-model lainnya seperti Ka+, C-MAX, Mondeo, hingga Puma hanya menyumbang ratusan 
       hingga puluhan sampel. Beberapa model seperti varian Tourneo dan satu entri tambahan Focus bahkan 
       hanya muncul kurang dari 50 kali. Secara keseluruhan, data ini menunjukkan bahwa hanya segelintir 
       model yang mendominasi, sedangkan sebagian besar lainnya memiliki jumlah kemunculan yang relatif 
       kecil.
       
     - **Analisis fitur categorical - Transmission**
       
       ![image](https://github.com/user-attachments/assets/a05d51fb-7b11-4058-8886-278057e99524)

       Fitur *transmission* memperlihatkan ketimpangan yang mencolok: mobil bertransmisi manual mencapai 14 
       305 sampel, sedangkan transmisi automatic dan semi-auto hanya 1.149 dan 996 sampel. Artinya, lebih 
       dari 90 % data dikuasai oleh mobil manual, sementara dua kategori lainnya bersama-sama menyumbang 
       kurang dari 10 %.

     - **Analisis fitur categorical - Fuel Type**
       
       ![image](https://github.com/user-attachments/assets/88906891-2e96-4a23-94ab-606d643a40c6)

       Distribusi pada fitur *fuelType* menunjukkan fokus pasar yang masih sangat konvensional. Mobil 
       bensin mendominasi dengan jumlah 11 500 sampel, diikuti diesel dengan 4962 sampel. Kategori lain 
       seperti hybrid hanya terdapat 8 unit, listrik 2 unit, dan “other” hanya satu. Dengan kata lain,    
       lebih dari dua-pertiga data diisi kendaraan berbahan bakar bensin, sepertiga sisanya    
       diesel,sementara teknologi alternatif baru muncul dengan jumlah yang sangat sedikit.
       
   - **Numerical Features**
     
     ![image](https://github.com/user-attachments/assets/634c4efa-32be-447d-9489-eed7c466e517)

     Distribusi fitur numerikal pada Ford Car Dataset menunjukkan bahwa sebagian besar mobil merupakan keluaran tahun 2017 
     hingga 2019. Harga mobil cenderung berkonsentrasi pada kisaran 7.500 hingga 15.000. Untuk fitur jarak tempuh 
     (mileage), mayoritas mobil memiliki jarak tempuh di bawah 20.000. Nilai pajak kendaraan paling umum berada pada angka 
     150. Pada fitur efisiensi bahan bakar (mpg), mobil umumnya memiliki nilai antara 50 hingga 70. Sementara itu, ukuran 
     mesin (engine size) yang paling sering ditemukan adalah 1.0L dan 1.5L.
   
- **Multivariate Analysis**
  
    Multivariate Analysis adalah proses analisis data yang melibatkan lebih dari satu variabel sekaligus, untuk melihat 
    hubungan, pola, atau pengaruh antar fitur dalam dataset. Berbeda dengan univariate analysis yang hanya fokus pada satu 
    variabel, analisis multivariat bertujuan untuk menggali keterkaitan antar fitur baik yang numerik maupun kategorikal.
    Proses multivariate data analysis pada masing-masing fitur kategorial dan numerik.
  
   - Categorical Features
     
     ![image](https://github.com/user-attachments/assets/8629d4c6-3a7e-49e7-96e5-43b0b005e4bd)

     Analisis multivariat terhadap rata-rata harga berdasarkan model mobil menunjukkan adanya variasi harga yang cukup 
     signifikan antar model. Model Puma, Edge, dan S-MAX memiliki rata-rata harga tertinggi dibandingkan model lainnya, 
     yang menunjukkan bahwa mobil-mobil ini mungkin termasuk dalam kategori mobil keluarga besar atau SUV dengan fitur 
     lebih lengkap. Sebaliknya, model seperti KA, B-MAX, dan Ka+ memiliki rata-rata harga yang paling rendah, menandakan 
     bahwa model ini kemungkinan merupakan jenis mobil kecil dengan spesifikasi standar. Model lain seperti Focus, Kuga, 
     dan Mondeo memiliki harga rata-rata di tingkat menengah, menunjukkan keseimbangan antara fitur dan harga. Perbedaan 
     ini mencerminkan segmentasi pasar dari masing-masing model Ford, mulai dari mobil ekonomis hingga premium.

     ![image](https://github.com/user-attachments/assets/2c4fe8f4-1e67-4c55-9485-865e2c48018f)

     Grafik menunjukkan bahwa mobil dengan transmisi Automatic memiliki harga rata-rata tertinggi, disusul oleh transmisi 
     Semi-Auto, dan terakhir Manual. Perbedaan ini mencerminkan tren pasar di mana mobil dengan transmisi otomatis 
     cenderung memiliki fitur yang lebih canggih dan kenyamanan lebih tinggi, sehingga dihargai lebih mahal. Sementara 
     itu, mobil bertransmisi manual umumnya lebih sederhana dan ekonomis, sehingga memiliki rata-rata harga terendah. 
     Rentang kesalahan (error bar) yang kecil pada ketiga kelompok menunjukkan bahwa rata-rata ini cukup representatif 
     terhadap sebaran data.
     
     ![image](https://github.com/user-attachments/assets/38cfad01-f1f4-4e6b-9c13-a94ff38dacd2)
     
     Grafik menunjukkan bahwa mobil dengan jenis bahan bakar Hybrid memiliki rata-rata harga tertinggi dibandingkan jenis 
     lainnya. Namun, nilai error bar yang besar menunjukkan adanya variasi harga yang signifikan dalam kelompok ini. Mobil 
     Electric menempati posisi kedua dengan harga rata-rata yang juga tinggi namun lebih stabil. Selanjutnya, mobil Diesel 
     dan Other berada di kisaran menengah, sedangkan Petrol merupakan yang paling terjangkau. Hasil ini mencerminkan tren 
     industri otomotif, di mana kendaraan ramah lingkungan seperti hybrid dan listrik memiliki harga lebih tinggi karena 
     teknologi yang digunakan.


   - Numerical Features
     - Memeriksa Korelasi antara Fitur Numerik dengan Fitur Target menggunakan Fungsi corr()
       ![image](https://github.com/user-attachments/assets/d1a3e5e4-2abf-462d-8c09-13e5546ce714)

       Pairplot digunakan untuk mengeksplorasi hubungan antar fitur numerik dengan cara memvisualisasikan scatter plot 
       untuk setiap pasangan fitur, serta distribusi dari masing-masing fitur di diagonal.


         1. **year vs price**
            Terlihat **korelasi positif**: mobil yang lebih baru (tahun lebih tinggi) cenderung memiliki **harga lebih 
            mahal**.
         
         2. **mileage vs price**
            Terdapat **korelasi negatif**: semakin besar mileage (jarak tempuh), **harga mobil menurun**. Ini logis karena 
            mobil dengan jarak tempuh tinggi biasanya lebih murah.
         
         3. **tax dan mpg**
         
            * Nilai **tax** cenderung bervariasi dalam kelompok (diskrit), terlihat seperti garis horizontal.
            * **mpg (miles per gallon)** memiliki sebaran yang cukup lebar, dan tampak hubungan **negatif terhadap 
            engineSize**, di mana mobil dengan mesin lebih besar cenderung memiliki efisiensi bahan bakar yang lebih 
             rendah.
         
         4. **engineSize vs price**
            Cenderung **korelasi positif**: mobil dengan mesin lebih besar biasanya lebih mahal.
         
         5. **Distribusi** (diagonal)
         
            * Fitur seperti `year`, `tax`, `engineSize` bersifat **diskrit** (nilai-nilai terbatas).
            * `price` dan `mileage` memiliki distribusi **right-skewed**, banyak data di harga/jarak rendah dan sedikit di 
            harga/jarak tinggi.

          6. **Hubungan antar fitur lainnya**

            
             * Korelasi antara `mpg` dan `engineSize` juga terlihat cukup kuat secara visual: makin besar mesin, makin boros bahan bakar.
             * `year` dan `mileage` menunjukkan **korelasi negatif**, mobil yang lebih baru cenderung memiliki mileage yang 
          lebih rendah.

      
      - Memeriksa Kolerasi Fitur Numerik Menggunakan Heatmap
        ![image](https://github.com/user-attachments/assets/3b9187a8-8b0d-44c9-a561-0edcd6169b50)
         Heatmap digunakan untuk menunjukkan **hubungan linear antar fitur numerik** dalam bentuk **koefisien korelasi Pearson**:


         1. **Korelasi Positif Tinggi**
         
            * `year` dengan `price` (**0.64**): Mobil yang lebih baru cenderung lebih mahal.
            * `tax` dengan `price` (**0.49**) dan `year` (\*\*0.52\`): Pajak kendaraan cenderung lebih tinggi untuk mobil yang baru dan mahal.
            * `engineSize` dengan `price` (**0.40**): Mesin lebih besar umumnya terdapat pada mobil yang lebih mahal.
         
         2. **Korelasi Negatif Kuat**
         
            * `mileage` dengan `year` (**-0.67**) dan `price` (**-0.48**): Mobil yang lebih tua cenderung memiliki jarak tempuh lebih tinggi dan harga yang lebih rendah.
            * `mpg` dengan `tax` (**-0.48**): Semakin irit mobil (mpg tinggi), biasanya semakin rendah pajaknya.
         
         3. **Korelasi Lemah atau Hampir Tidak Ada**
         
            * `engineSize` memiliki korelasi rendah terhadap sebagian besar fitur lainnya kecuali `price`.
            * `mpg` punya korelasi kecil negatif terhadap `year` dan `engineSize`.
              
   Karena di beberapa fitur terdapat korelasi yang rendah, maka dilakukan eliminasi terhadap beberapa fitur tersebut yaitu mileage dan mpg, sehingga hanya menyisakan 7 fitur lainnya.


## Data Preparation

Tahapan data preparation ini sangat penting untuk memastikan kualitas input data sebelum digunakan untuk pemodelan, agar hasil prediksi lebih akurat dan efisien. 

- **Encoding Fitur Kategorikal**

   Fitur kategorikal seperti `fuelType`, `transmission`, dan `model` tidak bisa langsung digunakan oleh algoritma ML, sehingga dilakukan encoding. Berikut merupakan hasil encoding

- **Reduksi Dimensi dengan PCA (Principal Component Analysis)**

   Principal Component Analysis (PCA) merupakan metode yang digunakan untuk mengurangi jumlah fitur dalam data, sambil tetap mempertahankan informasi penting di dalamnya. Dalam proses ini, data ditransformasikan dari ruang berdimensi n ke ruang berdimensi baru yang lebih rendah (m), di mana nilai m lebih kecil dari n. Tujuan utamanya adalah untuk menyederhanakan kompleksitas data, mengekstrak fitur utama, serta merepresentasikan data dalam sistem koordinat baru yang lebih ringkas namun tetap informatif. Hasil proporsi informasi dari fitur engineSize dan Tax dengan menggunakan Principal Component Analysis (PCA), yaitu
   `array([1., 0.])`
   
- **Membagi Data Latih & Data Uji dengan Train Test Split**
  
   Pada tahap ini, dataset dipisahkan menjadi dua bagian utama:
   
   * **Fitur (X):** Seluruh kolom selain `price`, karena `price` adalah nilai yang ingin diprediksi.
   * **Target (y):** Kolom `price`, yang berfungsi sebagai variabel target dalam pemodelan.
   
   Setelah itu, dilakukan pembagian data menjadi **data latih (train)** dan **data uji (test)** menggunakan fungsi `train_test_split`. Proporsi data yang digunakan yaitu 80% untuk pelatihan model dan 20% untuk pengujian model, dengan `random_state=69` untuk memastikan hasil pembagian yang konsisten.
   
   Langkah ini penting agar model dapat dilatih dengan sebagian data, lalu dievaluasi menggunakan data yang belum pernah dilihat sebelumnya untuk mengukur performa dan kemampuannya dalam melakukan generalisasi.

- **Standarisasi**

   Setelah proses pembagian data, dilakukan standarisasi pada fitur numerik agar berada dalam skala yang seragam. Fitur yang distandarisasi pada tahap ini adalah `year` dan `feature`.
   
   Langkah-langkahnya meliputi:
   
   * **Inisialisasi `StandardScaler`**, yaitu teknik dari `sklearn` yang mengubah distribusi data agar memiliki rata-rata (mean) 0 dan standar deviasi (std) 1.
   * **Melatih scaler hanya pada data latih**, untuk mencegah kebocoran data dari data uji.
   * **Menerapkan transformasi pada data latih** sehingga nilai-nilai dalam kolom numerik menjadi setara secara skala.
   
   Standarisasi penting dilakukan terutama saat menggunakan algoritma machine learning yang sensitif terhadap skala fitur seperti KNN, SVM, atau PCA. Dengan fitur yang terstandarisasi, model dapat belajar lebih efisien dan menghasilkan prediksi yang lebih akurat.


## Model Development

Sebelum membangun model prediksi, dilakukan tahap persiapan *dataframe* guna mendukung proses analisis model dengan menggunakan tiga algoritma utama, yaitu `K-Nearest Neighbor (KNN)`, `Random Forest`, dan `Boosting Algorithm`.

* Menyiapkan Dataframe

   ```python
   # Membuat DataFrame kosong untuk menyimpan nilai MSE (Mean Squared Error)
   # dari masing-masing model pada data train dan test
   models = pd.DataFrame(
       index=['train_mse', 'test_mse'],
       columns=['KNN', 'RandomForest', 'Boosting']
   )
   ```
  
* **Algoritma K-Nearest Neighbor (KNN)**

  KNN memprediksi nilai dari data baru berdasarkan kesamaan fitur dengan data yang sudah ada. Algoritma ini bekerja dengan cara menghitung jarak antara data baru dengan data pelatihan, lalu memilih sejumlah `k` tetangga terdekat (di mana `k` adalah bilangan positif). Dalam kasus ini, KNN menggunakan parameter `n-neighbors` sebesar 10.

  ```python
  # Inisialisasi model K-Nearest Neighbors Regressor dengan 10 tetangga terdekat
  KNN_model = KNeighborsRegressor(n_neighbors=10)

  # Melatih model KNN dengan data pelatihan
  KNN_model.fit(x_train, y_train)

  # Memprediksi harga mobil pada data pelatihan menggunakan model yang sudah dilatih
  y_pred_KNN_model = KNN_model.predict(x_train)
  ```

  * **Kelebihan**:
    KNN mudah dipahami dan diimplementasikan, serta cocok untuk berbagai permasalahan seperti klasifikasi, regresi, dan pencarian data serupa.

  * **Kekurangan**:
    Seiring bertambahnya jumlah data atau variabel, proses komputasi menjadi lambat karena KNN harus menghitung jarak terhadap seluruh data pelatihan.

* **Algoritma Random Forest**

  Random Forest adalah metode *supervised learning* yang digunakan untuk tugas klasifikasi maupun regresi. Algoritma ini termasuk dalam kategori *ensemble learning*, yaitu menggabungkan beberapa model untuk menghasilkan prediksi yang lebih kuat. Model ini dikonfigurasi dengan parameter `n-estimators` sebanyak 45 pohon, `max-depth` sebesar 16, `random-state` bernilai 69, serta `n-jobs` disetel ke -1 agar pemrosesan berjalan secara paralel.

  ```python
  # Inisialisasi model Random Forest Regressor
  RF_model = RandomForestRegressor(
      n_estimators=45,
      max_depth=16,
      random_state=69,
      n_jobs=-1
  )

  # Melatih model dengan data pelatihan
  RF_model.fit(x_train, y_train)

  # Menghitung Mean Squared Error (MSE) pada data pelatihan dan simpan ke dalam tabel `models`
  models.loc['train_mse', 'RandomForest'] = mean_squared_error(
      y_pred=RF_model.predict(x_train),
      y_true=y_train
  )
  ```

  * **Kelebihan**:
    Random Forest mampu memberikan akurasi prediksi yang tinggi, dapat memproses dataset besar, menangani banyak fitur tanpa perlu seleksi fitur, dan memiliki kemampuan mengestimasi pentingnya variabel serta mengatasi data yang hilang.

  * **Kekurangan**:
    Model ini rentan terhadap *overfitting* pada dataset yang mengandung noise tinggi. Selain itu, pada data kategorikal dengan banyak level, algoritma ini cenderung berpihak pada variabel dengan level terbanyak sehingga nilai *feature importance*-nya bisa menyesatkan.

* **Boosting Algorithm**

  Boosting adalah metode yang meningkatkan akurasi model dengan menggabungkan beberapa model lemah (*weak learners*) menjadi satu model yang kuat (*strong learner*). Pada implementasinya, digunakan parameter `n_estimators` sebanyak 50, `learning_rate` sebesar 0.05 untuk mengatur bobot tiap regressor, dan `random_state` disetel ke 69.

  ```python
  # Inisialisasi model AdaBoost Regressor
  BA_model = AdaBoostRegressor(
      n_estimators=50,
      learning_rate=0.05,
      random_state=69
  )

  # Melatih model dengan data pelatihan
  BA_model.fit(x_train, y_train)

  # Menghitung Mean Squared Error (MSE) pada data pelatihan dan simpan ke dalam tabel `models`
  models.loc['train_mse', 'Boosting'] = mean_squared_error(
      y_pred=BA_model.predict(x_train),
      y_true=y_train
  )  
  ```

  * **Kelebihan**:
    Boosting efektif dalam mengurangi bias dan cukup sederhana untuk diimplementasikan. Algoritma ini seringkali memberikan hasil lebih baik dibandingkan model sederhana seperti regresi logistik atau random forest.

  * **Kekurangan**:
    Model Boosting, terutama AdaBoost, sangat sensitif terhadap outlier atau data yang menyimpang jauh.

Setelah model K-Nearest Neighbor, Random Forest, dan Boosting diterapkan, dilakukan pengujian performa masing-masing model. Model terbaik akan dipilih berdasarkan nilai error terkecil, akurasi tertinggi, dan kemampuan prediksi yang paling mendekati nilai aktual.

## Evaluasi Model

Pada tahap evaluasi, dilakukan serangkaian proses untuk mengukur kinerja model yang telah dibangun. Proses ini dibagi ke dalam beberapa langkah sebagai berikut:

- Mengukur Nilai Error Menggunakan MSE

   Sebelum melakukan evaluasi, terlebih dahulu dilakukan transformasi fitur numerik pada data uji menggunakan **scaler** yang sebelumnya telah di-*fit* pada data latih. Hal ini bertujuan untuk menyamakan skala antara data latih dan data uji.
   
   Selanjutnya, dilakukan perhitungan **Mean Squared Error (MSE)** terhadap data latih dan data uji untuk masing-masing model yang digunakan, yaitu K-Nearest Neighbor (KNN), Random Forest (RF), dan Boosting. Nilai MSE digunakan untuk mengetahui seberapa besar kesalahan prediksi model terhadap nilai sebenarnya. Nilai MSE dibagi 1000 untuk mempermudah interpretasi dalam skala yang lebih kecil, dan hasilnya disimpan dalam bentuk **DataFrame**.

- Visualisasi Perbandingan Performa Model

   Setelah mendapatkan nilai MSE dari masing-masing model pada data latih dan uji, hasil tersebut divisualisasikan agar dapat dilihat secara jelas perbandingan performanya. Visualisasi ini membantu dalam memahami model mana yang memberikan kesalahan prediksi paling rendah.
   ![image](https://github.com/user-attachments/assets/6407dd83-f9c2-40a9-8c70-2da3f1969641)


- Mengevaluasi Akurasi Model

   Selain menggunakan MSE, evaluasi performa model juga dilakukan dengan mengukur **akurasi** menggunakan fungsi `.score()` dari masing-masing model terhadap data uji. Nilai akurasi dikalikan 100 untuk mengubahnya ke dalam bentuk persentase. Hasil akurasi ini kemudian dikompilasi ke dalam sebuah **DataFrame** evaluasi, sehingga dapat dibandingkan secara langsung.
   
   Berikut hasil akurasi dari ketiga model:
   
   | Model              | Accuracy (%) |
   | ------------------ | ------------ |
   | K-Nearest Neighbor | 88.09        |
   | Random Forest      | 89.73        |
   | Boosting Algorithm | 58.50        |
   
   Dari hasil tersebut, dapat disimpulkan bahwa model **Random Forest** memberikan akurasi prediksi terbaik pada data uji.

- Melakukan Prediksi dengan Model

   Langkah terakhir dalam evaluasi ini adalah melakukan prediksi terhadap satu data uji menggunakan ketiga model. Data uji diambil dari baris pertama, dan hasil prediksi dibandingkan dengan nilai sebenarnya. Hasil prediksi masing-masing model ditampilkan dalam bentuk **DataFrame** sebagai berikut:
   
   | y\_true | prediksi\_KNN | prediksi\_RF | prediksi\_Boosting |
   | ------- | ------------- | ------------ | ------------------ |
   | 9998    | 9873.8        | 9903.7       | 11707.4            |
   
   Berdasarkan hasil tersebut, dapat dilihat bahwa model **KNN** dan **Random Forest** memiliki prediksi yang cukup dekat dengan nilai sebenarnya dibandingkan model Boosting yang cenderung meleset lebih jauh.

### **Kesimpulan**

Berdasarkan hasil analisis dan evaluasi terhadap tiga algoritma machine learning yang diterapkan, yaitu **K-Nearest Neighbor (KNN)**, **Random Forest (RF)**, dan **Boosting Algorithm**, diperoleh beberapa poin penting sebagai berikut:

1. **Dari sisi error (Mean Squared Error/MSE):**

   * Model **Random Forest** menghasilkan nilai MSE paling rendah baik pada data latih (1435.29) maupun data uji (1628.64), menandakan bahwa model ini memiliki performa prediksi yang paling akurat dan stabil.
   * Model **KNN** menunjukkan performa yang cukup baik dengan nilai MSE sedikit lebih tinggi dari Random Forest.
   * Sementara itu, **Boosting Algorithm** mencatatkan MSE yang paling tinggi, mengindikasikan bahwa model ini kurang cocok untuk digunakan dalam kasus ini.

2. **Dari sisi akurasi (%):**

   * **Random Forest** mencatatkan akurasi tertinggi sebesar **89.73%** pada data uji.
   * Disusul oleh **KNN** dengan akurasi **88.09%**.
   * Sedangkan **Boosting Algorithm** memiliki akurasi yang jauh lebih rendah, yaitu **58.50%**, menunjukkan performa prediksi yang kurang baik.

3. **Dari hasil prediksi terhadap data uji:**

   * Model **KNN** dan **Random Forest** memberikan nilai prediksi yang paling mendekati nilai sebenarnya.
   * Model **Boosting** menghasilkan prediksi yang cukup jauh dari nilai aktual, sehingga kurang direkomendasikan untuk digunakan dalam konteks ini.

### **Kesimpulan Akhir:**

Dari keseluruhan proses modeling dan evaluasi, dapat disimpulkan bahwa **Random Forest** merupakan model terbaik yang paling cocok untuk digunakan pada dataset ini. Model ini memberikan keseimbangan yang baik antara **tingkat akurasi yang tinggi** dan **kesalahan prediksi yang rendah**, sehingga lebih andal dalam memprediksi nilai target dibandingkan dua model lainnya.

 



### Referensi

\[1] Ford Motor Company. *Wikipedia*. \[Online]. [https://id.wikipedia.org/wiki/Ford\_Motor\_Company](https://id.wikipedia.org/wiki/Ford_Motor_Company)

\[2] CT Insider, “Ford recalls over 148,000 vehicles due to brake fluid leaks and powertrain control issues,” *CT Insider*, 2023. \[Online]. [https://www-ctinsider-com.translate.goog/news/article/ford-car-recall-ford-f150-expedition-explorer-20282660.php](https://www-ctinsider-com.translate.goog/news/article/ford-car-recall-ford-f150-expedition-explorer-20282660.php)

---


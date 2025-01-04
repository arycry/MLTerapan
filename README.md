# Prediksi harga mobil bekas - Dita Ary Crystian

# Domain Projek

Perkembangan bisnis otomotif saat ini berkembang pesat. Perkembangan teknologi serta ketatnya persaingan antara perusahaan otomotif berpengaruh di produksi mobil baru yang pesat juga. Hal ini berimbas pada pasar mobil bekas. Pada pasar mobil bekas, harga cenderung turun secara periodik dan dipengaruhi oleh beberapa faktor, seperti tahun keluaran, merek mobil, penggunaan mobil, dan lain-lain. Seringkali pemilik mobil tidak mengetahui harga jual dari mobilnya yang berakibat kerugian dari pemulik mobil.  Dengan memperhatikan berbagai faktor tersebut, kita bisa membuat sistem prediksi yang membantu penjual dalam menentukan harga jual mobil bekas nantinya, sehingga transaksi berjalan lebih efektif dan efisien.  

Referensi: [Jurnal Prediksi harga mobil bekas](https://publikasiilmiah.unwahas.ac.id/JINRPL/article/view/10266/pdf).

# Business Understanding

## Problem Statement

Berdasarkan latar belakang yang dijelaskan di atas, kita mendapatkan rumusan masalah sebagai berikut:
1. Bagaimana menentukan harga jual yang tepat untuk mobil bekas dengan banyaknya faktor yang mempengaruhi?
2. Bagaimana membuat model yang baik untuk membuat prediksi harga jual mobil bekas?

## Goals
Berdasarkan rumusan masalah yang dijelaskan di atas, kita mendapatkan tujuan sebagai berikut:
1. Membuat model yang dapat memperkirakan harga jual mobil bekas dengan efektif dan efisien
2. Membuat model Machine learning untuk analisis prediksi harga jual mobil dengan error seminim mungkin

## Solution 
Berdasarkan tujuan di atas, ada beberapa solusi yang bisa dilakukan, yaitu:
1. Melakukan persiapan data supaya dapat dilatih menjadi model yang baik. 
2. Membuat model dengan 3 algoritma yang berbeda, yaitu *Random Forest Regressor*, *XGBoost*, dan *Gradient Boosting*.
3. Mengevaluasi model yang ada dengan MSE(*Mean Square Error*).

# Data Understanding
Dataset pada projek ini diambil dari kaggle, yaitu [Car Price Prediction](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset). Pada dataset ini berisi *file* dengan nama `cardekho` dengan ekstensi `csv` `(*Comma Separated Values*)`.

Berikut merupakan deskripsi variabel dari dataset *Car Price Prediction*:

![Screenshot 2025-01-04 203453](https://github.com/user-attachments/assets/20fb78d9-5cde-480a-a0b8-13e44d89ff3f)


Dataset ini memiliki 8128 baris data dengan 12 kolom variabel.

## Variabel
Berikut merupakan variabel-variabel yang ada di dataset Car Price Prediction:
- name               : Nama mobil (merek dan model).
- year               : Tahun pembuatan mobil.
- selling_price      : Harga jual mobil .
- km_driven          : Kilometer yang telah ditempuh mobil.
- fuel               : Jenis bahan bakar yang digunakan (misalnya, Petrol, Diesel, CNG).
- seller_type        : Jenis penjual (Individual atau Dealer).
- transmission       : Jenis transmisi mobil (Manual atau Automatic).
- owner              : Jumlah pemilik mobil sebelumnya.
- mileage(km/ltr/kg) : Jarak tempuh mobil per liter.
- max power          : Tenaga maksimum yang dapat dihasilkan oleh mobil.
- seats              : Banyaknya kursi yang tersedia di mobil.

Terlihat bahwa ada variabel-variabel yang memiliki tipe data yang berbeda dari yang kita inginkan, seperti 'max_power' dan 'seats' dan ada nama kolom yang bisa kita persingkat, yaitu 'mileage(km/ltr/kg)'.
## Missing Value
Setelah itu kita akan mengecek missing value pada data ini

![Screenshot 2025-01-04 183529](https://github.com/user-attachments/assets/0dae05a9-6998-480a-a697-e95f6e3d999c)

Kita melihat adanya missing value pada kolom 'mileage(km/ltr/kg)', 'engine, max_power', dan 'seats'.
## Duplicate data
Setelah itu, kita akan mengecek banyaknya data duplikat yang ada di data ini

![Screenshot 2025-01-04 183814](https://github.com/user-attachments/assets/e8474f61-ae55-43b7-b8f7-47b9a1102c6d)

terlihat bahwa data kita memiliki 1202 data duplikat.
## Outliers
Setelah itu, kita akan mengecek *outliers* dari fitur numerik di data ini

![3](https://github.com/user-attachments/assets/77219c39-1749-4d09-ace8-133d28de69bb)

terlihat bahwa semua fitur numerik memiliki *outliers*.

Berdasarkan yang kita dapat, dapat disimpulkan bahwa:
- Terdapat nilai null pada kolom mileage(km/ltr/kg), engine, max_power, dan seats
- Terdapat data duplikat sebanyak 1202 data
- Setiap fitur numerik memiliki nilai *Outlier*

# Data Preparation 
Data Preparation merupakan proses transformasi data mentah menjadi format yang bersih, terstruktur, dan siap untuk dianalisis atau dimodelkan. Data mentah seringkali tidak lengkap, inkonsisten, dan mengandung error, sehingga perlu dipersiapkan terlebih dahulu sebelum dapat digunakan secara efektif. Data preparation penting antara lain untuk  meningkatkan kualitas data, mempermudah analisis dan insight yang didapat, dan meningkatkan performa model machine learning. Tahap-tahap Data Preparation antara lain:
- Data Cleaning
- EDA
- Data Transformation 

## Data Cleaning
Pada langkah data cleaning ini, kita akan melakukan langkah-langkah sebagai berikut:
- Menghapus data yang memiliki missing value
- Menghapus data duplikat
- Mengganti tipe data pada kolom 'max power'
- Mengganti tipe data pada kolom 'seats'
- Mengganti nama kolom mileage(km/ltr/kg) menjadi mileage
- Menghapus *outlier* pada data

### Menghapus Missing Value
Jika dilihat dari gambar deskripsi variabel, terlihat ada beberapa kolom yang memiliki data kurang dari kolom 'mileage(km/ltr/kg)', 'engine, max_power', dan 'seats'. Sehingga akan dilakukan penghapusan missing value sehingga dataset terbebas dari missing value. Berikut adalah kode untuk menghapus data yang memiliki missing value:

```sh
car_df = car_df.dropna()
```

### Menghapus Duplicate Data
Data kita memiliki 1202 data duplikat, yang akan mengganggu proses modeling nanti nya jika dibiarkan sehingga kita perlu menghapus data duplikat tersebut. Berikut adalah kode untuk menghapus data duplikat:

```sh
car_df = car_df.drop_duplicates()
```

### Change data type and column name
#### max_power
Terlihat di deskripsi dataset bahwa pada kolom `max power` perlu kita ubah tipe data nya dari `object` menjadi `float` karena nilai `max power` yang memiliki nilai desimal pada dataset. Sebelumnya, tipe data belum bisa diubah karena ada data yang mempunyai nilai ' ' sehingga kolom tidak bisa langsung diubah ke tipe data float. Oleh karena itu, kita perlu menghapus data yang memiliki nilai ' ' kemudian baru kita mengubah tipe data pada kolom max power menjadi float. Berikut adalah kode untuk mengubah tipe data 'max_power' menjadi *float*:

```sh
car_df['max_power'] = car_df['max_power'].astype('float')

```

#### seats
Setelah itu, kita akan mengganti tipe data seats dari sebelumnya float menjadi integer karena seats tidak mungkin bilangan pecahan. Berikut adalah kode untuk mengubah tipe data 'max_power' menjadi *integer*:

```sh
car_df['seats'] = car_df['seats'].astype(int)
```

#### mileage (km/ltr/kg)
Setelah itu, kita mengubah nama kolom 'mileage(km/ltr/kg)' menjadi 'mileage' supaya lebih mudah dalam proses data preparation. 
Berikut adalah kode untuk mengubah nama kolom 'mileage(km/ltr/kg)' menjadi 'mileage':

```sh
car_df.rename(columns={'mileage(km/ltr/kg)':'mileage'}, inplace = True)
```

### Outliers
Outlier dapat didefenisikan sebagai amatan yang menyimpang sedemikian jauh dari pengamatan lainnya. Adanya data outlier ini dapat mempunyai efek bagi pengambilan suatu kesimpulan atau keputusan pada penelitian. Oleh karena itu,  kita perlu menghapus outlier supaya tidak merusak hasil analisis data. Pada tahap ini, kita akan mendeklarasikan 'num_features' terlebih dahulu sebagai kolom yang mempunyai fitur numerik. Setelah itu, kita akan menghapus nilai outlier pada data menggunakan metode IQR(*Interquartile range*). 

Pada metode IQR, kita perlu mencari nilai IQR dengan
$IQR = Q3 - Q1$.
Setelah itu, kita mencari batas bawah dan batas atas dengan 

$Batas Bawah = Q1 - 1.5*IQR$ 

dan

$Batas Atas = Q3 + 1.5*IQR$.

Setelah itu, kita akan menghapus data di luar rentang Batas Atas dan Batas bawah. 

Setelah dilakukan cleaning data, data bersih memiliki 5385 data. Berikut merupakan analisis deskripsi dari fitur numerik yang telah bersih

![Screenshot 2025-01-04 194757](https://github.com/user-attachments/assets/ab11d8be-1a3c-4af7-ad79-55bf3c2b39c2)


## EDA
EDA(*Exploratory Data Analysis*) adalah suatu proses uji investigasi awal yang bertujuan untuk mengidentifikasi pola, menemukan anomali, menguji hipotesis dan memeriksa asumsi. Dengan melakukan EDA, pengguna akan sangat terbantu dalam mendeteksi kesalahan dari awal, mengetahui hubungan antar data serta dapat menggali faktor-faktor penting dari data. Berikut adalah hasil proses dari EDA:

### Univariate Analysis
Univariate Analysis melibatkan pemeriksaan satu variabel pada satu waktu untuk meringkas dan menemukan pola. Pada proses ini, data dibagi menjadi 2 bagian, yaitu `number features` dan `categorical features`. Lalu akan ditunjukkan visualisasi menggunakan `barplot` dari kedua *features* tersebut. 

#### Categorical Features

![4](https://github.com/user-attachments/assets/96144904-d979-486e-a83d-369ed5b4feaf)


Dari Gambar diatas, kita bisa mengetahui bahwa:
- Top 5 mobil bekas yang paling banyak adalah Maruti Swift Dzire VDI, Maruti Alto 800 LXI, Maruti Alto LXi, Maruti Swift VDI, dan Maruti Alto K10 VXI
- Mobil terbanyak yang tersedia yaitu Maruti Swift Dzire VDI dengan 118 data dengan persentase sebesar 2,12%.
- Mobil bekas paling banyak menggunakan tipe bahan bakar Petrol/Bensin dengan 2848 data dengan persentase 52,9%.
- Mobil bekas paling sedikit menggunakan tipe bahan bakar LPG dengan 34 data dengan persentase 0,6%
- Mobil bekas paling banyak menggunakan tipe seller individual dengan 4877 data dengan persentase 90,6%
- Mobil bekas paling sedikit menggunakan tipe seller dealer terpercaya dengan 25 data dengan persentase 0,5%
- Mobil bekas paling banyak menggunakan transmisi manual dengan 5099 data dengan persentase 94,7%
- Mobil bekas paling sedikit menggunakan transmisi otomatis dengan 286 data dengan persentase 5,3%
- Mobil bekas paling banyak merupakan pemilik pertama dengan 3404 data dengan persentase 63,2%
- Mobil bekas paling sedikit merupakan pemilik keempat atau lebih dengan 116 data dengan persentase 2,2%
- Top 5 mobil bekas yang paling banyak mempunyai kursi sebanyak 5, 7, 4, 8, dan 6 kursi
- Mobil yang memiliki 5 kursi merupakan mobil bekas terbanyak dengan 4906 data dengan persentase sebesar 91,1%

#### Numenical Features

![5](https://github.com/user-attachments/assets/4883e309-a71e-43da-a9e0-29132b2e8c96)


Dari gambar diatas, kita bisa mengetahui bahwa:

- Banyak harga jual mobil yang termasuk murah di sekitar 200000 sd 400000.
- Kebanyakan mobil bekas yang dijual telah digunakan sepanjang 2500 sampai 12500 km.
- Kebanyakan mobil bekas yang dijual menempuh jarah 15-25 km per liter bahan bakarnya.
- Kebanyakan mobil bekas yang dijual memiliki mesin sebesar 1200 cc.

### Multivariate Analysis
Multivariate Analysis mengeksplorasi hubungan antara dua variabel atau lebih secara bersamaan. Pada proses ini, data dibagi menjadi 2 bagian, yaitu `number features` dan `categorical features`. Lalu akan ditunjukkan visualisasi menggunakan `catplot` pada *categorical features* dan `pairplot` dari *numerical features*.

#### Categorical Features

![6](https://github.com/user-attachments/assets/dbf4d3c6-fb71-4e91-9b0e-52dafb534031)

Dari gambar diatas, kita bisa mengetahui bahwa:
- Pada fitur 'year', rata-rata harga semakin naik. Mobil keluaran tahun 2019 mempunyai rata-rata harga tertinggi diantara tahun lainnya. 
- Pada fitur 'fuel', Mobil bertipe bahan bakar diesel/solar mempunyai rata-rata harga tertinggi dibandingkan tipe bahan bakar lainnya.
- Pada fitur 'seller', Mobil bertipe seller dealer terpercaya mempunyai rata-rata harga tertinggi dibandingkan tipe bahan bakar lainnya.
- Pada fitur 'transmission', Mobil bertipe transmisi otomatis mempunyai rata-rata harga lebih tinggi dibandingkan tipe transmisi manual.
- Pada fitur 'owner', Mobil bertipe kepemilikan orang pertama mempunyai rata-rata harga tertinggi dibandingkan tipe bahan bakar lainnya.
- Pada fitur 'seats', Mobil yang mempunyai 7 kursi mempunyai rata-rata harga tertinggi dibandingkan tipe bahan bakar lainnya.

#### Numerical Features

![7](https://github.com/user-attachments/assets/f61aa84e-a8d0-4083-991e-5a4c983623d3)

Dari gambar diatas, kita bisa mengetahui bahwa:
- Pada fitur 'year', 'mileage', 'engine', dan 'max_power' memiliki korelasi positif dengan 'selling_price'
- Pada fitur 'km_driven' memiliki korelasi negatif dengan 'selling_price
- Pada fitur 'seats' memiliki korelasi yang acak dengan 'selling_price'

### Correlation Matrix

Matriks korelasi adalah sebuah matriks yang menunjukkan koefisien korelasi antar variabel. Jika nilai dari matriks mendekati -1, maka korelasi negatif antar variabel semakin kuat. Jika nilai dari matriks mendekati , maka korelasi antar variabel semakin minim. Jika nilai dari matriks mendekati 1, maka korelasi positif antar variabel semakin kuat.

![8](https://github.com/user-attachments/assets/49047ffc-a2f9-41e9-8a01-d3c9aa2556cc)

Terlihat bahwa masing-masing fitur memiliki relasi yang cukup kuat dengan 'selling_price', sehingga tidak ada penghapusan fitur yang tidak memiliki korelasi.

## Data Transformation
Data Transformation merupakan tahap transformasi data kita. Data Transformation penting dilakukan supaya data kita bisa melakukan modeling data dengan baik. Berikut merupakan tahap Data Preparation:
1. Encoding fitur kategori(Target Encoding dan Label Encoding)
2. Train Test Split
3. Standarisasi(Standarization)
### Encoding fitur kategori

Pada tahap ini, kita menggunakan 2 metode untuk encoding, yaitu Target Encoder pada kolom `name` dan Label Encoder pada kolom kategori lainnya. Target encoder dipilih pada kolom `name` karena Encoder ini mengurangi dimensionalitas dan mempertahankan hubungan antara fitur kategoris dan variabel target. Terget Encoder membutuhkan kolom y(`selling price`) untuk menghitung rata-rata target per kategori. Proses yang dilakukan yaitu mengubah isi kolom `selling price` terlebih dahulu ke bentuk logaritma nya karena  target dari kolom y memiliki nilai yang cukup besar, sehingga berimbas ke Target Encoding yang tidak efektif. Setelah itu dilakukan proses Target Encoding. Setelah itu, kolom `selling price log` akan dihapus.
Setelah itu dilakukan proses Label Encoding. Label Encoding dipilih karena lebih efisien dalam memori dan komputasi. Label Encoding dilakukan di kolom kategori selain `name`.  
```sh
car_df['selling_price_log'] = np.log1p(car_df['selling_price'])
car_df['name']= target_encoder.fit_transform(car_df[['name']], car_df['selling_price_log'])
```
```sh
car_df['fuel']= label_encoder.fit_transform(car_df['fuel'])
```

### Train Test Split

Train test split adalah proses membagi data menjadi data latih dan data uji. Pada proses ini, kita membagi data dengan rasio 80:20. Kemudian didapat hasil pembagian data latih dan data uji.
```sh
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)
```
Hasil dari train test split adalah seperti dibawah

![9](https://github.com/user-attachments/assets/4c7a036c-403c-4ced-84ce-39b72e4054e5)

Terlihat bahwa kita mempunyai 4308 data latih dan 1077 data uji.

### Standarisasi pada kolom numerik

Standarisasi fitur numerik memiliki tujuan untuk memastikan bahwa semua fitur berkontribusi secara proporsional terhadap model. Standarisasi dilakukan dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. 
Hasil dari Standarisasi sebagai berikut

![10](https://github.com/user-attachments/assets/e75fb3b4-815d-48cc-864d-9e9a3307549f)


# Modeling
Modeling adalah tahapan di mana kita menggunakan algoritma machine learning untuk menjawab problem statement dari tahap business understanding. Ada 3 algoritma machine learning yang akan digunakan dalam projek ini, yaitu:
1. Random Forest Regressor
2. XGBoost
3. Gradient Boosting

## Random Forest Regressor

Random Forest merupakan teknik pembelajaran ensemble yang fleksibel dan canggih yang khususnya berguna untuk masalah regresi. Selama fase pelatihan, ia membangun sejumlah besar pohon keputusan dan menghasilkan prediksi rata-rata dari setiap pohon individu. Random Forest merupakan pilihan yang menarik untuk banyak aplikasi dunia nyata karena ia tahan terhadap gangguan dan outlier, mengelola kumpulan data berdimensi tinggi secara efektif, dan menghasilkan estimasi relevansi fitur. Random Forest beroperasi dengan membangun beberapa pohon keputusan selama pelatihan dan menghasilkan prediksi rata-rata (regresi) dari pohon individu. Prinsip yang mendasarinya melibatkan pembuatan serangkaian pohon yang beragam dan menggabungkan prediksi mereka untuk meningkatkan akurasi dan ketahanan secara keseluruhan. Adapun kelebihan dari Random Forest sebagai berikut:
- Akurasi Tinggi
- Ketahanan terhadap Noise
- Menaksir Pentingnya Fitur
- Menangani Data yang Hilang dan Outlier
- Menangani Data Numerik dan Kategoris

Sedangkan kekurangan dari Random Forest sebagai berikut:
- Kompleksitas Komputasi
- Penggunaan Memori yang lebih banyak
- Waktu Prediksi yang lebih lama
- Dapat mengalami Overfitting

berikut kode untuk model Random Forest Regressor

```sh
rf = RandomForestRegressor(n_estimators=100, random_state=123)
```

## XGBoost

XGBoost, atau *Extreme Gradient Boosting* adalah algoritma pembelajaran mesin yang populer dan canggih yang termasuk dalam kategori teknik peningkatan gradien. Algoritma ini banyak digunakan untuk tugas klasifikasi dan regresi. XGBoost menyempurnakan pendekatan peningkatan gradien tradisional dengan menggabungkan berbagai teknik pengoptimalan dan regularisasi, sehingga menghasilkan peningkatan akurasi dan efisiensi. XGBoost menggabungkan prediksi beberapa algoritma tradisional, biasanya pohon keputusan, untuk membuat model prediktif yang kuat. Intuisi di balik XGBoost melibatkan pengoptimalan melalui penurunan gradien dan peningkatan. Berikut Kelebihan XGBoost:
- Akurasi Tinggi
- Penanganan data Nonlinier
- Penanganan Data yang Hilang
- Pemrosesan Paralel
- XGBoost dioptimalkan untuk performa dan penggunaan memori
 
Berikut kekurangan XGBoost:
- Kompleksitas
- Risiko Overfitting

berikut kode untuk model XGBoost:

```sh
xgb_r = xgb.XGBRegressor(objective ='reg:squarederror', random_state=123)
```

## Gradient Boosting

Gradien Boosting adalah teknik pembelajaran mesin yang digunakan untuk tugas regresi dan klasifikasi. Teknik ini membangun model secara berurutan, setiap model mencoba memperbaiki kesalahan model sebelumnya. Tidak seperti algoritme lain yang berfokus pada satu model tunggal, Peningkatan Gradien menggabungkan beberapa model tradisional (biasanya pohon keputusan) untuk membentuk model prediktif yang kuat. Gradient Boosting bekerja dengan inisialisasi model dengan prediksi sederhana terlebih dahulu, lalu hitung residual untuk setiap titik data dengan menemukan perbedaan antara nilai aktual dan prediksi. Setelah itu, pasangkan model tradisional (biasanya pohon keputusan) ke residual ini. Lalu perbarui prediksi dengan menambahkan prediksi model baru, yang diskalakan berdasarkan laju pembelajaran, ke prediksi yang ada. Ulangi langkah 2–4 untuk sejumlah iterasi yang ditetapkan atau hingga residual diminimalkan secara memadai. Berikut kelebihan Gradient Boosting:
- Akurasi Tinggi
- Fleksibel
- Dapat menangani data non linear

Berikut kekurangan Gradient Boosting
- Gradient Boosting memakan banyak waktu pelatihan
- Dapat Overfitting
- Memerlukan penyetelan(Tuning)

berikut kode untuk model Gradient Boosting:

```sh
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=123)
```

Setelah dilakukan modeling, maka dilakukan evaluasi model mana yang memiliki kinerja paling baik. 

# Evaluasi 

Pada tahap ini, kita akan menguji seberapa efektifnya suatu model dan membandingkan 3 model mana yang memiliki kinerja paling baik. Sebelum melakukan evaluasi, fitur numerik pada data uji harus distandarisasi terlebih dahulu. supaya didapat mean = 0 dan standar deviasi = 1.  Pada tahap evaluasi ini, kita akan menggunakan MSE(*Mean Square Error*). MAE mencari selisih kuadrat antara nilai aktual dan nilai prediksi. Semakin kecil nilai MAE, maka semakin bagus juga modelnya. Berikut formula dari MSE

![rumus](https://github.com/user-attachments/assets/a6eeea90-cbba-475b-b2df-3a6b7c117e75)


dengan $N$ adalah jumlah dataset, $y_i$ adalah nilai sebenarnya, $y_{\text{pred}}$ adalah nilai prediksi. Setelah dicoba proses evaluasi, berikut adalah hasil dari evaluasi antara 3 model:

![Screenshot 2025-01-04 202014](https://github.com/user-attachments/assets/5732d995-c2c6-46b8-a421-a15e68c36e30)
![download (12)](https://github.com/user-attachments/assets/0d28f4ea-70f3-493f-be52-64036644a0e3)

Bisa terlihat bahwa

- Random Forest Regressor memiliki error di data latih paling kecil, dengan nilai 772684.832459, dengan error pada data uji sebesar 5295484.003658
- XGBoost memiliki error di data uji paling kecil 5039267.84, dengan error di data latih sebesar 1106514.176
- Gradient Boosting memiliki error paling besar dari kedua algoritma lain, baik di data latih maupun data uji, dengan masing-masing nilai 4083933.339346 dan 5371753.914372, sehingga model ini kurang efektif dengan dataset ini.

Selanjutnya, kita akan melihat prediksi model dari data actualnya. 
![Screenshot 2025-01-04 202201](https://github.com/user-attachments/assets/0118846f-bac5-4230-8911-a1dbec6757b0)


Bisa dilihat bahwa prediksi dari XGBoost lebih akurat daripada Random Forest Regressor karena hasil prediksi XGBoost lebih mendekati nilai data aktual. 
Sehingga bisa disimpulkan bahwa model yang cocok dengan projek prediksi penjualan harga mobil bekas ini adalah model XGBoost karena memiliki error yang lebih rendah dan nilai prediksi yang mendekati nilai sebenarnya.

Sehingga berikut kesimpulan yang didapat dari projek ini:

Projek ini mendemonstrasikan proses pembuatan model yang baik, mulai dari persiapan data (data cleaning, EDA, data transformation), pemilihan algoritma , pelatihan model, hingga evaluasi. Projek ini menggunakan 3 algoritma, yaitu *Random Forest Regressor*, *XGBoost*, dan *Gradient Boosting* dan mengevaluasi 3 model dari algoritma tersebut dengan MSE(solutions membantu proses ini). Hasil nya algoritma *XGBoost* memiliki MSE paling rendah, serta memiliki hasil yang mendekati data aktual. Sehingga kita dapat menggunakan model *XGBoost* untuk menentukan harga jual yang tepat untuk mobil bekas secara manual dengan banyaknya faktor yang mempengaruhi secara efektif dan efisien.(problem statement dan goals tercapai).

---

# Referensi
- Pardomuan Robinson Sihombing, Suryadiningrat, Deden Achmad Sunarjo, Yoshep Paulus, Apri Caraka Yuda, "Identifikasi Data Outlier (Pencilan) dan Kenormalan Data Pada Data Univariat serta Alternatif Penyelesaiannya", BPS-Statistics Indonesia, 2022, Retrieved from: https://jurnaljesi.com/index.php/jurnaljesi/article/view/112
- NORTH DAKOTA STATE UNIVERSITY, "MULTIVARIATE ANALYSES", Retrieved from: https://www.ndsu.edu/faculty/horsley/Introduction_and_describing_variables.pdf
- w3schools, "Data Science - Statistics Correlation Matrix", Retrieved from: https://www.w3schools.com/datascience/ds_stat_correlation_matrix.asp
- geeksforgeeks, "What are the Advantages and Disadvantages of Random Forest?", 2024, Retrieved from: https://www.geeksforgeeks.org/what-are-the-advantages-and-disadvantages-of-random-forest/
- Ambika, "XGBoost Algorithm in Machine Learning", 2023, Retrieved from: https://medium.com/@ambika199820/xgboost-algorithm-in-machine-learning-2391edb101ce
- Piyush Kashyap, "A Comprehensive Guide to Gradient Boosting and Regression in Machine Learning: Step-by-Step Intuition and Example", 2024, Retrieved From: https://medium.com/@piyushkashyap045/a-comprehensive-guide-to-gradient-boosting-and-regression-in-machine-learning-step-by-step-faa17fbd0e2c#:~:text=Pros%20and%20Cons%20of%20Gradient%20Boosting,-Pros%3A&text=High%20accuracy%3A%20Often%20outperforms%20other,Boosting%20can%20model%20complex%20relationships.
- Raghav Agrawal, "Know The Best Evaluation Metrics for Your Regression Model !", 2024, Retrieved from: https://www.analyticsvidhya.com/blog/2021/05/know-the-best-evaluation-metrics-for-your-regression-model/#h-mean-squared-error-mse
- DQLAB, "Exploratory Data Analysis : Pahami Lebih Dalam untuk Siap Hadapi Industri Data", 2020, Retrieved from: https://dqlab.id/data-analisis-machine-learning-untuk-proses-pengolahan-data



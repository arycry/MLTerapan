# -*- coding: utf-8 -*-
"""Prediksi harga mobil bekas.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12e3FwJ0aEJVdbh8pnwWXDgn-LfH9s2bm

# Data Collection
"""

# import library untuk persiapan data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# menampilkan dataframe
car_df = pd.read_csv('cardekho.csv')
car_df.head(10)

# melihat info variabel data
car_df.info()

"""# Data Understanding

## Remove null value
"""

# mengecek nilai null
car_df.isnull().sum()

# menghapus nilai null
car_df = car_df.dropna()

"""## Remove Duplicate Data"""

#mengecek nilai duplikat
duplicate_rows = car_df.duplicated()

print(duplicate_rows.sum())

#menghapus nilai duplikat
car_df = car_df.drop_duplicates()

#mengecek info variabel
car_df.info()

"""## Change data type and column name"""

#mengganti tipe data max power
car_df['max_power'] = car_df['max_power'].astype('float')

#mengecek nilai pada max power yang memiliki nilai ' '
car_df[car_df['max_power'] == ' ']

#menghapus data yang memiliki nilai ' '
car_df['max_power'] = car_df['max_power'].replace(' ', np.nan)

car_df = car_df.dropna(subset=['max_power'])

#mengecek kembali nilai pada max power yang memiliki nilai ' '
car_df[car_df['max_power'] == ' ']

#mengganti tipe data max power
car_df['max_power'] = car_df['max_power'].astype('float')

#mengganti tipe data seats
car_df['seats'] = car_df['seats'].astype(int)

#mengecek info variabel
car_df.info()

#mengganti nama kolom mileage(km/ltr/kg) menjadi mileage
car_df.rename(columns={'mileage(km/ltr/kg)':'mileage'}, inplace = True)

#mengecek deskripsi stastistik
car_df.describe()

"""## Remove Outlier"""

#mengecek outlier selling price
sns.boxplot(x=car_df['selling_price'])

#mengecek outlier km driven
sns.boxplot(x=car_df['km_driven'])

#mengecek outlier mileage
sns.boxplot(x=car_df['mileage'])

#mengecek outlier engine
sns.boxplot(x=car_df['engine'])

# mengecek outlier max power
sns.boxplot(x=car_df['max_power'])

#mendeklarasikan num_features sebagai fitur numerik
num_features = ['selling_price',	'km_driven',	'mileage',	'engine',	'max_power']
car_df[num_features]

#menghapus outlier menggunakan IQR
Q1 = car_df[num_features].quantile(0.25)
Q3 = car_df[num_features].quantile(0.75)
IQR=Q3-Q1
car_df=car_df[~((car_df[num_features]<(Q1-1.5*IQR))|(car_df[num_features]>(Q3+1.5*IQR))).any(axis=1)]

#mengecek info variabel
car_df.info()

"""## EDA"""

#mendeklarasikan cat_features sebagai fitur kategori
cat_features = ['year', 'name',	'fuel',	'seller_type', 'transmission', 'owner', 'seats']

"""### Univariate Analysis

#### Categorical Features
"""

#visualisasi 5 besar dari kolom year
feature = cat_features[0]
count = car_df[feature].value_counts()
percent = 100*car_df[feature].value_counts(normalize=True)

top_5_count = count.head(5)
top_5_percent = percent.head(5)

df = pd.DataFrame({'jumlah sampel':top_5_count, 'persentase':top_5_percent.round(1)})
print(df)
top_5_count.plot(kind='bar', title=feature);

#visualisasi 5 besar dari kolom name
feature = cat_features[1]
count = car_df[feature].value_counts()
percent = 100*car_df[feature].value_counts(normalize=True)

top_5_count = count.head(5)
top_5_percent = percent.head(5)

df = pd.DataFrame({'jumlah sampel':top_5_count, 'persentase':top_5_percent.round(1)})
print(df)
top_5_count.plot(kind='bar', title=feature);

#visualisasi dari kolom fuel
feature = cat_features[2]
count = car_df[feature].value_counts()
percent = 100*car_df[feature].value_counts(normalize=True)

df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

#visualisasi  dari kolom seller type
feature = cat_features[3]
count = car_df[feature].value_counts()
percent = 100*car_df[feature].value_counts(normalize=True)

df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

#visualisasi dari kolom transmission
feature = cat_features[4]
count = car_df[feature].value_counts()
percent = 100*car_df[feature].value_counts(normalize=True)

df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

#visualisasi dari kolom owner
feature = cat_features[5]
count = car_df[feature].value_counts()
percent = 100*car_df[feature].value_counts(normalize=True)

top_5_count = count.head(5)
top_5_percent = percent.head(5)

df = pd.DataFrame({'jumlah sampel':top_5_count, 'persentase':top_5_percent.round(1)})
print(df)
top_5_count.plot(kind='bar', title=feature);

#visualisasi 5 besar dali kolom seats
feature = cat_features[6]
count = car_df[feature].value_counts()
percent = 100*car_df[feature].value_counts(normalize=True)

top_5_count = count.head(5)
top_5_percent = percent.head(5)

df = pd.DataFrame({'jumlah sampel':top_5_count, 'persentase':top_5_percent.round(1)})
print(df)
top_5_count.plot(kind='bar', title=feature);

"""#### Numerical Features"""

#Visualisasi univariate analysis untuk fitur numerik
car_df.hist(bins=50, figsize=(20,15))
plt.show()

"""### Multivariate Analysis"""

#Visualisasi multivariate analysis untuk fitur kategori
for col in cat_features:
  sns.catplot(x=col, y="selling_price", kind="bar", dodge=False, height = 4, aspect = 3,  data=car_df, palette="Set3")
  plt.title("Rata-rata 'selling_price' Relatif terhadap - {}".format(col))

#Visualisasi multivariate analysis untuk fitur numerik
sns.pairplot(car_df, diag_kind = 'kde')

"""### Correlasion Matrix"""

#Membuat confussion matrix
plt.figure(figsize=(10, 8))
correlation_matrix = car_df[num_features].corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

#mengecek kembali dataset
car_df.head()

"""# Data Preparation"""

#import library untuk target encoding dan label encoder
from sklearn.preprocessing import TargetEncoder, LabelEncoder

target_encoder = TargetEncoder()
label_encoder = LabelEncoder()

"""## Target Encoding"""

#membuat kolom selling_price_log dan proses target encoding
car_df['selling_price_log'] = np.log1p(car_df['selling_price'])
car_df['name']= target_encoder.fit_transform(car_df[['name']], car_df['selling_price_log'])

#mengecek kembali dataframe
car_df.head()

"""## Label Encoding"""

#label encoding untuk kolom fuel
car_df['fuel']= label_encoder.fit_transform(car_df['fuel'])

car_df['fuel'].unique()

#label encoding untuk kolom seller type
car_df['seller_type']= label_encoder.fit_transform(car_df['seller_type'])

car_df['seller_type'].unique()

#label encoding untuk kolom transmission
car_df['transmission']= label_encoder.fit_transform(car_df['transmission'])

car_df['transmission'].unique()

#label encoding untuk kolom owner
car_df['owner']= label_encoder.fit_transform(car_df['owner'])

car_df['owner'].unique()

#mengecek kembali dataframe
car_df.head()

#menghapus kolom selling_price_log
car_df.drop(['selling_price_log'], axis = 1, inplace = True)

#mengecek kembali dataframe
car_df.head()

"""## Train-Test_Split"""

#mendefinisikan variabel x dan y
x = car_df.drop(['selling_price'], axis = 1)
y = car_df['selling_price']

#proses train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)

#Melihat banyaknya data train dan data test setelah train test split
print(f'Total # of sample in whole dataset: {len(x)}')
print(f'Total # of sample in train dataset: {len(x_train)}')
print(f'Total # of sample in test dataset: {len(x_test)}')

"""## Standarization"""

#proses standarisasi pada data latih
from sklearn.preprocessing import StandardScaler

numerical_features = ['km_driven',	'mileage',	'engine',	'max_power']
scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
x_train[numerical_features].head()

"""# Modeling"""

#membuat datafram yang memuat hasil MSE 3 model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=[ 'RandomForest', 'XGBoost', 'GradientBoosting'])

#import MSE
from sklearn.metrics import mean_squared_error

"""## Random Forest Regressor"""

#import, proses Random Forest Regressor, dan menyimpan masih MSE
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=123)
rf.fit(x_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred = rf.predict(x_train), y_true=y_train)

"""## XGBoost"""

#install xgboost
!pip install xgboost

#import, proses XGBoost, dan menyimpan masih MSE
import xgboost as xgb

xgb_r = xgb.XGBRegressor(objective ='reg:squarederror', random_state=123)
xgb_r.fit(x_train, y_train)

models.loc['train_mse','XGBoost'] = mean_squared_error(y_pred=xgb_r.predict(x_train), y_true=y_train)

"""## Gradient Boosting"""

#import, proses Gradient Boosting, dan menyimpan masih MSE
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=1)
gbr.fit(x_train, y_train)

models.loc['train_mse','GradientBoosting'] = mean_squared_error(y_pred=gbr.predict(x_train), y_true=y_train)

"""# Evaluation"""

#standarisasi pada data uji
x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])

#Menampilkan hasil MSE 3 model untuk data latih dan data uji

mse = pd.DataFrame(columns=['train', 'test'], index=['RF','XGBoost','GradientBoosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = { 'RF': rf, 'XGBoost':xgb_r, 'GradientBoosting':gbr}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))/1e3

# Panggil mse
mse

#visualisasi MSE 3 model
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

#menampilkan prediksi dari 3 model
prediksi = x_test.iloc[:5].copy()
pred_dict = {'y_true':y_test[:5]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
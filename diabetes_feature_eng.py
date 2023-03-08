#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

# To-Do list
# outlier değerleri bul (diabet verisinde bazı değişkenler min değeri sıfır olamaz)
# değeri minimum sıfır olanları Na/median/mean olarak değiştir.

"""
İş Problemi

Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
gerçekleştirmeniz beklenmektedir.


Veri Seti Hikayesi

Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian
kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. Hedef değişken "outcome" olarak belirtilmiş
olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir
9 Değişken - 768 Gözlem - 24 KB

Pregnancies               -> Hamilelik sayısı
Glucose Oral              -> glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
Blood Pressure            -> Kan Basıncı (Küçük tansiyon) (mm Hg)
SkinThickness             -> Cilt Kalınlığı
Insulin                   -> 2 saatlik serum insülini (mu U/ml)
DiabetesPedigreeFunction  -> Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
BMI                       -> Vücut kitle endeksi
Age                       -> Yaş (yıl)
Outcome                   -> Hastalığa sahip (1) ya da değil (0)
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv("Hafta_06/Odev/diabets/diabetes.csv")
df = df_.copy()

df.head()

##############################
#        GOREV 1
##############################

# Adım 1: Genel resmi inceleyiniz.


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df.describe().T

#                            count    mean     std    min    25%     50%     75%     max
# Pregnancies              768.000   3.845   3.370  0.000  1.000   3.000   6.000  17.000
# Glucose                  768.000 120.895  31.973  0.000 99.000 117.000 140.250 199.000
# BloodPressure            768.000  69.105  19.356  0.000 62.000  72.000  80.000 122.000
# SkinThickness            768.000  20.536  15.952  0.000  0.000  23.000  32.000  99.000 # sıfır değeri düzeltilmeli
# Insulin                  768.000  79.799 115.244  0.000  0.000  30.500 127.250 846.000 # sıfır değeri düzeltilmeli ve max değeri de
# BMI                      768.000  31.993   7.884  0.000 27.300  32.000  36.600  67.100 # sıfır değeri düzeltilmeli
# DiabetesPedigreeFunction 768.000   0.472   0.331  0.078  0.244   0.372   0.626   2.420
# Age                      768.000  33.241  11.760 21.000 24.000  29.000  41.000  81.000
# Outcome                  768.000   0.349   0.477  0.000  0.000   0.000   1.000   1.000

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
# kategorik değişkenimiz yok fakat kendimiz yaratırsak diye cat_sumaary fonksiyonunu da ekleyelim

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.ylabel(numerical_col)
        plt.show()


num_summary(df, "Insulin")

for col in num_cols:
    num_summary(df, col, plot=True)

#  4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

def target_summary_with_cat(dataframe, target, categorical_col):

    print(pd.Dataframe({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")



def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# Adım 5: Aykırı gözlem analizi yapınız.

def outliers_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in num_cols:
    print(col, ": ", outliers_thresholds(df, col))

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False):
    low, up = outliers_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]).head()
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

# aykırı değerleri eşik değeri ile basıklayalım

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outliers_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# Adım 6: Eksik gözlem analizi yapınız.
#############################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss= dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Adım 7: Korelasyon analizi yapınız.

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

######################
#     GOREV 2
######################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.


for col in num_cols:
    if col != "Pregnancies":
        df[col].replace({0: np.nan}, inplace=True)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# eksik değerler için KNN imputer algoritmasını kullanmak istiyorum

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)

dff = df.copy()

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff.describe().T

# KNN imputer uyguladığımızda hala bazı değerlerin mantıklı olmadığını gördüm bu sebeple tekrar aykırı değer analizi yapacağım

for col in num_cols:
    print(col, outliers_thresholds(dff, col, q1=0.25, q3=0.75))

# şimdi daha mantıklı bir yapı oluşmaya başladı
# sonraki adım olarakta aykırı değerleri eşik ile değiştirelim

for col in num_cols:
    replace_with_thresholds(dff, col)
# aykırı değer var mı bakalım

for col in num_cols:
    print(check_outlier(dff, col)) #aykırı değişken yok
# nan değer var mo ona da bakalım
dff.isnull().sum() # yok
#######################################
# Adım 2: Yeni değişkenler oluşturunuz.
#######################################
# kendi değişkenlerimin hepsini küçük harf yapıyorum ki benim yarattığım anlaşılsın

dff["bmi_range"] = pd.cut(dff["BMI"], bins=[0,18.5,25,30,100], labels=["Underweight", "Healthy",
                                                                       "Overweight", "Obese"])
dff["insulin_desc"] = [ "Normal" if val >= 16 and val <= 166 else "Abnormal" for val in dff["Insulin"]]

dff["glucose_desc"] = pd.cut(x=dff["Glucose"], bins=[0,70,99,126,200], labels=["Low","Normal","Secret","High"])

# Adım 3: Encoding işlemlerini gerçekleştiriniz.

#grab col fonksiyonu ile kategorik ve sayısal değişkenleri belirleyelim.

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

cat_cols = [col for col in cat_cols if "Outcome" not in col] #bağımlı değişkeni çıkardık

dff = pd.get_dummies(dff, columns=cat_cols ,drop_first=True)
dff.head()

cat_cols, num_cols, cat_but_car = grab_col_names(dff)
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

scaler = MinMaxScaler()
dff[num_cols] = scaler.fit_transform(dff[num_cols])
dff.head()

# Adım 5: Model oluşturunuz.

y = dff["Outcome"]
X = dff.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler

df=pd.read_csv("C:\VBO_DOSYALAR\ders öncesi notlar\hitters.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#############################
#Veri setine genel bakış
#############################
df.shape
df.info()
df.head(10)
df.isnull().sum() #salary değişkeninde 59 eksik gözlem var
df.describe().T

#############################
#Değişkenlerin ayrılması(numerik,kategorik,kardinal)
#############################
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
        car_th: int, optional
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

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

###########################
#veri setindeki aykırı değerlerin çeyrek değerlerce baskılanması
###########################
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
check_outlier(df,num_cols)

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df,col)

###########################
#veri setindeki eksik gözlemlerin doldurulması
###########################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

#knnle eksikleri tahminlemek için veriyi standartlaştırmamız gerekiyor
from sklearn.preprocessing import MinMaxScaler

cat_cols, num_cols, cat_but_car = grab_col_names(df)
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

#knn ile tahminleme yaparak eksik gözlemleri doldurduk
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

#standartlaştırmayı geri dönüştürüyoruz
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["Salary"]=dff["Salary"]

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.isnull().sum()

##################################
#Kategorik değişkenlerin analizi
##################################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

##################################
#Numerik değişkenlerin analizi
##################################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


##################################
# Kategorik değişkenlerin targeta göre analizi
##################################


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

#################################
#Label encoding
################################
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#####################################
#Özellik çıkarımı
#####################################
#league için A=0 ve N=1 dönüşümü yapmıştık
df.loc[(df["Assists"] <= 39.500) & (df["League"] == 1) , 'NEW_Assist&League'] = 'N-'
df.loc[(df["Assists"] <= 39.500) & (df["League"] == 0), 'NEW_Assist&League'] = 'A-'
df.loc[(df["Assists"] > 39.500) & (df["League"] == 1) , 'NEW_Assist&League'] = 'N+'
df.loc[(df["Assists"] > 39.500) & (df["League"] == 0), 'NEW_Assist&League'] = 'A+'

#Oyuncunun sezonda isabetli atış sayısının topa vuruşuna oranı
df["NEW_Hits_rate"]=df["Hits"]/(df["AtBat"]+0.01)

#Oyuncunun kariyeri boyunca isabetli atış sayısının topa vuruşuna oranı
df["NEW_CHist_rate"]=df["CHits"]/(df["CAtBat"]+0.01)

#Oyuncunun sezondaki isabet sayısının tüm kariyerindeki isabetine oranı
df["NEW_Hits"]=df["Hits"]/(df["CHits"]+0.01)

#Oyuncunun sezondaki en değerli vuruş sayısının tüm kariyerindeki en değerli vuruş sayısına oranı
df["NEW_HmRun"]=df["HmRun"]/(df["CHmRun"]+0.01)

#Oyuncunun sezondaki takıma kazandırıdığı sayısının tüm kariyerindeki takıma kazandırdığı sayıya oranı
df["NEW_HmRun"]=df["HmRun"]/(df["CHmRun"]+0.01)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#encode  etmek için kategorik değişkenlerin sınıf sayısına bakıyoruz
for i in cat_cols:
    print(i,df[i].nunique())

#oluşturula yeni değişken 'NEW_Assist&League',NewLeague nunique sayısı iki olduuğu için label encoderı tekrar
#çalıştırıp değişkenleri ayrıştırıyoruz

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

########################################
#Linear-regression uygulanışı
########################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

#Modelin daha iyi sonuçvermesi için nmerik değererleri standartlaştırıyoruz
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#Bağımlı ve bağımsız değişkenleri tanıttık
X = df.drop('Salary', axis=1)
y = df[["Salary"]]

#80 eğitim 20 test verisi olacak şekilde veri setini böldük
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)
#Modeli kurduk ve eğitim verisiyle fit ettik
reg_model = LinearRegression()
reg_model_fit=reg_model.fit(X_train, y_train)

#tahmin değeri için xtest verisi üzerinden y predict elde ettik ettik
y_pred = reg_model.predict(X_test)

#hata kare ortalamalarının kareköküne bakarak model başarısını değerlendirdik
np.sqrt(mean_squared_error(y_test, y_pred))

#rastgelelikten kurtarmak için k=10 katlı çapraz doğrulama uygulayalım ve hata kare ortalamalarına bakalım
np.mean(np.sqrt(-cross_val_score(reg_model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")))

#Modele etki eden değişkenlerin etki göstergeleri
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value":model.coef_[0], 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(reg_model,X_train)












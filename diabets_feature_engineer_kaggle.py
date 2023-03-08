#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING #
#############################################

"""
Bussines Problem

It is desired to develop a machine learning model that can predict whether people have diabetes when their characteristics are specified.
You are expected to perform the necessary data analysis and feature engineering steps before developing the model.

Data Set Story

The dataset is part of the large dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases in the USA.
Data used for diabetes research on Pima Indian women aged 21 and over living in Phoenix, the 5th largest city of the State of Arizona in the USA.
The target variable is specified as "outcome"; 1 indicates positive diabetes test result, 0 indicates negative
9 Variables - 768 Observations - 24 KB

Pregnancies               -> Number of pregnancies
Glucose Oral              -> 2-hour plasma glucose concentration in glucose tolerance test
Blood Pressure            -> Blood Pressure (Small blood pressure) (mmHg)
SkinThickness             -> Skin Thickness
Insulin                   -> 2-hour serum insulin (mu U/ml)
DiabetesPedigreeFunction  -> Function (2 hour plasma glucose concentration in oral glucose tolerance test)
BMI                       -> body mass index
Age                       -> Age (year)
Outcome                   -> Have the disease (1) or not (0)

"""

# Importing neccesary Libraries #

import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from datetime import date

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv("Hafta_06/Odev/diabets/diabetes.csv")
df = df_.copy()

# EDA (explatory data analysis) #

check_df(df)
"""
Firt Thing i have realized is that, Some feature has value of 0.00 which is imposiible to exist because
the person can not have that results, Therefore we are going to fix wrong datas
"""

# lets grab nunmeric and categoric variables

cat_cols, num_cols, cat_but_car = grab_col_names(df)
"""
Observations: 768
Variables: 9
cat_cols: 1
num_cols: 8
cat_but_car: 0
num_but_cat: 1
"""

# Analysis of numeric and categoric variables #

for col in cat_cols:
    cat_summary(df, col) #Only we have one cat. columns which is a dependent variable

for col in num_cols:
    num_summary(df, col) #some feature max and min values are obviously is an outlier. FIX IT

# Target variable Anlysis (The mean of the target variable according to the categorical variables, the mean of the numerical variables according to the target variable)

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)
"""
People who has diabetes also has higher glucose,Insulin level also their Bmi index higher than healthy people
We can see that older people have higher chance to be a Diabetes
"""

# I am not going to check cat. cols because There are no cot. variables except Outcome.

# Outlier Analysis #

for col in num_cols:
    print(col, check_outlier(df, col))
"""
Pregnancies True
Glucose True
BloodPressure True
SkinThickness True            # As i told you, All of them has a outlier but the thing is
Insulin True                    we need to be clever about what to do outliers (erase,filling with
BMI True                        mean median ???)
DiabetesPedigreeFunction True
Age True
"""

"The Insulin levels show 0.00 on 374 people so we need to impute values of them..."
"because we can not find any outliers algorithm does not see as a outliers...."

#Düşünüyorum da sıfır(0) değerlerini Nan ile doldurup Knnİmputer kullanabiliriz gibi sanki ?
# üsttyeki düşünce için dha fazla öğrnenmem lazım.

# 0 değerleri Insulin, glucose ve başka koılonlarda var şiömdilik onları median ile doldurucam.

# Glucose, BloodPressure, SkinTchikness,Insulin,BMI median ile dolduralacak.


median_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in median_cols:
    df[col].replace(0, df[col].median(), inplace=True)

# keyifli oldu keyifli !!!
# median ile doldurduk am bi bakalım bakalım aykırı değer var mı ?

for col in num_cols:
    print(col, check_outlier(df, col))

missing_values_table(df)

high_correlated_cols(df, plot=True)

# değişken isimlerini büyütmek
df.columns = [col.upper() for col in df.columns]

# let create new features

# Glucose
df['NEW_GLUCOSE_CAT'] = pd.cut(x=df['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

# Age

# Age
df.loc[(df['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

# BMI
df["NEW_BMI_RANGE"] = pd.cut(x=df["BMI"], bins=[-1, 18.5, 24.9, 29.9, 100], labels=["underweight", "healty", "overweight", "obese"])

# Blood Pressure
df["NEW_BLOOD_PRESSURE"] = pd.cut(x=df["BLOODPRESSURE"], bins=[-1, 79, 89, 123], labels=["normal", "hs1", "hs2"])

check_df(df)

# Yeni kolonları da ayrıştıralım tekrardan

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

df = one_hot_encoder(df, cat_cols, drop_first=True)
check_df(df)


# Son güncel değişken türlerimi tutuyorum.
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

X_scaled = RobustScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)


y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

check_df(X)


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


base_models(X, y, scoring="f1")
import sklearn
sklearn.metrics.SCORERS.keys()

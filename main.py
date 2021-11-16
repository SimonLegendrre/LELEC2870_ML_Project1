# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


# import data set
explanatory_train = pd.read_csv('X1.csv')
train_target = pd.read_csv('Y1.csv', header=None)
train_target.columns = ["target"]

# Data Cleaning :
# Je modifie les variable catégorielle sous forme de nombre en mots compréhensibles, c'est mieux pour la suite

# Création du dictionnaire
cleaned_gender = {"FCVC": {1.0: "Never", 2.0: "Sometimes", 3.0: "Always"},
                  "NCP": {1.0: "Betw1_2", 2.0: "Three", 3.0: "Tree++"},
                  "CH2O": {1.0: "Less1L", 2.0: "Betw1_2", 3.0: "Two++"},
                  "FAF": {1.0: "no_acti", 2.0: "1_2Day", 3.0: "2_4Day", 4.0: "4_5Day"},
                  "TUE": {1.0: "0_2hour", 2.0: "3_5hour", 3.0: "Five++"}}

# Remplacement dans le dataframe
explanatory_train = explanatory_train.replace(cleaned_gender)

# Binarized categorical variables
explanatory_train = pd.get_dummies(explanatory_train,
                                   columns=["Gender", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC",
                                            "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"],
                                   prefix=["Gender", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC",
                                           "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"])

# features selection for the linear regression
# scat_plot_1
x = explanatory_train['Age']
y = train_target['target']

stats = linregress(x, y)

m = stats.slope
b = stats.intercept

plt.scatter(x, y)
plt.title('target~age')
plt.plot(x, m * x + b, color='red')
plt.savefig("target_age.png")

# scat_plot_2
x = explanatory_train['Height']
y = train_target['target']

stats = linregress(x, y)

m = stats.slope
b = stats.intercept

plt.scatter(x, y)
plt.title('target~height')
plt.plot(x, m * x + b, color='red')
plt.savefig("target_height.png")


# %%

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# housing = pd.DataFrame(pd.read_csv("Housing.csv"))
# housing.head()
# housing.shape
# housing.info()
# housing.describe()
# housing.isnull().sum()*100/housing.shape[0]



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


df_train = pd.read_csv('csvjson.csv')



df_train['gas'] = df_train['gas'].map({'yes':1 ,'no':0})

df_train['furnished'] = df_train['furnished'].map({'yes':1 ,'no':0})




X = df_train[['gas', 'furnished', 'number_of_rooms', 'number_of_bathrooms']]
Y = df_train['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=101)

# X_train

lm = LinearRegression()

lm.fit(X_train, Y_train)

pr = lm.predict(X_test)
# plt.scatter(Y_test,pr)
# print(X_test)
# print(Y_test)
# print(pr)
# new_X_test=[[1,1,5,4]]
# print(lm.predict(new_X_test))
def predict(new_X_test):
    return lm.predict([new_X_test])

# print(predict([1,1,5,4])[0])

from forex_python.converter import CurrencyRates

c = CurrencyRates()

print(c.get_rate('EGP', 'USD'))

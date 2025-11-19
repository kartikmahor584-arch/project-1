import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("areaprice071.csv")
print(df)

x = df[['area']]
print(x)

y = df[['price']]
print(y)

model = LinearRegression()
model.fit(x, y)

p = model.predict(x) 

plt.title("Area Price Graph")
plt.xlabel("Area in sq ft")
plt.ylabel("Price in INR")
plt.scatter(x, y)
plt.plot(x, p, color='green')  
plt.show()

predicted_price = model.predict(pd.DataFrame([[5000]], columns=['area']))
print(predicted_price)

m = model.coef_[0][0]
c = model.intercept_[0]
print(model.coef_)
print(model.intercept_)

manual_prediction = m * 5000 + c
print(manual_prediction)

print(df)
print(x)
print(y)
print(p)
model.predict([[5600]])
print(model.predict([[5600]]))
a = int(input("Enter area: "))
user_prediction = model.predict(pd.DataFrame([[a]], columns=['area']))
print("The price of area", a, "is", user_prediction[0][0])

import pickle
pickle.dump(model,open("areaprice.pkl","wb"))
import pandas as pd

path = 'house_prediction_model/house_prediction.csv'
dataset = pd.read_csv(path)
#print(dataset.describe())
#print(len(dataset.columns))

#cleaning of dataset
dataset = dataset.dropna(axis=0)

#my target is predictong the price, let symbolises as y
y = dataset.Price

# those intries whyich are independence and on execution yield y are x features

feature = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = dataset[feature]
#print(x.head())

'''
steps in model learning start with 
defining the model>> type, decision tree,etc
fit>> learning the model
predict>> testing the prediction
evaluation/>> mode accuracy
'''

# defining the model
from sklearn.tree import DecisionTreeRegressor

house_prediction_model = DecisionTreeRegressor(random_state = 1)

#fitting
house_prediction_model.fit(x, y)
print(x.head())

#predicting the price of house
predicted_house_prices = house_prediction_model.predict(x)
print(predicted_house_prices)



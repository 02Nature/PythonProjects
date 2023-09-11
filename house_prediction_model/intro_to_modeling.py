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

#its good practices to measuere the quality of the model, this is done by finding the Mean Absolute Error of the mode

from sklearn.metrics import mean_absolute_error

print("MAR", mean_absolute_error(y, predicted_house_prices))

#testing the model with the data that was not present while building the model is known as validation of the model
#let try this

from sklearn.model_selection import train_test_split
# train test split the dataset randomly for training and validation for both target and features

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)
#just defining the model
valid_house_prediction_model = DecisionTreeRegressor()
valid_house_prediction_model.fit(train_x, train_y)
val_prediction =valid_house_prediction_model.predict(val_x)

print("First in-sample predictions:", valid_house_prediction_model.predict(x.head()))
print("Actual target values for those homes:", y.head().tolist())

print('valid moel' ,mean_absolute_error(val_y, val_prediction))



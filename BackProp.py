from keras.models import Sequential
from keras.layers import Dense
import numpy

months = [1,2,3,4,5,6,7,8,9,10,11,12]
Data = []

model = Sequential()
#create a layer of type dense having shape of input = 3 and activation function : sigmoidal function and no of neurons :10
model.add(Dense(10,input_dim=3,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(1,activation='linear'))
# As you increase the number of hidden layer, effectiveness of backpropagation decreases; plus you need to deal with overfitting problem i.e. your network will perform very well on the training data set but it cannot generalize to new data it has not seen and gives awful performance on the new data. 

for x in months:
    dataset = numpy.loadtxt("./Training_Data/Month"+`x`+"/Train.csv", delimiter=",")
    Data.append(dataset)

Data = numpy.vstack(Data)
input = Data[:,0:3]
output = Data[:,3]

model.compile(optimizer='nadam',loss='mean_absolute_error',metrics=['mean_absolute_error'])
#fit trains the model for a fixed no of epochs
model.fit(input,output, epochs=1500, batch_size=45)
scores = model.evaluate(input, output)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))


for y in months:
    dataset = numpy.loadtxt("./Testing_Data/Month"+`y`+"/Test.csv", delimiter=",")
    input = dataset[:,0:3]
    predictions = model.predict(input)
    numpy.savetxt("./Results/Month"+`y`+".csv",numpy.hstack((dataset,predictions)),delimiter = ',',fmt='%f')

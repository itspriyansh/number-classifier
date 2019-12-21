import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

def visualize(X,Y):
	plt.figure(figsize=(10,10))
	for i in range(25):
	    plt.subplot(5,5,i+1)
	    plt.xticks([])
	    plt.yticks([])
	    plt.grid(False)
	    plt.imshow(X[i].reshape(20,20), cmap=plt.cm.binary)
	    plt.xlabel(Y[i].dot( np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[0]]) ).sum())
	plt.show()

def sigmoid(z):
	x = 1/(1+np.exp(-z))
	return x

def costFunc(theta,X,Y,lambd):
	theta = theta.reshape(401,10)
	m = X.shape[0]
	h = sigmoid( np.c_[np.ones(m), X].dot(theta) )

	J = -(np.log(h[Y==1]).sum()+np.log(1-h[Y!=1]).sum())/m + lambd * np.sum(theta[1:]**2)/(2*m)
	return J

def gradient(theta,X,Y,lambd):
	m = X.shape[0]
	theta = theta.reshape(401,10)
	h = sigmoid( np.c_[np.ones(m), X].dot(theta ))

	# grad = ( np.c_[np.ones(m), X].transpose().dot(h-Y))/m
	grad = ( np.c_[np.ones(m), X].transpose().dot(h-Y) + lambd*np.r_[np.zeros((1,theta.shape[1])), theta[1:]] )/m
	return np.ndarray.flatten(grad)
	# return grad

def trainLogisticRegression(X,Y,lambd,iteration):
	c = Y.shape[1]
	n = X.shape[1]
	theta = np.random.rand((n+1),c)/400.0
	theta = np.ndarray.flatten(theta)

	theta = minimize(costFunc, theta, args=(X,Y,lambd), method='TNC', jac=gradient, options={'disp': False,'maxiter': iteration})

	return theta.x

def predict(X,theta):
	m=X.shape[0]
	output = sigmoid(np.c_[np.ones(m), X].dot(theta))

	filtr = np.zeros((output.shape[0], output.shape[1]))

	for i in range(output.shape[0]):
		filtr[i, np.where(output[i]== np.amax(output[i]))]=1
	return filtr

def selectNormalization(X,Y,X_,Y_):
	lambd=0
	minCost = costFunc(trainLogisticRegression(X,Y,lambd,100), X_,Y_,0)

	for i in range(1,11):
		cost = costFunc(trainLogisticRegression(X,Y,i*0.1,100), X_,Y_,0)
		if(minCost>cost):
			minCost=cost
			lambd=i*0.1

	for i in range(1,11):
		cost = costFunc(trainLogisticRegression(X,Y,i,100), X_,Y_,0)
		if(minCost>cost):
			minCost=cost
			lambd=i

	return lambd

def learningCurve(X,Y,X_,Y_,lambd):
	i=100
	m=X.shape[0]
	trainingCosts = []
	validationCosts = []
	indices = []
	while i<=m:
		theta = trainLogisticRegression(X[:i],Y[:i],lambd,100)
		trainingCosts.append(costFunc(theta,X[:i],Y[:i],0))
		validationCosts.append(costFunc(theta,X_,Y_,0))
		indices.append(i)
		i=i+100
	plt.plot(indices,trainingCosts)
	plt.plot(indices,validationCosts)
	plt.show()

def main():
	dataset = spio.loadmat('ex3data1.mat')

	X=np.array(dataset['X'])
	y=np.array(dataset['y'])

	Y = np.zeros((y.shape[0],10))
	for i in range(y.shape[0]):
		Y[i][y[i]-1]=1

	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
	X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=0.25,random_state=1)

	# visualize(X_train[:25],Y_train[:25])

	# lambd=selectNormalization(X_train,Y_train,X_validation,Y_validation)
	lambd=2.0 # Pre Calculated
	print(lambd)

	learningCurve(X_train,Y_train,X_validation,Y_validation,lambd)

	iteration=300
	theta = trainLogisticRegression(X_train,Y_train,lambd,iteration)

	theta=theta.reshape(401,10)

	y_predict = predict(X_test,theta)

	classes = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

	visualize(X_test[:25],y_predict[:25])

	accuracy = (y_predict.dot(classes)==Y_test.dot(classes)).sum()/Y_test.shape[0]
	print("Prediction Accuracy: " + str( accuracy*100 ))
	# 89.0 %

main()

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

def sigmoidGradient(z):
	x = sigmoid(z)
	return x*(1-x)

def predict(X,theta,n,l1,c):
	m=X.shape[0]
	theta1 = theta[:(n+1)*l1].reshape(n+1,l1)
	theta2 = theta[(n+1)*l1:].reshape(l1+1,c)

	z2 = np.c_[np.ones(m), X].dot(theta1)
	a2 = sigmoid(z2)

	z3 = np.c_[np.ones(m), a2].dot(theta2)
	output = sigmoid(z3)

	filtr = np.zeros((output.shape[0], output.shape[1]))

	for i in range(output.shape[0]):
		filtr[i, np.where(output[i]== np.amax(output[i]))]=1
	return filtr

def costFunc(theta, X,Y,lambd, n, l1, c):
	m=X.shape[0]
	theta1 = theta[:(n+1)*l1].reshape(n+1,l1)
	theta2 = theta[(n+1)*l1:].reshape(l1+1,c)

	z2 = np.c_[np.ones(m), X].dot(theta1)
	a2 = sigmoid(z2)

	z3 = np.c_[np.ones(m), a2].dot(theta2)
	a3 = sigmoid(z3)

	J = -(np.log(a3[Y==1]).sum()+np.log(1-a3[Y==0]).sum())/m + lambd * (np.sum(theta1[1:]**2)+np.sum(theta2[1:]**2))/(2*m)

	return J

def gradient(theta, X,Y,lambd, n, l1, c):
	m=X.shape[0]
	theta1 = theta[:(n+1)*l1].reshape(n+1,l1)
	theta2 = theta[(n+1)*l1:].reshape(l1+1,c)

	z2 = np.c_[np.ones(m), X].dot(theta1)
	a2 = sigmoid(z2)

	z3 = np.c_[np.ones(m), a2].dot(theta2)
	a3 = sigmoid(z3)

	delta3 = a3-Y
	delta2 = delta3.dot(theta2[1:].transpose())*sigmoidGradient(z2)

	theta2_grad = (np.c_[np.ones(m), a2].transpose().dot(delta3) + lambd * np.r_[np.zeros((1,theta2.shape[1])), theta2[1:]]) /m
	theta1_grad = (np.c_[np.ones(m), X].transpose().dot(delta2) + lambd * np.r_[np.zeros((1,theta1.shape[1])), theta1[1:]]) /m

	return np.r_[np.ndarray.flatten(theta1_grad), np.ndarray.flatten(theta2_grad)]

def trainNeuralNetwork(X,Y,lambd,n,l1,c,iteration):
	m=X.shape[0]
	theta1 = np.random.rand(n+1,l1)/n
	theta2 = np.random.rand(l1+1,c)/l1

	theta = np.r_[np.ndarray.flatten(theta1), np.ndarray.flatten(theta2)]

	theta = minimize(costFunc, theta, args=(X,Y,lambd,n,l1,c), method='TNC', jac=gradient, options={'disp': False,'maxiter': iteration})

	return theta.x[:(n+1)*l1].reshape(n+1,l1),theta.x[(n+1)*l1:].reshape(l1+1,c)

def selectNormalization(X,Y,X_,Y_,n,l1,c):
	lambd=0
	theta1,theta2=trainNeuralNetwork(X,Y,lambd,n,l1,c,100)
	minCost = costFunc(np.r_[theta1.flatten(),theta2.flatten()], X_,Y_,0,n,l1,c,)

	for i in range(1,11):
		theta1,theta2=trainNeuralNetwork(X,Y,i*0.1,n,l1,c,100)
		cost = costFunc(np.r_[theta1.flatten(),theta2.flatten()], X_,Y_,0,n,l1,c,)
		if(minCost>cost):
			minCost=cost
			lambd=i*0.1

	for i in range(1,11):
		theta1,theta2=trainNeuralNetwork(X,Y,i,n,l1,c,100)
		cost = costFunc(np.r_[theta1.flatten(),theta2.flatten()], X_,Y_,0,n,l1,c,)
		if(minCost>cost):
			minCost=cost
			lambd=i

	return lambd

def learningCurve(X,Y,X_,Y_,lambd,n,l1,c):
	i=100
	m=X.shape[0]
	trainingCosts = []
	validationCosts = []
	indices = []
	while i<=m:
		theta1,theta2 = trainNeuralNetwork(X[:i],Y[:i],lambd,n,l1,c,100)
		theta = np.r_[np.ndarray.flatten(theta1), np.ndarray.flatten(theta2)]

		trainingCosts.append(costFunc(theta,X[:i],Y[:i],0,n,l1,c,))
		validationCosts.append(costFunc(theta,X_,Y_,0,n,l1,c,))
		indices.append(i)
		i=i+500
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

	n=X.shape[1]
	l1=200
	c=10

	# lambd=selectNormalization(X_train,Y_train,X_validation,Y_validation,n,l1,c)
	lambd=0.2
	print(lambd)
	# pre computed

	learningCurve(X_train,Y_train,X_validation,Y_validation,lambd,n,l1,c)

	iteration=300
	theta1,theta2 = trainNeuralNetwork(X_train,Y_train,lambd,n,l1,c,iteration)

	theta = np.r_[np.ndarray.flatten(theta1),np.ndarray.flatten(theta2)]
	print(theta.shape)

	y_predict = predict(X_test,theta,n,l1,c)

	classes = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

	visualize(X_test[:25],y_predict[:25])

	accuracy = (y_predict.dot(classes)==Y_test.dot(classes)).sum()/Y_test.shape[0]
	print("Prediction Accuracy: " + str( accuracy*100 ))
	# 93.40 %

main()
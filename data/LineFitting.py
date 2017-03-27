import tensorflow as tf
import numpy as np

#construct tensorflow graph: w*x=y
x_placeHolder1=tf.placeholder(tf.float32, [None])
x_placeHolder2 = tf.placeholder(tf.float32, [None])
weight1=tf.Variable([0.0]) #the weight we will adjust to fit our model
weight2 = tf.Variable([0.0])
bias = tf.Variable([1.0]) # added so we can compute datapoints for y = x + 1 
y_fit=weight1*x_placeHolder1 + weight2*x_placeHolder2 #the y values estimated from our model

y_placeHolder=tf.placeholder(tf.float32, [None]) #the correct y values
loss=tf.reduce_mean(tf.square(tf.subtract(y_fit, y_placeHolder))) #aka mean squared error

#specify the optimization function
global_step=tf.Variable(0, name='global_step', trainable=False) 
optimizer=tf.train.GradientDescentOptimizer(0.1) #gradient descent optimizer with steps scaled by 0.1
train_op=optimizer.minimize(loss, global_step=global_step) #optimization function

session=tf.Session() #compile graph
session.run(tf.initialize_all_variables()) 

#data points from function we want to fit
X_data1=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
X_data2=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
Y_data=np.asarray([0.0, 4.0, 8.0, 12.0, 16.0])

#fit our model
for step in range(1000):
    session_loss, _=session.run(
        [loss, train_op], #graph variable we want to compute
        feed_dict={x_placeHolder1: X_data1, x_placeHolder2: X_data2, y_placeHolder: Y_data}) #inputs to the graph we must specify so we can compute above vars
    print('loss', session_loss)
    
    fitted_ys=session.run(
        y_fit, #graph variable we want to compute
        feed_dict={x_placeHolder1: X_data1, x_placeHolder2: X_data2}) #inputs to the graph we must specify so we can compute above vars
    print('fitted y\'s', fitted_ys)
    
    
    
    
# weight=1.0   
# def fit_points_with_model(x_placeHolder):
#     global weight
#     y_fit=weight*x_placeHolder
#     return y_fit

# fitted_ys=fit_points_with_model(X_data)
# print('fitted y\'s', fitted_ys)

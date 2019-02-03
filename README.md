an exercise in gradient descent and OOP object creation/inherintance by  
creating a batch_gradient regressor class to implement the 
simple naive batch gradient descent on, using the
linear regression using the LMS update rule as outlined
in the CS229 notes from Andrew Ng.

the batch_gradient has several main methods, most useful of which are:
    
    1. seeAlpha: visualizes the change in loss function as a result of different
    learning rates (alpha) chosen from a user specified range. Determines the 
    optimal rate for subsequent fitting (with the option of overriding) and 
    predictions
    
    2. fit: finds the weight vectors describing the linear regression of the dataset.
    if no parameter is passed, learning rate alpha is the optimal alpha found from
    the seeAlpha method
    
    3. plot: for 2-dimensional data, visualizes the training data, MSE over iterations,
    and effect of different learning rates. for >2D data, not called 

NOTE that the X array must be shape (n_sample,n_dim-1)
and that the y array must be shape (n_sample,)
see accompanying main.py for example

a basic definition for stochastic gradient descent is also written below. It is
implemented in a stochastic_graident class, which inherited from the batch_gradient
class. Not as fully developed as the batch version at the moment.

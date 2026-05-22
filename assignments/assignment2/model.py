import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

        self.layers = [self.layer1, self.relu1, self.layer2]


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!

        for param in self.params().values():
            param.grad = np.zeros_like(param.value)


        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        output1 = self.layer1.forward(X)
        output2 = self.relu1.forward(output1)
        output3 = self.layer2.forward(output2)
        loss, d_out = softmax_with_cross_entropy(output3, y)
        backprop3 = self.layer2.backward(d_out)
        backprop2 = self.relu1.backward(backprop3)
        self.layer1.backward(backprop2)

        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in self.params().values():
            reg_loss, grad = l2_regularization(param.value, self.reg)
            param.grad += grad
            loss += reg_loss

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], int)

        output1 = self.layer1.forward(X)
        output2 = self.relu1.forward(output1)
        output3 = self.layer2.forward(output2)

        pred = np.argmax(output3, axis=1)

        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        result["layer1_w"] = self.layer1.W
        result["layer1_b"] = self.layer1.B
        result["layer2_w"] = self.layer2.W
        result["layer2_b"] = self.layer2.B


        return result

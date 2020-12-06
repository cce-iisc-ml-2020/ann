# Importing exp, random and matrix multiple function
from numpy import exp, array, random, matmul, transpose

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.weights = random.random((number_of_inputs_per_neuron, number_of_neurons)) 

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function. The gradient descent
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # dot product as sum of list comprehension doing multiplication
    def __dot(self, x, y):
        return matmul(x, y)

    # The neural network forward pass.
    def forward_pass(self, inputs):
        output_from_layer1 = self.__sigmoid(self.__dot(inputs, self.layer1.weights))
        output_from_layer2 = self.__sigmoid(self.__dot(output_from_layer1, self.layer2.weights))
        return output_from_layer1, output_from_layer2

    # We train the neural network through a process of trial and error.
    # Adjusting the weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):

        for iteration in range(number_of_training_iterations):

            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.forward_pass(training_set_inputs)

            # Back Propogation Calculation 
            # Calculate The Error for Layer 2 
            # (The difference between the desired output and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 
            # (By looking at the weights in layer 1, 
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = self.__dot(layer2_delta, transpose(self.layer2.weights))
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Adjust the weights.
            layer1_adjustment = self.__dot(transpose(training_set_inputs), layer1_delta)
            self.layer1.weights += layer1_adjustment

            layer2_adjustment = self.__dot(transpose(output_from_layer_1), layer2_delta)
            self.layer2.weights += layer2_adjustment

    # The neural network prints its weights
    def print_weights(self):
        print("# Layer 1 (4 neurons, each with 4 inputs):")
        print(self.layer1.weights)
        print("# Layer 2 (1 neuron, with 4 inputs):")
        print(self.layer2.weights)

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 4 inputs)
    layer1 = NeuronLayer(4, 4)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # 2 layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print("\nStage 1) Random starting weights: ")
    neural_network.print_weights()

    # The training set. We have 10 examples, each consisting of 4 input values
    # and 1 output value.
    training_set_inputs = array([[3.6216,8.6661,-2.8073,-0.44699],
								[4.5459,8.1674,-2.4586,-1.4621],
								[3.866,-2.6383,1.9242,0.10645],
								[3.4566,9.5228,-4.0112,-3.5944],
								[0.32924,-4.4552,4.5718,-0.9888],
								[-3.2778,1.8023,0.1805,-2.3931],
								[-2.2183,-1.254,2.9986,0.36378],
								[-3.5895,-6.572,10.5251,-0.16381],
								[-5.0477,-5.8023,11.244,-0.3901],
								[-3.5741,3.944,-0.07912,-2.1203],
                                [3.6077,6.8576,-1.1622,0.28231],
                                [3.2403,-3.7082,5.2804,0.41291],
                                [3.9166,10.2491,-4.0926,-4.4659],
                                [3.9262,6.0299,-2.0156,-0.065531],
                                [5.591,10.4643,-4.3839,-4.3379],
                                [1.5514,3.8013,-4.9143,-3.7483],
                                [-4.1479,7.1225,-0.083404,-6.4172],
                                [-2.2625,-0.099335,2.8127,0.48662],
                                [-1.7479,-5.823,5.8699,1.212],
                                [-0.95923,-6.7128,4.9857,0.32886],
                                [1.3451,0.23589,-1.8785,1.3258]])
    training_set_outputs = transpose(array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,0,0,0,0,1,1,1,1,1,1]]))

    ''' #This is the handling of csv file read 
    training_set_inputs = []
    training_set_outputs = []
    training_set_outputs_1 = []

    with open('data_banknote_authentication.csv','r') as f:
        lines = csv.reader(f)
        for line in lines:
            #print (str(float(line[0])) + "--" + str(float(line[1])) + "--" + str(float(line[2])) + "--" + str(float(line[3])) + "==" + str(float(line[4])))
            training_set_inputs.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
            training_set_outputs_1.append(float(line[4]))

    training_set_outputs = transpose(array([training_set_outputs_1]));

    print (training_set_inputs)
    print (training_set_outputs_1)
    print (training_set_outputs)
    '''

    # Train the neural network using the training set.
    # Do it 1,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    print("\nStage 2) New weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("\nTesting: Expectation O = [1]")
    test1 = array([-2.4941,3.5447,-1.3721,-2.8483])
    print ("Input: ", end='')
    print (test1)
    hidden_state, output = neural_network.forward_pass(test1)
    print ("Output: ", end='')
    print(output)
    print("\nTesting: Expectation O = [0]")
    test1 = array([3.9362,10.1622,-3.8235,-4.0172])
    print ("Input: ", end='')
    print (test1)
    hidden_state, output = neural_network.forward_pass(test1)
    print ("Output: ", end='')
    print(output)

from __future__ import division
import random
import math
from Pattern import Pattern
from Data import Data
import numpy as np


class RBFNetwork:
    def __init__(self, no_of_input, no_of_hidden, no_of_output, data):
        self.no_of_input = no_of_input
        self.no_of_hidden = no_of_hidden
        self.no_of_output = no_of_output
        self.data = data
        self.input = np.zeros(self.no_of_input)
        self.centroid = np.zeros((self.no_of_hidden, self.no_of_input))
        self.sigma = np.zeros(self.no_of_hidden)
        self.hidden_output = np.zeros(self.no_of_hidden)
        self.hidden_to_output_weight = np.zeros((self.no_of_hidden, self.no_of_output))
        self.output = np.zeros(self.no_of_output)
        self.output_bias = np.zeros(self.no_of_output)
        self.actual_target_values = []
        self.total = 0
        self.learningRate = 0.0262
        self.setup_center()
        self.setup_sigma_spread_radius()
        self.set_up_hidden_to_ouput_weight()
        self.set_up_output_bias()

    def setup_center(self):
        """Setup center using clustering ,for now just randomize between 0 and 1"""
        # print("Setup center")
        for i in range(self.no_of_hidden):
            self.centroid[i] = np.random.uniform(0, 1, self.no_of_input)

    def setup_sigma_spread_radius(self):
        # print("Setup Sigma spread radius")
        for i in range(self.no_of_hidden):
            center = self.centroid[i]
            self.sigma[i] = self.set_up_sigma_for_center(center)
            # print("Sigma i",i, self.sigma[i])

    def set_up_sigma_for_center(self, center):
        # print("Get sigma for center")
        p = self.no_of_hidden / 3
        sigma = 0
        distances = [0 for i in range(self.no_of_hidden)]
        for i in range(self.no_of_hidden):
            distances[i] = self.euclidean_distance(center, self.centroid[i])
            # print("Distance ", i, distances[i])
        sum = 0
        for i in range(int(p)):
            nearest = self.get_smallest_index(distances)
            distances[nearest] = float("inf")

            neighbour_centroid = self.centroid[nearest]
            for j in range(len(neighbour_centroid)):
                sum += (center[j] - neighbour_centroid[j]) ** 2

        sigma = sum / p
        sigma = math.sqrt(sigma)
        #return random.uniform(0, 1) * 6
        return sigma

    @staticmethod
    def euclidean_distance( x, y):
        return np.linalg.norm(x-y)

    @staticmethod
    def get_smallest_index( distances):
        min_index = 0
        for i in range(len(distances)):
            if (distances[min_index] > distances[i]):
                min_index = i
        return min_index

    def set_up_hidden_to_ouput_weight(self):
        print("Setup hidden to output weight")
        self.hidden_to_output_weight = np.random.uniform(0, 1, (self.no_of_hidden, self.no_of_output))

        print("Hiden to output weight ", self.hidden_to_output_weight)

    def set_up_output_bias(self):
        print("Setup output bias")
        self.output_bias = np.random.uniform(0, 1, self.no_of_output)

    # train n iteration
    def train(self, n):
        for i in range(n):
            error = self.pass_one_epoch()
            print("Iteration ", i, " Error ", error)

        return error

    # Train an epoch and return total MSE
    def pass_one_epoch(self):
        # print("Pass one epoch")
        all_error = 0
        all_index = []
        for i in range(len(self.data.patterns)):
            all_index.append(i)
        # print("All index ",all_index)

        for i in range(len(self.data.patterns)):
            random_index = (int)(random.uniform(0, 1) * len(all_index))
            # print("Random index ",random_index, " Len ", len(all_index))
            """Get a random pattern to train"""
            pattern = self.data.patterns[random_index]
            del all_index[random_index]

            input = pattern.input
            self.actual_target_values = pattern.output
            self.pass_input_to_network(input)

            error = self.get_error_for_pattern()
            all_error += error
            self.gradient_descent()

        all_error = all_error / (len(self.data.patterns))
        return all_error

    def pass_input_to_network(self, input):
        self.input = input
        self.pass_to_hidden_node()
        self.pass_to_output_node()

    def pass_to_hidden_node(self):
        # print("Pass to hidden node")
        self.hidden_output = np.zeros(self.no_of_hidden)
        for i in range(len(self.hidden_output)):
            euclid_distance = self.euclidean_distance(self.input, self.centroid[i]) ** 2
            self.hidden_output[i] = math.exp(- (euclid_distance / (2 * self.sigma[i] * self.sigma[i])))

            # print("Hdiden node output ",self.hidden_output)

    def pass_to_output_node(self):
        # print("Pass to output node")
        self.output = [0 for i in range(self.no_of_output)]
        total = 0
        for i in range(self.no_of_output):
            output_value = 0
            for j in range(self.no_of_hidden):
                self.output[i] += self.hidden_to_output_weight[j][i] * self.hidden_output[j]
        """Normalize """
        for i in range(self.no_of_output):
            total += self.output[i]
        for i in range(self.no_of_output):
            if (self.output[i] != 0):
                self.output[i] = self.output[i] / total
        self.total = total

    # Compute error for the pattern
    def get_error_for_pattern(self):
        error = 0
        for i in range(len(self.output)):
            error += (self.actual_target_values[i] - self.output[i]) ** 2
        return error

    # Weight update by gradient descent algorithm
    def gradient_descent(self):
        # compute the error of output layer
        self.mean_error = 0
        self.error_of_output_layer = [0 for i in range(self.no_of_output)]
        for i in range(self.no_of_output):
            self.error_of_output_layer[i] = (float)(self.actual_target_values[i] - self.output[i])
            e = (float)(self.actual_target_values[i] - self.output[i]) ** 2 * 0.5
            self.mean_error += e

        # Adjust hidden to output weight
        for o in range(self.no_of_output):
            for h in range(self.no_of_hidden):
                delta_weight = self.learningRate * self.error_of_output_layer[o] * self.hidden_output[h]
                self.hidden_to_output_weight[h][o] += delta_weight

        # For bias
        for o in range(self.no_of_output):
            delta_bias = self.learningRate * self.error_of_output_layer[o]
            self.output_bias[o] += delta_bias

        # Adjust center , input to hidden weight
        for i in range(self.no_of_input):
            for j in range(self.no_of_hidden):
                summ = 0
                for p in range(self.no_of_output):
                    summ += self.hidden_to_output_weight[j][p] * (self.actual_target_values[p] - self.output[p])

                second_part = (float)((self.input[i] - self.centroid[j][i]) / math.pow(self.sigma[j], 2))
                delta_weight = (float)(self.learningRate * self.hidden_output[j] * second_part * summ)
                self.centroid[j][i] += delta_weight

        # Adjust sigma and spread radius
        for i in range(self.no_of_input):
            for j in range(self.no_of_hidden):
                summ = 0
                for p in range(self.no_of_output):
                    summ += self.hidden_to_output_weight[j][p] * (self.actual_target_values[p] - self.output[p])

                second_part = (float)(
                    (math.pow((self.input[i] - self.centroid[j][i]), 2)) / math.pow(self.sigma[j], 3));
                delta_weight = (float)(0.1 * self.learningRate * self.hidden_output[j] * second_part * summ);
                self.sigma[j] += delta_weight
        return self.mean_error

    def get_accuracy_for_training(self):
        correct = 0
        for i in range(len(self.data.patterns)):
            pattern = self.data.patterns[i]
            self.pass_input_to_network(pattern.input)
            n_output = self.output
            act_output = pattern.output
            n_neuron = self.get_fired_neuron(n_output)
            a_neuron = self.get_fired_neuron(act_output)

            if n_neuron == a_neuron:
                correct += 1
        accuracy = (float)(correct / len(self.data.patterns)) * 100
        return accuracy

    def get_fired_neuron(self, output):
        max = 0
        for i in range(len(output)):
            if (output[i] > output[max]):
                max = i
        return max


"""Create test data """
p1 = Pattern(1, [0, 0], [1, 0])
p2 = Pattern(2, [0, 1], [0, 1])
p3 = Pattern(3, [1, 0], [0, 1])
p4 = Pattern(4, [1, 1], [1, 0])

patterns = [p1, p2, p3, p4]
classLabels = ['0', '1']
data = Data(patterns, classLabels)
rbf = RBFNetwork(2, 6, 2, data)
mse = rbf.train(1500)
accuracy = rbf.get_accuracy_for_training()
print("Total accuracy is ", accuracy)
print("Last MSE ",mse)

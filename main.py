import neuralnetwork as nn
import numpy as np
import matplotlib.pyplot as plt

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

# display images
plt.imshow(training_images[4].reshape(28, 28), cmap='gray')
plt.show()

layer_sizes = (784, 5, 10)

net = nn.NeuralNetwork(layer_sizes)
# prediction = net.predict(training_images)
print(net.print_accuracy(training_images, training_labels))

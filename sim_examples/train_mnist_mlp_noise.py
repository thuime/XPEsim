from __future__ import print_function, division
import numpy as np
import simulator
from netlang import functional
from netlang.core import compile
from netlang.data_provider import Batch
from netlang.dataset import mnist
from netlang.module.activation import ReLU, Softmax
from netlang.module.linear import Linear
from netlang.module.loss import CrossEntropy
from netlang.module.metric import OneHot, Accuracy
from netlang.module.optimizer import SGD
from netlang.module.structure import Sequential
from netlang.preprocess import cast
from netlang.module.noise import NoiseLinear

# TRAING PROCESS

original_net = Sequential([
    Linear([784, 500]), ReLU(),
    Linear([500, 100]), ReLU(),
    Linear([100, 10]), Softmax()
], name='mnist-500-100')

try:
    original_net.load('./data/mnist-500-100.npz')
except:
    pass

net = Sequential([
    NoiseLinear(original_net.submodule('Linear0'), weight_bits=8, noise=0.07,name='NoiseLinear0'), ReLU(),
    NoiseLinear(original_net.submodule('Linear1'), weight_bits=8, noise=0.07,name='NoiseLinear1'), ReLU(),
    NoiseLinear(original_net.submodule('Linear2'), weight_bits=8, noise=0.07,name='NoiseLinear2'), Softmax()
], name='mnist-500-100')

x = functional.placeholder('x', dims=2)
y = functional.placeholder('y', dims=1, dtype='int32')

y_ = net.forward(x)

loss = CrossEntropy().minimize(y_, OneHot(10).turn(y))
accuracy = Accuracy().measure(y_, y)

updates = SGD(learning_rate=0.1, momentum=0.9).updates(net.parameters(), net.differentiate(loss))

train_op = compile(inputs=[x, y], outputs=[accuracy], updates=updates)
test_op = compile(inputs=[x, y], outputs=[accuracy])

batch_size = 100

train_set = mnist.subset('train')
train_provider = Batch(train_set, batch_size, y_preprocess=[cast('int32')])

print('Start training process')

for epoch in xrange(2):
    train_accuracies = []
    for i in xrange(60000 // batch_size):
        x, y = train_provider.get()
        accuracy, = train_op(x, y)
        train_accuracies.append(accuracy)
    train_accuracy = sum(train_accuracies) / len(train_accuracies)

    test_set = mnist.subset('test')
    test_provider = Batch(test_set, batch_size, y_preprocess=[cast('int32')])
    test_accuracies = []
    for j in xrange(10000 // batch_size):
        x, y = test_provider.get()
        accuracy, = test_op(x, y)
        test_accuracies.append(accuracy)
    test_accuracy = sum(test_accuracies) / len(test_accuracies)

    print('Epoch %d, train_accuracy %0.5f, test_accuracy %0.5f' % (epoch, train_accuracy, test_accuracy))

    net.save('./data/mnist-500-100-noise.npz')



# Hardware Evaluation
weightsdata_file = './data/mnist-500-100-noise.npz'
testdata_file = './data/dataset/mnist/test.npy'

params = simulator.Parameterinput() # Read parameters from simconfig

net = [
    ['Linear', [784, 500], 'ReLU'],
    ['Linear', [500, 100], 'ReLU'],
    ['Linear', [100, 10], 'Softmax']
]

HWsim = simulator.SystemSim(params)

# Load the weights
weights = np.load(weightsdata_file)["arr_0"].item()

# Load the testdata
testdata = np.load(testdata_file)
batchsize = 400
images = testdata[0:batchsize, 0]
labels = testdata[0:batchsize, 1]

# Algorithm evaluation
HWsim.apply(net, weights, images, labels)
HWsim.show()

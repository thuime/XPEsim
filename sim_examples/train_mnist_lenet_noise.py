import sys
from netlang import functional
from netlang.core import Scope, Args, compile
from netlang.data_provider import Batch
from netlang.dataset import mnist
from netlang.module.activation import ReLU, Softmax
from netlang.module.conv import Conv2d, MaxPool2d
from netlang.module.linear import Linear
from netlang.module.loss import CrossEntropy
from netlang.module.metric import OneHot, Accuracy
from netlang.module.noise import NoiseConv2d, NoiseLinear
from netlang.module.optimizer import SGD
from netlang.module.structure import Sequential
from netlang.preprocess import cast
from netlang.module.round import Round

weight_bits = 8
io_bits = 8
noise = 0.03

with Scope(Args(padding='valid')):
    orignal_net = Sequential([
        Conv2d([20, 1, 5, 5]), ReLU(), MaxPool2d([2, 2], 2),
        Conv2d([50, 20, 5, 5]), ReLU(), MaxPool2d([2, 2], 2),
        Linear([800, 1250]), ReLU(),
        Linear([1250, 120]), ReLU(),
        Linear([120, 10]), Softmax()
    ], name='mnist-lenet')
orignal_net.load('./data/mnist-lenet.npz')
with Scope(Args(padding='valid', weight_bits=weight_bits, io_bits=io_bits, noise=noise)):
    net = Sequential([
        NoiseConv2d(orignal_net.submodule('Conv2d0')), Round(), MaxPool2d([2, 2], 2),
        NoiseConv2d(orignal_net.submodule('Conv2d1')), Round(), MaxPool2d([2, 2], 2),
        NoiseLinear(orignal_net.submodule('Linear0')), Round(),
        NoiseLinear(orignal_net.submodule('Linear1')), Round(),
        NoiseLinear(orignal_net.submodule('Linear2')), Softmax()
    ], name='mnist-lenet.low')

x = functional.placeholder('x', dims=2)
y = functional.placeholder('y', dims=1, dtype='int32')

y_ = net.forward(functional.reshape(x, (-1, 1, 28, 28)))

loss = CrossEntropy().minimize(y_, OneHot(10).turn(y))
accuracy = Accuracy().measure(y_, y)

updates = SGD(learning_rate=0.05, momentum=0.9).updates(net.parameters(), net.differentiate(loss))+ net.updates()

print('Begin compile')
train_op = compile(inputs=[x, y], outputs=[accuracy], updates=updates)
print('Compiled train_op')
test_op = compile(inputs=[x, y], outputs=[accuracy])
print('Compiled test_op')

batch_size = 100

train_set = mnist.subset('train')
train_provider = Batch(train_set, batch_size, y_preprocess=[cast('int32')])

print('Start training')
testa=[]
for epoch in xrange(20):
    train_accuracies = []
    for i in xrange(60000 / batch_size):
        x, y = train_provider.get()
        accuracy, = train_op(x, y)
        train_accuracies.append(accuracy)
    train_accuracy = sum(train_accuracies) / len(train_accuracies)

    test_set = mnist.subset('test')
    test_provider = Batch(test_set, batch_size, y_preprocess=[cast('int32')])
    test_accuracies = []
    for j in xrange(10000 / batch_size):
        x, y = test_provider.get()
        accuracy, = test_op(x, y)
        test_accuracies.append(accuracy)
    test_accuracy = sum(test_accuracies) / len(test_accuracies)
    testa.append(test_accuracy)
    print('Epoch %d, train_accuracy %0.5f, test_accuracy %0.5f' % (epoch, train_accuracy, test_accuracy))
    net.save('./data/mnist-lenet-noise.npz')


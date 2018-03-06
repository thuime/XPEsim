from netlang.module.concept import Forwardable, Parameterable, Updateful


class Sequential(Forwardable, Parameterable, Updateful):
    def __init__(self, layers, name=None):
        super(Sequential, self).__init__(name)
        self._layers = layers
        for layer in layers:
            self.add_module(layer)

    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return x
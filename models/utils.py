from math import floor

class LayerParams:
    def __init__(self, n_channels=None, kernel_size=1, padding=0, stride=1, dilation=1):
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def output_shape(self, c_in, h_in, w_in):
        if self.n_channels is None:
            c=c_in
        else:
            c = self.n_channels
        h = floor(1 + (h_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride)
        w = floor(1 + (w_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride)
        return c, h, w

    @property
    def params(self):
        return {
            "out_channels": self.n_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding ": self.padding,
            "dilation ": self.dilation
        }

class LayerList:
    def __init__(self, *layers):
        self.layers = [layer for layer in layers]

    def output_shape(self, c_in, h_in, w_in):
        shape = (c_in, h_in, w_in)
        for layer in self.layers:
            shape = layer.output_shape(*shape)
        return shape
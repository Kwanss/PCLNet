import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.Gates = nn.Conv2d(self.input_channels + self.hidden_channels , 4*self.hidden_channels,
                self.kernel_size, 1, self.padding, bias=True)

    def forward(self, x, h, c):

        stacked_inputs = torch.cat((x, h), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across the channel dimension
        xi, xf, xo, xg = gates.chunk(4, 1)

        # apply sigmoid non linearity
        xi = torch.sigmoid(xi)
        xf = torch.sigmoid(xf)
        xo = torch.sigmoid(xo)
        xg = torch.tanh(xg)

        # compute current cell and hidden state
        c = (xf * c) + (xi * xg)
        h = xo * torch.tanh(c)
        
        return h, c

    def init_hidden(self, batch_size, hidden, shape):
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):  
        #input : (num, seq_len, channel, H,W)
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[:, step, :,:,:]   
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                            shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)
        return outputs, (x, new_c)


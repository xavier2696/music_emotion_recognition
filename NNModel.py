import torch


class NeuralNet(torch.nn.Module):
    def __init__(self, d_in, d_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(NeuralNet, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, 128)
        self.linear2 = torch.nn.Linear(128, d_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        #h_softmax = torch.nn.functional.softmax(self.linear1(x), dim=0)
        h_relu1 = self.linear1(x).clamp(min=0) #clamp(min=0) is ReLu, torch.nn.functional.relu(self.linear1(x))
        #h_relu2 = self.linear2(h_relu1).clamp(min=0)
        y_pred = self.linear2(h_relu1)
        return y_pred
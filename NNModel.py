import torch


class NeuralNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(NeuralNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 256)
        self.linear2 = torch.nn.Linear(256, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        #h_softmax = torch.nn.functional.softmax(self.linear1(x), dim=0)
        h_relu = self.linear1(x).clamp(min=0) #clamp(min=0) is ReLu, torch.nn.functional.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred
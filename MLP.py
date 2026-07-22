import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size=(128,64), activation_func = "relu", batch_norm = False, dropout_rate = 0.0):

        super().__init__()

        activation = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU    
        }

        activation_class = activation.get(activation_func, nn.ReLU)

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layer_size:
            layers.append(nn.Linear(prev_size,hidden_size))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_class())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size,output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, input_batch):
        return self.network(input_batch)

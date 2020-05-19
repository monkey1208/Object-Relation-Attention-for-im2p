import torch
import torch.nn as nn

def build_layers(layers_dim, dropout_rate=0):
    layers = []
    for i in range(len(layers_dim) - 1):
        input_dim = layers_dim[i]
        output_dim = layers_dim[i+1]
        layers.append(nn.Linear(input_dim, output_dim))
        #layers.append(nn.ReLU())
        
        if i != (len(layers_dim) - 2): # not last layer
            #layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            #layers.append(nn.LeakyReLU())
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()

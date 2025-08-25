from ..layers.layers import Activation
import numpy as np

class Sequential:
    def __init__(self, *layers):
        self.layers = layers
        self.compile_layers(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def compile_layers(self, layers):
        for i, layer in enumerate(layers):
            assert self.first_layer_has_input_size(i), f"Layer 0 must have input_size defined."
            if self.is_activation_layer(layer):
                layer.n = layers[i-1].n
                continue
            layer.input_size = layer.input_size if self.is_first_layer(i) else layers[i-1].n
            layer.compile()

    def is_first_layer(self, index):
        return index == 0
    
    def is_activation_layer(self, layer):
        return isinstance(layer, Activation)
    
    def first_layer_has_input_size(self, index):
        if self.is_first_layer(index):
            return self.layers[index].input_size is not None
        return True
    
    def predict(self, x):
        x = x.T
        with np.errstate(divide='ignore', invalid='ignore'):
            for layer in self.layers:
                x = layer(x)
        return x.data[0]
    
    def train(self, X, Y, optimizer, loss_fn, epochs=100, print_every=50):
        optimizer.model = self
        for epoch in range(epochs):
            logits = self(X.T)
            loss = loss_fn(logits, Y)

            loss.backward()
            optimizer.step()
            
            if epoch % print_every == 0:
                print(f"(Epoch {epoch}) Loss: {loss.data}")

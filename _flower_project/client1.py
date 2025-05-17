import flwr as fl

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        # model train
        return model.get_weights(), len(data), {}
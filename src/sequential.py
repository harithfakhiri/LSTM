
class Sequential:
    def __init__(self, LSTM, Flatten, Dense):
        self.LSTMLayers = LSTM
        self.FlattenLayers = Flatten
        self.DenseLayers = Dense

    def predict(self, column, columnName):
        output_lstm = self.LSTMLayers.predict(column)

        output_flatten = self.FlattenLayers.forward(output_lstm)

        curr_out_dense = output_flatten
        for idx, dense in enumerate(self.DenseLayers):
            dense.set_input(curr_out_dense)
            dense.initialize_weight()
            curr_out_dense = dense.forward(curr_out_dense)

        self.predicted = curr_out_dense
        print(f"PREDICTED {columnName}: {self.predicted}")
        return self.predicted
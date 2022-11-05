import numpy as np
class Sequential:
    def __init__(self, LSTM, Flatten, Dense):
        self.LSTMLayers = LSTM
        self.FlattenLayers = Flatten
        self.DenseLayers = Dense
        self.summary_ = []

    def train(self, column, columnName):
        curr_out_lstm = column
        for idx, lstm in enumerate(self.LSTMLayers):
            output_lstm = lstm.train(curr_out_lstm)
            self.summary_.append([lstm.type_(idx), np.array(output_lstm).shape, lstm.getParam(len(output_lstm.shape))])

        output_flatten = self.FlattenLayers.forward(output_lstm)
        self.summary_.append([self.FlattenLayers.type_, np.array(output_flatten).shape, 0])
        curr_out_dense = output_flatten
        for idx, dense in enumerate(self.DenseLayers):
            dense.set_input(curr_out_dense)
            dense.initialize_weight()
            curr_out_dense = dense.forward(curr_out_dense)
            # self.summary_.append([dense.type_(idx), np.array(curr_out_dense).shape, (self.summary_[-1][1][-1]+1)*np.array(curr_out_dense).shape[-1]])

        self.predicted = curr_out_dense
        print(f"PREDICTED {columnName}: {self.predicted}")
        return self.predicted
    
    def predict(self, column, columnName, days:int):
        self.predicted = []
        for t in range (days):
            curr_out_lstm = []
            for idx, lstm in enumerate(self.LSTMLayers):
                output_lstm = lstm.predict(curr_out_lstm, t, column)

            output_flatten = self.FlattenLayers.forward(np.array([output_lstm]))
            curr_out_dense = output_flatten
            for idx, dense in enumerate(self.DenseLayers):
                dense.set_input(curr_out_dense)
                dense.initialize_weight()
                curr_out_dense = dense.forward(curr_out_dense)

            self.predicted.append(curr_out_dense)
        
        print(f"PREDICTED TEST {columnName}: \n{self.predicted}")

        return self.predicted
    
    def summary(self):
        total = 0
        print('---------------------------------------------------------------')
        print('Layer (type)    Output Shape    Param #')
        print('===============================================================')
        for s in self.summary:
            total += int(s[2])
            print((str(s[0])+'{between}'+str(s[1])+'{between}'+str(s[2])).format(between=' '*8))
            print('---------------------------------------------------------------')
            print('===============================================================')
            print('Total params: '+str(total))
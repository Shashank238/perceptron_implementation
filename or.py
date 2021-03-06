from calendar import EPOCH
from statistics import mode
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data,eta,epochs):
    

    df = pd.DataFrame(OR)
    print(df)

    X,y = prepare_data(df)


    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename="or.model")
    save_plot(df, "or.png", model)

if __name__ == '__main__':
    OR = {

        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    main(OR,ETA,EPOCHS)
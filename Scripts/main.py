import Models

if __name__ == "__main__":

    modelFW = Models.SimpleCNNModel()

    modelFW.load_dataset()
    modelFW.view_train_data_sample()
    modelFW.train_model()
    modelFW.evaluate_model()
    print("Hello World!")
import Models

if __name__ == "__main__":

    modelFW = Models.SimpleModel()

    modelFW.load_dataset()
    modelFW.train_model()
    modelFW.evaluate_model()
    print("Hello World!")
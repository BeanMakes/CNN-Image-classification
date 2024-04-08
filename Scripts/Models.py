import tensorflow as tf

class SimpleModel:

    def __init__(self) -> None:

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._build_model()
        self.model.compile(optimizer='adam',
              loss=self.loss_fn,
              metrics=['accuracy'])
        pass

    def load_dataset(self, predefined = True):
        if predefined:
            mnist = tf.keras.datasets.mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
    


    def _build_model(self):
        self.model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=5)

    def evaluate_model(self):
        self.model.evaluate(self.x_test,  self.y_test, verbose=2)

    def evaluate_model_with_prob(self):
        probability_model = tf.keras.Sequential([
        self.model,
        tf.keras.layers.Softmax()
        ])
        probability_model(self.x_test[:5])
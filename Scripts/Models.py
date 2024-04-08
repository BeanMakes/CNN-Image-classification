import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class SimpleCNNModel:

    def __init__(self) -> None:
        self._build_model()
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        

    def load_dataset(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
    
    def view_model_summary(self):
        self.model.summary()

    def view_train_data_sample(self):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i])
            # The CIFAR labels happen to be arrays, 
            # which is why you need the extra index
            plt.xlabel(class_names[self.train_labels[i][0]])
        plt.show()
    
    def train_model(self):
        self.modelHistory = self.model.fit(self.train_images, self.train_labels, epochs=10, 
                    validation_data=(self.test_images, self.test_labels))
    
    def evaluate_model(self):
        plt.plot(self.modelHistory.history['accuracy'], label='accuracy')
        plt.plot(self.modelHistory.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        plt.show()

        test_loss, test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)

    def _build_model(self):
        self.model = models.Sequential()

        #CNN layers
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        #Output layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))


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

    def _build_model(self):
        self.model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])
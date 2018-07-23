import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.summary()

# set optimizer, learning rate, loss, metrics (string list to evaluate model)
model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# X features
X_features = []
# y labels
y_labels = []

# train the model
# model.fit(X_features, y_labels, batch_size=10, epochs=20, shuffle=True, verbose=2)

# or, train the model with a validation set
model.fit(X_features, y_labels, validation_split=0.1, batch_size=10, epochs=20, shuffle=True, verbose=2)

# don't forget to scale the test samples
scaled_test_samples = []  
# predict on the test samples, which will return probabilities for each class
predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

# predict classes
rounded_predictions = model.predict_classes(scaled_test_samples, batch_size=10, verbose=0)

# to save model architecture and weights, config (loss, optimizer), state of optimizer to resume testing
model.save('file_path_to_save.h5')

# to load model
from keras.models import load_model
new_model = load_model('file_path_to_save.h5')
new_model.summary()  # this will show the model architecture just as before

new_model.get_weights()  # will show the weights as it was left off before

# to save only just the architecture
json_string = model.to_json()

# to reconstruct from json
from keras.models import model_from_json
model_arch = model_from_json(json_string)

# to save just the weights of the model
model.save_weights('path_to_save.h5')

# create a new model, but then need to recreate the architecture of original model,
# then load those weights
model2 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

# loading original weights
model2.load_weights('path_to_save.h5')

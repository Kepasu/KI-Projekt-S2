import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


merged_dfs = np.load('city_dict.npy', allow_pickle='TRUE').item()

X = []
y = []


for df in merged_dfs.values():
    temperature = df['tmax']
    precipitation = df['prcp']
    fire_occurred = df['Fire']

    X.extend(list(zip(temperature, precipitation)))
    y.extend(fire_occurred)


X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)


# Define the neural network architecture
input_size = 2
hidden_size = 4
output_size = 1  # Assuming the output is binary (0 or 1)

model = Sequential()
model.add(Dense(hidden_size, input_shape=(input_size,), activation='sigmoid'))
model.add(Dense(output_size, activation='sigmoid'))
model.add(Dense(output_size, activation='sigmoid'))
model.add(Dense(output_size, activation='sigmoid'))
model.add(Dense(output_size, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 100
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

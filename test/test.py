from Model import Model
import numpy as np
model = Model()

inputs = np.random.uniform(-1.0, 1.0, size=(1, 1, 20))
output = model.forward(inputs)
print(output)
model.details()
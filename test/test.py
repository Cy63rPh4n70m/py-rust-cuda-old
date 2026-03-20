from TestModel import TestModel
import numpy as np

model = TestModel(lr=0.001, l2=0.0001, optimizer="adamw")
model.add_dense_layer("layer1", 10, 15, 1, 1, "silu")
model.add_dense_layer("layer2", 15, 10, 1, 1, "silu")
model.add_dense_layer("layer3", 10, 5, 1, 1, "sigmoid")

model.get_details()
model.load_weights("test_weights/test_weights.json")
np.random.seed(0)
inputs = np.random.uniform(0, 1, size=(64, 10))
outputs = np.random.randint(0, 2, size=(64, 5))

epochs = 1000
for i in range(epochs):
    total_loss = 0
    for x, y in zip(inputs, outputs):
        pred = model.forward(x)
        loss, loss_grads = model.calc_bce_loss(pred, y)
        model.backward(loss_grads)
        model.update_params()

        total_loss += loss
    
    print(i, total_loss / len(inputs))

model.save_weights("test_weights/test_weights.json")
model.get_details()

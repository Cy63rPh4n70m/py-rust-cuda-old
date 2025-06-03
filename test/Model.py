from nnpackage.model import CudaModel
from nnpackage.layers import LayerAdder
from nnpackage.train_infer import Trainer

class Model:
    def __init__(self):
        self.model = CudaModel("./nnpackage/backend/rust_backend.dll")
        self.trainer = Trainer(self.model)
        self.layer_adder = LayerAdder(self.model)

        self.layer_adder.add_dense_cuda(n_in=20, n_out=10, batch=1, rows=1, use_bias=True, id="layer0")
        self.layer_adder.add_activation_cuda(batch=1, rows=1, cols=10, activation="silu", id="act0")
        self.layer_adder.add_dense_cuda(n_in=10, n_out=5, batch=1, rows=1, use_bias=True, id="layer1")
        self.layer_adder.add_activation_cuda(batch=1, rows=1, cols=10, activation="silu", id="act1")
        self.layer_adder.add_dense_cuda(n_in=5, n_out=3, batch=1, rows=1, use_bias=True, id="layer2")
        self.layer_adder.add_softmax_cuda(batch=1, rows=1, cols=3, temperature=1.0, id="act2")

    def forward(self, inputs):
        ptr = self.trainer.pass_input("input", inputs)
        ptr = self.trainer.forward("layer0", ptr)
        ptr = self.trainer.forward("act0", ptr)
        ptr = self.trainer.forward("layer1", ptr)
        ptr = self.trainer.forward("act1", ptr)
        ptr = self.trainer.forward("layer2", ptr)
        ptr = self.trainer.forward("act2", ptr)

        output = self.trainer.pass_output("output", ptr, 1, 1, 3)

        return output

    def details(self):
        self.model.details()
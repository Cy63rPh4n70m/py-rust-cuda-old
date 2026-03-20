from nnpackage.model import CudaModel
from nnpackage.layers import LayerAdder
from nnpackage.train_infer import Trainer

class TestModel:
    """
    Class for building models to test the Rust and CUDA backend libraries.
    """
    def __init__(self, lr: float, l2: float, optimizer: str = "adamw"):
        self.model = CudaModel("./nnpackage/backend/rust_backend.dll")
        self.trainer = Trainer(self.model)
        self.layer_adder = LayerAdder(self.model)

        self.layers: list[str] = []
        self.layer_shapes: list[tuple[int]] = []

        self.last_shape: tuple[int] = None

        self.lr = lr
        self.l2 = l2
        self.optimizer = optimizer
    
    def add_dense_layer(
            self, layer_id: str, in_dim: int, out_dim: int, 
            batch_dim: int, row_dim: int, activation: str, use_bias: bool = True
        ):
        dense_id = self.layer_adder.add_dense(in_dim, out_dim, batch_dim, row_dim, use_bias, layer_id)
        activation_id = self.layer_adder.add_activation(batch_dim, row_dim, out_dim, activation, layer_id + "_act")
        self.layers.append(dense_id)
        self.layers.append(activation_id)
        self.layer_shapes.append((in_dim, out_dim))
        self.layer_shapes.append((out_dim,))
        
    def forward(self, inputs):
        ptr = self.trainer.pass_input("input", inputs)

        for layer_id in self.layers:
            ptr = self.trainer.forward(layer_id, ptr)

        self.last_shape = self.layer_shapes[-1]
        output = self.trainer.pass_output("output", ptr, 1, 1, self.last_shape[0])

        return output
    
    def calc_bce_loss(self, pred, actual):
        loss_val, loss_array = self.trainer.calc_loss_and_grads(
            loss_type="bce_loss", pred=pred, actual=actual,
            flattened_len=self.last_shape[0]
        )

        return loss_val, loss_array
    
    def backward(self, loss_grads):
        self.trainer.pass_backward_grad_to("output", loss_grads)
        self.trainer.backward()

    def update_params(self):
        for layer_id in self.layers:
            self.trainer.update_params(layer_id, self.optimizer, self.lr, self.l2)

    def get_details(self):
        self.model.get_details()
    
    def save_weights(self, path: str):
        self.model.save_weights(path)
    
    def delete_memory(self):
        self.model.delete_memory()
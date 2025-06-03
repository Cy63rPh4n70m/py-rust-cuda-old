import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model import CudaModel

class Trainer:
    def __init__(self, model_obj: "CudaModel"):
        self.lib_funcs = model_obj.lib_funcs
        self.model = model_obj.model

    def pass_input(self, input_id: str, array: np.ndarray):
        array = array.astype(np.float32)
        flattened_array = array.flatten()
        array_len = len(flattened_array)

        traverse_ptr = self.lib_funcs.get("pass_to_input")(self.model, input_id.encode(), flattened_array, array_len)
        return traverse_ptr
    
    def pass_output(self, input_id: str, ptr, batch: int, rows: int, cols: int):
        array_ptr = self.lib_funcs.get("pass_to_output")(
            self.model, input_id.encode(), ptr, batch * rows * cols
        )
        array = np.array([array_ptr[i] for i in range(batch * rows * cols)], dtype=np.float32)
        array = array.reshape((batch, rows, cols))

        return array

    def ce_loss(self, output_id: str, target_classes: np.ndarray, batch: int, rows: int, cols: int):
        '''
        Built-in crossentropy loss for softmax (gradients are automatically passed for backpropagation)
        '''
        target_classes = target_classes.astype(np.float32)
        loss_val_array = self.lib_funcs.get("ce_loss")(
            self.model, output_id.encode(), 
            target_classes, 
            batch, rows, cols
        )

        loss_val_array = np.array([loss_val_array[i] for i in range(batch * rows * 1)], dtype=np.float32)
        loss_val_array = loss_val_array.reshape(batch, rows, 1)

        return loss_val_array
    
    def pass_backward_grad_to(self, id: str, array: np.ndarray):
        array = array.astype(np.float32)
        flattened_array = array.flatten()
        array_len = len(flattened_array)

        self.lib_funcs.get("pass_to_output_grad")(self.model, id.encode(), flattened_array, array_len)

    def forward(self, layer_id: str, in_ptr, weight_ptr=None):
        new_ptr = self.lib_funcs.get("forward")(self.model, layer_id.encode(), in_ptr, weight_ptr)
        return new_ptr
        
    def backward(self):
        self.lib_funcs.get("backward")(self.model)
        
    def update_params(self, layer_id: str, optimizer: str, lr: float, l2: float, alpha: float = 0.9, beta: float = 0.999):
        if optimizer.lower() == "sgd":
            optimizer_type = 0
        elif optimizer.lower() == "adamw":
            optimizer_type = 1
        else:
            print(f"Error: {optimizer} not allowed, (sgd or adamw)")
            raise SystemExit()
        
        self.lib_funcs.get("update_params")(self.model, layer_id.encode(), optimizer_type, lr, l2, alpha, beta)
        
    def calc_loss_and_grads(self, loss_type: str, pred: np.ndarray, actual: np.ndarray, flattened_len: int) -> tuple[float, np.ndarray]:
        pred = pred.astype(np.float32).flatten()
        actual = actual.astype(np.float32).flatten()
            
        loss_val = self.lib_funcs.get("loss")(loss_type.encode(), pred, actual, flattened_len)
        loss_grads_ptr = self.lib_funcs.get("loss_grads")(loss_type.encode(), pred, actual, flattened_len)
            
        loss_grads = np.array([loss_grads_ptr[i] for i in range(flattened_len)], dtype=np.float32)
        #loss_grads = loss_grads.reshape(self.output_shape)
        self.lib_funcs.get("free_array")(loss_grads_ptr, flattened_len)
        return loss_val, loss_grads
    
    def set_dropout(self, use_dropout: bool):
        self.lib_funcs.get("set_dropout")(self.model, use_dropout)
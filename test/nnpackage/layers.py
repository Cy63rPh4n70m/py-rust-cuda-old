from typing import TYPE_CHECKING
from ctypes import string_at
if TYPE_CHECKING:
    from model import CudaModel

class LayerAdder:
    def __init__(self, model_obj: "CudaModel"):
        self.lib_funcs = model_obj.lib_funcs
        self.model = model_obj.model
    
    def add_dense(self, n_in: int, n_out: int, batch: int, rows: int, use_bias: bool, id: str) -> str:
        layer_id = self.lib_funcs.get("add_dense_cuda_layer")(self.model, n_in, n_out, batch, rows, use_bias, id.encode())
        return layer_id.decode("utf-8")

    def add_activation(self, batch: int, rows: int, cols: int, activation: str, id: str, scale: float = 1.0) -> str:
        layer_id = self.lib_funcs.get("add_activation_cuda_layer")(self.model, batch, rows, cols, activation.encode(), scale, id.encode())
        return layer_id.decode("utf-8")
    
    def add_l2norm(self, batch: int, rows: int, cols: int, id: str) -> str:
        layer_id = self.lib_funcs.get("add_l2norm_cuda_layer")(self.model, batch, rows, cols, id.encode())
        return layer_id.decode("utf-8")
    
    def add_embedding2d(self, vocab_size: int, embedding_len: int, seq_len: int, id: str) -> str:
        layer_id = self.lib_funcs.get("add_embedding_2d_cuda_layer")(self.model, vocab_size, embedding_len, seq_len, id.encode())
        return layer_id.decode("utf-8")
    
    def add_conv2d(
            self, n_filters: int, filter_dim: int, strides: int, 
            batch: int, rows: int, cols: int, id: str
        ) -> str:
        
        layer_id = self.lib_funcs.get("add_conv2d_cuda_layer")(
            self.model, n_filters, filter_dim, strides, batch, rows, cols, id.encode()
        )

        return layer_id.decode("utf-8")
        
    def add_softmax(self, batch: int, rows: int, cols: int, temperature: float, id: str) -> str:
        layer_id = self.lib_funcs.get("add_softmax_cuda_layer")(self.model, batch, rows, cols, temperature, id.encode())
        return layer_id.decode("utf-8")
    
    def add_dropout(self, batch: int, rows: int, cols: int, dropout_rate: float, id: str) -> str:
        layer_id = self.lib_funcs.get("add_dropout_cuda_layer")(self.model, batch, rows, cols, dropout_rate, id.encode())
        return layer_id.decode("utf-8")
    
    def add_elementwise(
            self, batch: int, rows: int, cols: int, scale_range: float, 
            op: str, dropout_rate: float, id: str, activation_str: str = "linear", 
            activation_scale: float = 1.0
        ) -> str:
        if "add" in op.lower():
            op = 0
        elif "mul" in op.lower():
            op = 1
        else:
            print(f"Error: {op.lower()} not valid (only 'add' or 'mul' allowed)")
            raise SystemExit
        
        layer_id = self.lib_funcs.get("add_elementwise_cuda_layer")(
            self.model, batch, rows, cols, scale_range, op, dropout_rate, 
            id.encode(), activation_str.encode(), activation_scale
        )
        return layer_id.decode("utf-8")

    def add_broadcast(
            self, out_batch: int, out_rows: int, out_cols: int, 
            axis: int, id: str
        ) -> str:
        layer_id = self.lib_funcs.get("add_broadcast_cuda_layer")(self.model, out_batch, out_rows, out_cols, axis, id.encode())
        return layer_id.decode("utf-8")
    
    def add_sum(
            self, in_batch: int, in_rows: int, in_cols: int, 
            axis: int, id: str
        ) -> str:
        layer_id = self.lib_funcs.get("add_sum_cuda_layer")(self.model, in_batch, in_rows, in_cols, axis, id.encode())
        return layer_id.decode("utf-8")
    
    def add_transpose(
            self, in_batch: int, in_rows: int, in_cols: int, id: str
        ) -> str:
        layer_id = self.lib_funcs.get("add_transpose_cuda_layer")(self.model, in_batch, in_rows, in_cols, id.encode())
        return layer_id.decode("utf-8")
    
    def add_cls(
            self, in_batch: int, in_rows: int, in_cols: int, token_idx: int, id: str
        ) -> str:
        layer_id = self.lib_funcs.get("add_cls_cuda_layer")(self.model, in_batch, in_rows, in_cols, token_idx, id.encode())
        return layer_id.decode("utf-8")
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model import CudaModel

class LayerAdder:
    def __init__(self, model_obj: "CudaModel"):
        self.lib_funcs = model_obj.lib_funcs
        self.model = model_obj.model
    
    def add_dense_cuda(self, n_in: int, n_out: int, batch: int, rows: int, use_bias: bool, id: str):
        self.lib_funcs.get("add_dense_cuda_layer")(self.model, n_in, n_out, batch, rows, use_bias, id.encode())
        
    def add_activation_cuda(self, batch: int, rows: int, cols: int, activation: str, id: str, scale: float = 1.0):
        self.lib_funcs.get("add_activation_cuda_layer")(self.model, batch, rows, cols, activation.encode(), scale, id.encode())
        
    def add_l2norm_cuda(self, batch: int, rows: int, cols: int, id: str):
        self.lib_funcs.get("add_l2norm_cuda_layer")(self.model, batch, rows, cols, id.encode())

    def add_pos_encoding2d_cuda(self, batch: int, rows: int, cols: int, lr: float, l2: float, id: str):
        self.lib_funcs.get("add_pos_encode_2d_cuda_layer")(self.model, batch, rows, cols, lr, l2, id.encode())

    def add_embedding2d_cuda(self, vocab_size: int, embedding_len: int, seq_len: int, id: str):
        self.lib_funcs.get("add_embedding_2d_cuda_layer")(self.model, vocab_size, embedding_len, seq_len, id.encode())
        
    def add_conv2d_cuda(
        self, n_filters: int, filter_dim: int, strides: int, 
        batch: int, rows: int, cols: int, flatten: bool, id: str
    ):
        
        self.lib_funcs.get("add_conv2d_cuda_layer")(
            self.model, n_filters, filter_dim, strides, batch, rows, cols, flatten, id.encode()
        )
        
    def add_softmax_cuda(self, batch: int, rows: int, cols: int, temperature: float, id: str):
        self.lib_funcs.get("add_softmax_cuda_layer")(self.model, batch, rows, cols, temperature, id.encode())

    def add_dropout_cuda(self, batch: int, rows: int, cols: int, dropout_rate: float, id: str):
        self.lib_funcs.get("add_dropout_cuda_layer")(self.model, batch, rows, cols, dropout_rate, id.encode())

    def add_elementwise_cuda(
        self, batch: int, rows: int, cols: int, scale_range: float, 
        op: str, dropout_rate: float, id: str, activation_str: str = "linear", 
        activation_scale: float = 1.0
    ):
        if "add" in op.lower():
            op = 0
        elif "mul" in op.lower():
            op = 1
        else:
            print(f"Error: {op.lower()} not valid (only 'add' or 'mul' allowed)")
            raise SystemExit
        
        self.lib_funcs.get("add_elementwise_cuda_layer")(
            self.model, batch, rows, cols, scale_range, op, dropout_rate, 
            id.encode(), activation_str.encode(), activation_scale
        )

    def add_broadcast_cuda(
            self, out_batch: int, out_rows: int, out_cols: int, 
            axis: int, id: str
        ):
        self.lib_funcs.get("add_broadcast_cuda_layer")(self.model, out_batch, out_rows, out_cols, axis, id.encode())

    def add_sum_cuda(
            self, in_batch: int, in_rows: int, in_cols: int, 
            axis: int, id: str
        ):
        self.lib_funcs.get("add_sum_cuda_layer")(self.model, in_batch, in_rows, in_cols, axis, id.encode())

    def add_transpose_cuda(
            self, in_batch: int, in_rows: int, in_cols: int, id: str
        ):
        self.lib_funcs.get("add_transpose_cuda_layer")(self.model, in_batch, in_rows, in_cols, id.encode())

    def add_cls_cuda(
            self, in_batch: int, in_rows: int, in_cols: int, token_idx: int, id: str
        ):
        self.lib_funcs.get("add_cls_cuda_layer")(self.model, in_batch, in_rows, in_cols, token_idx, id.encode())
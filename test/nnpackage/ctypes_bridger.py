import ctypes as C
import numpy as np
#from ctypes import CDLL

# allows methods from the rust/cuda shared library to be called in python

def load_lib(path) -> dict:
    func_dict = {}

    lib = C.cdll.LoadLibrary(path)
    func_create_model = lib.create_model
    func_create_model.restype = C.c_void_p
    func_dict.update({"create_model": func_create_model})

    # CUDA layers
    func_add_dense_cuda_layer = lib.add_dense_cuda_layer
    func_add_dense_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint64, C.c_bool, C.c_char_p
    ]
    func_add_dense_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_dense_cuda_layer": func_add_dense_cuda_layer})

    func_add_activation_cuda_layer = lib.add_activation_cuda_layer
    func_add_activation_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p, C.c_float, C.c_char_p
    ]
    func_add_activation_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_activation_cuda_layer": func_add_activation_cuda_layer})

    func_add_l2norm_cuda_layer = lib.add_l2norm_cuda_layer
    func_add_l2norm_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p
    ]
    func_add_l2norm_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_l2norm_cuda_layer": func_add_l2norm_cuda_layer})

    func_add_embedding_2d_cuda_layer = lib.add_embedding_2d_cuda_layer
    func_add_embedding_2d_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p
    ]
    func_add_embedding_2d_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_embedding_2d_cuda_layer": func_add_embedding_2d_cuda_layer})

    func_add_conv2d_cuda_layer = lib.add_conv2d_cuda_layer
    func_add_conv2d_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, 
        C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p
    ]
    func_add_conv2d_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_conv2d_cuda_layer": func_add_conv2d_cuda_layer})

    func_add_softmax_cuda_layer = lib.add_softmax_cuda_layer
    func_add_softmax_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_float, C.c_char_p
    ]
    func_add_softmax_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_softmax_cuda_layer": func_add_softmax_cuda_layer})

    func_add_dropout_cuda_layer = lib.add_dropout_cuda_layer
    func_add_dropout_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_float, C.c_char_p
    ]
    func_add_dropout_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_dropout_cuda_layer": func_add_dropout_cuda_layer})

    func_add_elementwise_cuda_layer = lib.add_elementwise_cuda_layer
    func_add_elementwise_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_float, 
        C.c_uint32, C.c_float, C.c_char_p, C.c_char_p, C.c_float
    ]
    func_add_elementwise_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_elementwise_cuda_layer": func_add_elementwise_cuda_layer})

    func_add_broadcast_cuda_layer = lib.add_broadcast_cuda_layer
    func_add_broadcast_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_int32, C.c_char_p
    ]
    func_add_broadcast_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_broadcast_cuda_layer": func_add_broadcast_cuda_layer})

    func_add_sum_cuda_layer = lib.add_sum_cuda_layer
    func_add_sum_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_int32, C.c_char_p
    ]
    func_add_sum_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_sum_cuda_layer": func_add_sum_cuda_layer})

    func_add_transpose_cuda_layer = lib.add_transpose_cuda_layer
    func_add_transpose_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p
    ]
    func_add_transpose_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_transpose_cuda_layer": func_add_transpose_cuda_layer})

    func_add_cls_cuda_layer = lib.add_cls_cuda_layer
    func_add_cls_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p
    ]
    func_add_cls_cuda_layer.restype = C.c_char_p
    func_dict.update({"add_cls_cuda_layer": func_add_cls_cuda_layer})

    ###################################################################

    func_pass_to_input = lib.pass_to_input
    func_pass_to_input.argtypes = [
        C.c_void_p, C.c_char_p, np.ctypeslib.ndpointer(np.float32), C.c_uint32
    ]
    func_pass_to_input.restype = C.c_void_p
    func_dict.update({"pass_to_input": func_pass_to_input})

    func_pass_to_output = lib.pass_to_output
    func_pass_to_output.argtypes = [
        C.c_void_p, C.c_char_p, C.c_void_p, C.c_uint32
    ]
    func_pass_to_output.restype = C.POINTER(C.c_float)
    func_dict.update({"pass_to_output": func_pass_to_output})

    func_ce_loss = lib.ce_loss
    func_ce_loss.argtypes = [
        C.c_void_p, C.c_char_p, np.ctypeslib.ndpointer(np.float32),
        C.c_uint64, C.c_uint64, C.c_uint64
    ]
    func_ce_loss.restype = C.POINTER(C.c_float)
    func_dict.update({"ce_loss": func_ce_loss})
    
    func_pass_to_output_grad = lib.pass_to_output_grad
    func_pass_to_output_grad.argtypes = [
        C.c_void_p, C.c_char_p, np.ctypeslib.ndpointer(np.float32), C.c_uint32
    ]
    func_dict.update({"pass_to_output_grad": func_pass_to_output_grad})

    func_forward = lib.forward
    func_forward.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p]
    func_forward.restype = C.c_void_p
    func_dict.update({"forward": func_forward})

    func_backward = lib.backward
    func_backward.argtypes = [C.c_void_p]
    func_dict.update({"backward": func_backward})

    func_update_params = lib.update_params
    func_update_params.argtypes = [
        C.c_void_p, C.c_char_p, C.c_int, C.c_float, C.c_float, C.c_float, C.c_float
    ]
    func_dict.update({"update_params": func_update_params})

    func_loss = lib.loss
    func_loss.argtypes = [
        C.c_char_p, 
        np.ctypeslib.ndpointer(np.float32), 
        np.ctypeslib.ndpointer(np.float32), 
        C.c_uint64
    ]
    func_loss.restype = C.c_float
    func_dict.update({"loss": func_loss})

    func_loss_grads = lib.loss_grads
    func_loss_grads.argtypes = [
        C.c_char_p, 
        np.ctypeslib.ndpointer(np.float32), 
        np.ctypeslib.ndpointer(np.float32), 
        C.c_uint64
    ]
    func_loss_grads.restype = C.POINTER(C.c_float)
    func_dict.update({"loss_grads": func_loss_grads})

    func_set_dropout = lib.set_dropout
    func_set_dropout.argtypes = [C.c_void_p, C.c_bool]
    func_dict.update({"set_dropout": func_set_dropout})
    
    func_details = lib.details
    func_details.argtypes = [C.c_void_p]
    func_dict.update({"details": func_details})

    func_save = lib.save
    func_save.argtypes = [C.c_void_p, C.c_char_p]
    func_dict.update({"save": func_save})

    func_load = lib.load
    func_load.argtypes = [C.c_void_p, C.c_char_p]
    func_dict.update({"load": func_load})

    func_delete = lib.delete
    func_delete.argtypes = [C.c_void_p]
    func_dict.update({"delete": func_delete})

    func_free_raw_vector = lib.free_raw_vector
    func_free_raw_vector.argtypes = [C.POINTER(C.c_float), C.c_uint64]
    func_dict.update({"free_raw_vector": func_free_raw_vector})

    return func_dict

def create_new_nn(funcs):
    return funcs.get("create_model")()

if __name__ == "__main__":
    pass
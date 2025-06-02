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

    func_add_dense_layer = lib.add_dense_layer
    func_add_dense_layer.argtypes = [C.c_void_p, C.c_uint64, C.c_float, C.c_float]
    func_dict.update({"add_dense_layer": func_add_dense_layer})

    func_add_activation = lib.add_activation
    func_add_activation.argtypes = [C.c_void_p, C.c_char_p, C.c_float]
    func_dict.update({"add_activation": func_add_activation})

    func_add_adaptive_activation = lib.add_adaptive_activation
    func_add_adaptive_activation.argtypes = [C.c_void_p, C.c_char_p, C.c_float]
    func_dict.update({"add_adaptive_activation": func_add_adaptive_activation})

    func_add_dropout_layer = lib.add_dropout_layer
    func_add_dropout_layer.argtypes = [C.c_void_p, C.c_float]
    func_dict.update({"add_dropout_layer": func_add_dropout_layer})

    func_add_norm_layer = lib.add_norm_layer
    func_add_norm_layer.argtypes = [C.c_void_p, C.c_float, C.c_uint64]
    func_dict.update({"add_norm_layer": func_add_norm_layer})

    func_add_reward_layer = lib.add_reward_layer
    func_add_reward_layer.argtypes = [C.c_void_p, C.c_float, C.c_float]
    func_dict.update({"add_reward_layer": func_add_reward_layer})

    func_add_conv2d_layer = lib.add_conv2d_layer
    func_add_conv2d_layer.argtypes = [
        C.c_void_p, 
        C.c_uint64, C.c_uint64,
        C.c_uint64, C.c_uint64, C.c_float]
    func_dict.update({"add_conv2d_layer": func_add_conv2d_layer})

    func_add_flatten_layer = lib.add_flatten_layer
    func_add_flatten_layer.argtypes = [C.c_void_p]
    func_dict.update({"add_flatten_layer": func_add_flatten_layer})

    func_add_recurrent_dense_layer = lib.add_recurrent_dense_layer
    func_add_recurrent_dense_layer.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_bool, C.c_float]
    func_dict.update({"add_recurrent_dense_layer": func_add_recurrent_dense_layer})

    func_add_temporal_dense_layer = lib.add_temporal_dense_layer
    func_add_temporal_dense_layer.argtypes = [
        C.c_void_p, 
        C.c_uint64, C.c_uint64,
        C.c_uint64,
        C.c_char_p, C.c_char_p, 
        C.c_bool, C.c_float
    ]
    func_dict.update({"add_temporal_dense_layer": func_add_temporal_dense_layer})

    func_add_attention_layer = lib.add_attention_layer
    func_add_attention_layer.argtypes = [
        C.c_void_p, C.c_uint64, 
        C.c_uint64, C.c_uint64, C.c_uint64,
        C.c_char_p, 
        C.c_bool, C.c_float, C.c_float
    ]
    func_dict.update({"add_attention_layer": func_add_attention_layer})

    ###################################################################
    # CUDA layers
    func_add_dense_cuda_layer = lib.add_dense_cuda_layer
    func_add_dense_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint64, C.c_bool, C.c_char_p
        ]
    func_dict.update({"add_dense_cuda_layer": func_add_dense_cuda_layer})

    func_add_activation_cuda_layer = lib.add_activation_cuda_layer
    func_add_activation_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p, C.c_float, C.c_char_p
    ]
    func_dict.update({"add_activation_cuda_layer": func_add_activation_cuda_layer})

    func_add_l2norm_cuda_layer = lib.add_l2norm_cuda_layer
    func_add_l2norm_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p
    ]
    func_dict.update({"add_l2norm_cuda_layer": func_add_l2norm_cuda_layer})

    func_add_embedding_2d_cuda_layer = lib.add_embedding_2d_cuda_layer
    func_add_embedding_2d_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p
    ]
    func_dict.update({"add_embedding_2d_cuda_layer": func_add_embedding_2d_cuda_layer})

    func_add_conv2d_cuda_layer = lib.add_conv2d_cuda_layer
    func_add_conv2d_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, 
        C.c_uint64, C.c_uint64, C.c_uint64, C.c_bool, C.c_char_p
    ]
    func_dict.update({"add_conv2d_cuda_layer": func_add_conv2d_cuda_layer})

    func_add_softmax_cuda_layer = lib.add_softmax_cuda_layer
    func_add_softmax_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_float, C.c_char_p
    ]
    func_dict.update({"add_softmax_cuda_layer": func_add_softmax_cuda_layer})

    func_add_dropout_cuda_layer = lib.add_dropout_cuda_layer
    func_add_dropout_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_float, C.c_char_p
    ]
    func_dict.update({"add_dropout_cuda_layer": func_add_dropout_cuda_layer})

    func_add_elementwise_cuda_layer = lib.add_elementwise_cuda_layer
    func_add_elementwise_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_float, 
        C.c_uint32, C.c_float, C.c_char_p, C.c_char_p, C.c_float
    ]
    func_dict.update({"add_elementwise_cuda_layer": func_add_elementwise_cuda_layer})

    func_add_broadcast_cuda_layer = lib.add_broadcast_cuda_layer
    func_add_broadcast_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_int32, C.c_char_p
    ]
    func_dict.update({"add_broadcast_cuda_layer": func_add_broadcast_cuda_layer})

    func_add_sum_cuda_layer = lib.add_sum_cuda_layer
    func_add_sum_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_int32, C.c_char_p
    ]
    func_dict.update({"add_sum_cuda_layer": func_add_sum_cuda_layer})

    func_add_transpose_cuda_layer = lib.add_transpose_cuda_layer
    func_add_transpose_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p
    ]
    func_dict.update({"add_transpose_cuda_layer": func_add_transpose_cuda_layer})

    func_add_cls_cuda_layer = lib.add_cls_cuda_layer
    func_add_cls_cuda_layer.argtypes = [
        C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint64, C.c_char_p
    ]
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

    '''
    func_join_input = lib.join_input
    func_join_input.argtypes = [
        C.c_void_p, C.c_char_p, C.c_char_p
    ]
    func_dict.update({"join_input": func_join_input})

    func_join_output = lib.join_output
    func_join_output.argtypes = [
        C.c_void_p, C.c_char_p, C.c_char_p
    ]
    func_dict.update({"join_output": func_join_output})

    func_join_layers = lib.join_layers
    func_join_layers.argtypes = [
        C.c_void_p, C.c_char_p, C.c_char_p,
        C.c_char_p, C.c_char_p,
        C.c_char_p, C.c_char_p,
        C.c_uint32
    ]
    func_dict.update({"join_layers": func_join_layers})

    func_append_layer_chain = lib.append_layer_chain
    func_append_layer_chain.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_char_p]
    func_dict.update({"append_layer_chain": func_append_layer_chain})
    '''

    func_forward = lib.forward
    func_forward.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p]
    func_forward.restype = C.c_void_p
    func_dict.update({"forward": func_forward})

    '''
    func_get_output = lib.get_output
    func_get_output.argtypes = [C.c_void_p, C.c_char_p]
    func_get_output.restype = C.POINTER(C.c_float)
    func_dict.update({"get_output": func_get_output})
    '''

    func_backward = lib.backward
    func_backward.argtypes = [C.c_void_p]
    func_dict.update({"backward": func_backward})

    func_update_params = lib.update_params
    func_update_params.argtypes = [
        C.c_void_p, C.c_char_p, C.c_int, C.c_float, C.c_float, C.c_float, C.c_float
    ]
    func_dict.update({"update_params": func_update_params})
    
    func_free_array = lib.free_array
    func_free_array.argtypes = [C.POINTER(C.c_float), C.c_uint64]
    func_dict.update({"free_array": func_free_array})

    func_free_nn = lib.free_nn
    func_free_nn.argtypes = [C.c_void_p]
    func_dict.update({"free_nn": func_free_nn})

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
    func_save.argtypes = [C.c_void_p, C.c_char_p, C.c_bool]
    func_dict.update({"save": func_save})

    func_load = lib.load
    func_load.argtypes = [C.c_char_p]
    func_load.restype = C.c_void_p
    func_dict.update({"load": func_load})

    ################################################################
    # for storage buffer
    func_create_replay_buf = lib.create_replay_buf
    func_create_replay_buf.argtypes = [C.c_uint64, C.c_uint64]
    func_create_replay_buf.restype = C.c_void_p
    func_dict.update({"create_replay_buf": func_create_replay_buf})

    func_add_state = lib.add_state
    func_add_state.argtypes = [
        C.c_void_p, 
        np.ctypeslib.ndpointer(np.float32), C.c_uint64, 
        np.ctypeslib.ndpointer(np.uint64), C.c_uint64,
        np.ctypeslib.ndpointer(np.float32), C.c_uint64, 
        np.ctypeslib.ndpointer(np.uint64), C.c_uint64,
        C.c_float, C.c_float
    ]
    func_dict.update({"add_state": func_add_state})

    #func_train_on_rand_seq = lib.train_on_rand_seq
    #func_train_on_rand_seq.argtypes = [
    #    C.c_void_p, C.c_void_p, C.c_void_p
    #]
    #func_train_on_rand_seq.restype = C.c_float
    #func_dict.update({"train_on_rand_seq": func_train_on_rand_seq})

    #func_mov_seq_idx = lib.mov_seq_idx
    #func_mov_seq_idx.argtypes = [C.c_void_p]
    #func_mov_seq_idx.restype = C.c_uint64
    #func_dict.update({"mov_seq_idx": func_mov_seq_idx})

    #func_update_saved_seq = lib.update_saved_seq
    #func_update_saved_seq.argtypes = [C.c_void_p]
    #func_dict.update({"update_saved_seq": func_update_saved_seq})

    #func_get_saved_seq_len = lib.get_saved_seq_len
    #func_get_saved_seq_len.argtypes = [C.c_void_p]
    #func_get_saved_seq_len.restype = C.c_uint64
    #func_dict.update({"get_saved_seq_len": func_get_saved_seq_len})

    #func_get_seq_rating_at = lib.get_seq_rating_at
    #func_get_seq_rating_at.argtypes = [C.c_void_p, C.c_uint64]
    #func_get_seq_rating_at.restype = C.c_float
    #func_dict.update({"get_seq_rating_at": func_get_seq_rating_at})

    func_get_state_input_at = lib.get_state_input_at
    func_get_state_input_at.argtypes = [C.c_void_p, C.c_uint64]
    func_get_state_input_at.restype = C.POINTER(C.c_float)
    func_dict.update({"get_state_input_at": func_get_state_input_at})

    func_get_state_output_at = lib.get_state_output_at
    func_get_state_output_at.argtypes = [C.c_void_p, C.c_uint64]
    func_get_state_output_at.restype = C.POINTER(C.c_float)
    func_dict.update({"get_state_output_at": func_get_state_output_at})

    func_reset_online_queue = lib.reset_online_queue
    func_reset_online_queue.argtypes = [C.c_void_p]
    func_dict.update({"reset_online_queue": func_reset_online_queue})

    func_get_ave_rating = lib.get_ave_rating
    func_get_ave_rating.argtypes = [C.c_void_p]
    func_get_ave_rating.restype = C.c_float
    func_dict.update({"get_ave_rating": func_get_ave_rating})

    func_get_count = lib.get_count
    func_get_count.argtypes = [C.c_void_p]
    func_get_count.restype = C.c_uint64
    func_dict.update({"get_count": func_get_count})

    ################################################################
    # for autoencoder buffer
    '''
    func_create_ae_buf = lib.create_ae_buf
    func_create_ae_buf.argtypes = [C.c_uint64, C.c_void_p, C.c_void_p]
    func_create_ae_buf.restype = C.c_void_p
    func_dict.update({"create_ae_buf": func_create_ae_buf})

    func_add_to_autoencoder_buf = lib.add_to_autoencoder_buf
    func_add_to_autoencoder_buf.argtypes = [
        C.c_void_p, 
        np.ctypeslib.ndpointer(np.float32), C.c_uint64, 
        np.ctypeslib.ndpointer(np.uint64), C.c_uint64,
    ]
    func_dict.update({"add_to_autoencoder_buf": func_add_to_autoencoder_buf})

    func_train_autoencoder = lib.train_autoencoder
    func_train_autoencoder.argtypes = [C.c_void_p]
    func_train_autoencoder.restype = C.c_float
    func_dict.update({"train_autoencoder": func_train_autoencoder})
    '''

    return func_dict

def create_new_nn(funcs):
    return funcs.get("create_model")()

if __name__ == "__main__":
    pass
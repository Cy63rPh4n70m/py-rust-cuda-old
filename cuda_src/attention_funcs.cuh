#pragma once

extern "C"
{
    __declspec(dllexport) void kqv_forward_ext(
        float* input_ptr, unsigned int z0, unsigned int y0, unsigned int x0,
        float* k_weight_ptr, float* k_bias_ptr, unsigned int z1, unsigned int y1, unsigned int x1, 
        float* q_weight_ptr, float* q_bias_ptr, 
        float* v_weight_ptr, float* v_bias_ptr,
        float* k_result_ptr, float* q_result_ptr, float* v_result_ptr
    );
    __declspec(dllexport) void kqv_backward_ext(
        float* input_tensor,
        float* input_grads, unsigned int z0, unsigned int y0, unsigned int x0, // (3, 2, 5, 1)
        float* k_weight_grads, unsigned int z1, unsigned int y1, unsigned int x1, // (3, 1, 5, 4)
        float* k_bias_grads, unsigned int z2, unsigned int y2, unsigned int x2, // (3, 2, 5, 4) -> (3, 2, 4)
        float* k_original_grads, // (3, 2, 4)
        float* k_weight_tensor,

        float* q_weight_grads, 
        float* q_bias_grads,
        float* q_original_grads, // (3, 2, 4)
        float* q_weight_tensor,

        float* v_weight_grads, 
        float* v_bias_grads,
        float* v_original_grads, // (3, 2, 4)
        float* v_weight_tensor
    );
    __declspec(dllexport) void kqv_update_params_ext(
        float lr,
        float* k_weight, float* k_weight_grads, float* k_weight_vel, unsigned int z0, unsigned int y0, unsigned int x0,
        float* k_biases, float* k_biases_grads, float* k_bias_vel, unsigned int z1, unsigned int y1, unsigned int x1,
        float* q_weight, float* q_weight_grads, float* q_weight_vel, 
        float* q_biases, float* q_biases_grads, float* q_bias_vel, 
        float* v_weight, float* v_weight_grads, float* v_weight_vel, unsigned int z2, unsigned int y2, unsigned int x2,
        float* v_biases, float* v_biases_grads, float* v_bias_vel, unsigned int z3, unsigned int y3, unsigned int x3
    );
}
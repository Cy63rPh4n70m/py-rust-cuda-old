#pragma once

extern "C"
{
    __declspec(dllexport) void embedding_forward_ext(
        float* index_vec, unsigned int seq_len, 
        float* embedding_lookup, unsigned int vocab_size, unsigned int embedding_len,
        float* result_embedding
    );

    __declspec(dllexport) void embedding_backward_ext(
        float* index_vec, unsigned int seq_len, 
        float* embedding_lookup, unsigned int vocab_size, unsigned int embedding_len,
        float* embedding_lookup_grads, 
        float* embedding_lookup_grads_temp, 
        float* embedding_lookup_grads_count, float* prev_grads,
        float* result_grads
    );
}

__global__ void embedding_forward(
    float* index_vec, unsigned int seq_len, 
    float* embedding_lookup, unsigned int vocab_size, unsigned int embedding_len,
    float* result_embedding
);

__global__ void embedding_backward(
    float* index_vec, unsigned int seq_len, 
    float* embedding_lookup_grads_temp, // (1, vocab_len, embedding)
    float* embedding_lookup_grads_count,
    unsigned int vocab_size, unsigned int embedding_len,
    float* prev_grads // (1, seq_len, embedding)
);

__global__ void ave_embedding_grad(
    float* embedding_lookup_grads, 
    float* embedding_lookup_grads_temp, 
    float* embedding_lookup_grads_count,
    unsigned int vocab_size, unsigned int embedding_len
);
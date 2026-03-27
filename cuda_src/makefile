cudakernelfuncs.dll: external.obj dense_funcs.obj array_ops.obj conv_funcs.obj activation.obj layer_norm_funcs.obj attention_funcs.obj reduction_funcs.obj matmul_tiling.obj embedding_funcs.obj softmax.obj dropout.obj l2norm.obj
	nvcc -use_fast_math -Xptxas -O3 -shared -lcublas -o cudakernelfuncs.dll external.obj dense_funcs.obj array_ops.obj conv_funcs.obj activation.obj layer_norm_funcs.obj attention_funcs.obj reduction_funcs.obj matmul_tiling.obj embedding_funcs.obj softmax.obj dropout.obj l2norm.obj

external.obj: external.cu external.cuh
	nvcc -use_fast_math -Xptxas -O3 -dc -c external.cu

dense_funcs.obj: dense_funcs.cu dense_funcs.cuh
	nvcc -use_fast_math -Xptxas -O3 -dc -c dense_funcs.cu

array_ops.obj: array_ops.cu array_ops.cuh
	nvcc -use_fast_math -Xptxas -O3 -dc -c external.cu
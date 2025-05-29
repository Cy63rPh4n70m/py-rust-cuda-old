cls
cd cuda_lib
nvcc -use_fast_math -Xptxas -O3 -dc -c external.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c dense_funcs.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c array_ops.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c conv_funcs.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c activation.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c layer_norm_funcs.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c reduction_funcs.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c matmul_tiling.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c embedding_funcs.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c l2norm.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c softmax.cu
nvcc -use_fast_math -Xptxas -O3 -dc -c dropout.cu
nvcc -use_fast_math -Xptxas -O3 -shared -lcublas -o cudakernelfuncs.dll external.obj dense_funcs.obj array_ops.obj conv_funcs.obj activation.obj layer_norm_funcs.obj reduction_funcs.obj matmul_tiling.obj embedding_funcs.obj softmax.obj dropout.obj l2norm.obj
move *.dll ../nnpackage/rust_backend/
move *.lib ../nnpackage/rust_backend/
move *.exp ../nnpackage/rust_backend/
del *.obj
cd ..
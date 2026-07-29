###### Benchmarking
## DataType : float32 -> explicit cast
## No padding
## Constant stride = 1

## Synthetic tensors float32 SEED: 42


## INPUT -> Synthetic SEED 42
    video_tensor_shape (N, C, H, W)
    
    Filter bank containing all our filters

    cross_correlation_tensor_shape (K, C, KH, KW)

    ### Performance benchmarking
        Ranging N, C, K, H, W
## INPUT -> Real data
    ImageIO video -> npy file containing the video tensor
    video_tensor_shape (N, C, H, W)

    Filter bank containing all our filters

    cross_correlation_tensor_shape (K, C, KH, KW)

## Transition and compute
    Im2Col 6D tensor view

    im2col_tensor_view_shape (N, H_out, W_out, C, KH, KW)
    constant stride = 1

    H_out -> floor((H - Kh)/stride) + 1
    W_out -> floor((W - Kw)/stride) + 1
## OUTPUT 
    4D Tensor representing post processed video

    processed_video_tensor_shape (N, H_out, W_out, K)

    
  

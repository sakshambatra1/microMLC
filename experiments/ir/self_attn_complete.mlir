// --- MAP DEFINITIONS ---
#map_identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_reduce_3d_2d = affine_map<(d0, d1, d2) -> (d0, d1)>

// Used for Q: (B, H, S, D) <- Input(B, S, H, D)
// Loops are B, H, S, D. Input needs B(d0), S(d2), H(d1), D(d3)
#map_transpose_q = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// Used for K: (B, H, D, S) <- Input(B, S, H, D)
// Loops are B, H, D, S. Input needs B(d0), S(d3), H(d1), D(d2)
#map_transpose_k = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// Used for Output: (B, S, H, D) <- Input(B, H, S, D)
// Loops are B, S, H, D. Input needs B(d0), H(d2), S(d1), D(d3)
#map_transpose_out = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>


module {
  func.func public @main(%arg0: tensor<1x128x128xf32>, 
                         %arg1: tensor<1x128x128xf32>, 
                         %arg2: tensor<1x128x128xf32>, 
                         %arg3: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
    
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_neg_inf = arith.constant 0xFF800000 : f32
    %cst_scale = arith.constant 0.176776695 : f32 

    // ========================================================================
    // PART 1: LINEAR PROJECTIONS
    // ========================================================================
    %in_flat = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %w_q_flat = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %w_k_flat = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %w_v_flat = tensor.collapse_shape %arg3 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>

    %init_proj = tensor.empty() : tensor<128x128xf32>
    %q_proj = linalg.matmul ins(%in_flat, %w_q_flat : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init_proj : tensor<128x128xf32>) -> tensor<128x128xf32>
    %k_proj = linalg.matmul ins(%in_flat, %w_k_flat : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init_proj : tensor<128x128xf32>) -> tensor<128x128xf32>
    %v_proj = linalg.matmul ins(%in_flat, %w_v_flat : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init_proj : tensor<128x128xf32>) -> tensor<128x128xf32>

    // ========================================================================
    // PART 2: HEAD SPLITTING (4D Tensors)
    // ========================================================================
    %q_expanded = tensor.expand_shape %q_proj [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %k_expanded = tensor.expand_shape %k_proj [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %v_expanded = tensor.expand_shape %v_proj [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>

    %init_heads = tensor.empty() : tensor<1x4x128x32xf32>
    
    // Q Heads: (B, S, H, D) -> (B, H, S, D)
    %q_heads = linalg.generic {
        indexing_maps = [#map_transpose_q, #map_identity_4d],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%q_expanded : tensor<1x128x4x32xf32>) outs(%init_heads : tensor<1x4x128x32xf32>) {
        ^bb0(%in: f32, %out: f32): linalg.yield %in : f32
    } -> tensor<1x4x128x32xf32>

    %init_heads_T = tensor.empty() : tensor<1x4x32x128xf32>
    
    // K Heads: (B, S, H, D) -> (B, H, D, S)  <-- The mismatch was here
    // [FIX] Using #map_transpose_k
    %k_heads_T = linalg.generic {
         indexing_maps = [#map_transpose_k, #map_identity_4d],
         iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%k_expanded : tensor<1x128x4x32xf32>) outs(%init_heads_T : tensor<1x4x32x128xf32>) {
        ^bb0(%in: f32, %out: f32): linalg.yield %in : f32
    } -> tensor<1x4x32x128xf32>
    
    // V Heads: (B, S, H, D) -> (B, H, S, D)
    %v_heads = linalg.generic {
        indexing_maps = [#map_transpose_q, #map_identity_4d],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%v_expanded : tensor<1x128x4x32xf32>) outs(%init_heads : tensor<1x4x128x32xf32>) {
        ^bb0(%in: f32, %out: f32): linalg.yield %in : f32
    } -> tensor<1x4x128x32xf32>

    // ========================================================================
    // PART 3: ATTENTION CORE (3D Tensors - Rank reduced)
    // ========================================================================
    %q_flat = tensor.collapse_shape %q_heads [[0, 1], [2], [3]] : tensor<1x4x128x32xf32> into tensor<4x128x32xf32>
    %k_flat = tensor.collapse_shape %k_heads_T [[0, 1], [2], [3]] : tensor<1x4x32x128xf32> into tensor<4x32x128xf32>
    %v_flat = tensor.collapse_shape %v_heads [[0, 1], [2], [3]] : tensor<1x4x128x32xf32> into tensor<4x128x32xf32>

    %scores_init = tensor.empty() : tensor<4x128x128xf32>
    %scores = linalg.batch_matmul ins(%q_flat, %k_flat : tensor<4x128x32xf32>, tensor<4x32x128xf32>)
                                  outs(%scores_init : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>

    // Scaling
    %scaled_init = tensor.empty() : tensor<4x128x128xf32>
    %scaled = linalg.generic {
        indexing_maps = [#map_identity_3d, #map_identity_3d],
        iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%scores : tensor<4x128x128xf32>) outs(%scaled_init : tensor<4x128x128xf32>) {
        ^bb0(%in: f32, %out: f32):
            %res = arith.mulf %in, %cst_scale : f32
            linalg.yield %res : f32
    } -> tensor<4x128x128xf32>

    // Softmax: Max
    %max_init = tensor.empty() : tensor<4x128xf32>
    %max_fill = linalg.fill ins(%cst_neg_inf : f32) outs(%max_init : tensor<4x128xf32>) -> tensor<4x128xf32>
    %max = linalg.generic {
        indexing_maps = [#map_identity_3d, #map_reduce_3d_2d],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%scaled : tensor<4x128x128xf32>) outs(%max_fill : tensor<4x128xf32>) {
        ^bb0(%in: f32, %acc: f32):
            %gt = arith.cmpf ogt, %in, %acc : f32
            %sel = arith.select %gt, %in, %acc : f32
            linalg.yield %sel : f32
    } -> tensor<4x128xf32>

    // Softmax: Exp
    %exp_init = tensor.empty() : tensor<4x128x128xf32>
    %exp = linalg.generic {
        indexing_maps = [#map_identity_3d, #map_reduce_3d_2d, #map_identity_3d],
        iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%scaled, %max : tensor<4x128x128xf32>, tensor<4x128xf32>) outs(%exp_init : tensor<4x128x128xf32>) {
        ^bb0(%in: f32, %in_max: f32, %out: f32):
            %sub = arith.subf %in, %in_max : f32
            %res = math.exp %sub : f32
            linalg.yield %res : f32
    } -> tensor<4x128x128xf32>

    // Softmax: Sum
    %sum_init = tensor.empty() : tensor<4x128xf32>
    %sum_fill = linalg.fill ins(%cst_0 : f32) outs(%sum_init : tensor<4x128xf32>) -> tensor<4x128xf32>
    %sum = linalg.generic {
        indexing_maps = [#map_identity_3d, #map_reduce_3d_2d],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%exp : tensor<4x128x128xf32>) outs(%sum_fill : tensor<4x128xf32>) {
        ^bb0(%in: f32, %acc: f32):
            %add = arith.addf %in, %acc : f32
            linalg.yield %add : f32
    } -> tensor<4x128xf32>

    // Softmax: Div
    %prob_init = tensor.empty() : tensor<4x128x128xf32>
    %probs = linalg.generic {
        indexing_maps = [#map_identity_3d, #map_reduce_3d_2d, #map_identity_3d],
        iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%exp, %sum : tensor<4x128x128xf32>, tensor<4x128xf32>) outs(%prob_init : tensor<4x128x128xf32>) {
        ^bb0(%in: f32, %in_sum: f32, %out: f32):
            %div = arith.divf %in, %in_sum : f32
            linalg.yield %div : f32
    } -> tensor<4x128x128xf32>

    // Context MatMul
    %context_init = tensor.empty() : tensor<4x128x32xf32>
    %context = linalg.batch_matmul ins(%probs, %v_flat : tensor<4x128x128xf32>, tensor<4x128x32xf32>)
                                   outs(%context_init : tensor<4x128x32xf32>) -> tensor<4x128x32xf32>

    // ========================================================================
    // PART 4: OUTPUT PROJECTION (4D Tensors)
    // ========================================================================
    %context_expanded = tensor.expand_shape %context [[0, 1], [2], [3]] : tensor<4x128x32xf32> into tensor<1x4x128x32xf32>
    %output_transposed_init = tensor.empty() : tensor<1x128x4x32xf32>
    
    // [FIX] Using #map_transpose_out
    %output_transposed = linalg.generic {
        indexing_maps = [#map_transpose_out, #map_identity_4d], 
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%context_expanded : tensor<1x4x128x32xf32>) outs(%output_transposed_init : tensor<1x128x4x32xf32>) {
        ^bb0(%in: f32, %out: f32): linalg.yield %in : f32
    } -> tensor<1x128x4x32xf32>

    %result = tensor.collapse_shape %output_transposed [[0], [1], [2, 3]] : tensor<1x128x4x32xf32> into tensor<1x128x128xf32>

    return %result : tensor<1x128x128xf32>
  }
}
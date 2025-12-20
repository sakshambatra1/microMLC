#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func public @main(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.176776692 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_2 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_3 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_4 = tensor.collapse_shape %arg3 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %1) -> (tensor<128x128xf32>) {
      %30 = scf.for %arg6 = %c0 to %c128 step %c32 iter_args(%arg7 = %arg5) -> (tensor<128x128xf32>) {
        %extracted_slice = tensor.extract_slice %collapsed[0, %arg6] [128, 32] [1, 1] : tensor<128x128xf32> to tensor<128x32xf32>
        %extracted_slice_12 = tensor.extract_slice %collapsed_2[%arg6, %arg4] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
        %extracted_slice_13 = tensor.extract_slice %arg7[0, %arg4] [128, 32] [1, 1] : tensor<128x128xf32> to tensor<128x32xf32>
        %31 = scf.for %arg8 = %c0 to %c32 step %c8 iter_args(%arg9 = %extracted_slice_13) -> (tensor<128x32xf32>) {
          %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, %arg8] [128, 8] [1, 1] : tensor<128x32xf32> to tensor<128x8xf32>
          %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[%arg8, 0] [8, 32] [1, 1] : tensor<32x32xf32> to tensor<8x32xf32>
          %32 = linalg.matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<128x8xf32>, tensor<8x32xf32>) outs(%arg9 : tensor<128x32xf32>) -> tensor<128x32xf32>
          scf.yield %32 : tensor<128x32xf32>
        } {done_tiling}
        %inserted_slice = tensor.insert_slice %31 into %arg7[0, %arg4] [128, 32] [1, 1] : tensor<128x32xf32> into tensor<128x128xf32>
        scf.yield %inserted_slice : tensor<128x128xf32>
      } {done_tiling}
      scf.yield %30 : tensor<128x128xf32>
    } {done_tiling}
    %3 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %1) -> (tensor<128x128xf32>) {
      %30 = scf.for %arg6 = %c0 to %c128 step %c32 iter_args(%arg7 = %arg5) -> (tensor<128x128xf32>) {
        %extracted_slice = tensor.extract_slice %collapsed[0, %arg6] [128, 32] [1, 1] : tensor<128x128xf32> to tensor<128x32xf32>
        %extracted_slice_12 = tensor.extract_slice %collapsed_3[%arg6, %arg4] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
        %extracted_slice_13 = tensor.extract_slice %arg7[0, %arg4] [128, 32] [1, 1] : tensor<128x128xf32> to tensor<128x32xf32>
        %31 = scf.for %arg8 = %c0 to %c32 step %c8 iter_args(%arg9 = %extracted_slice_13) -> (tensor<128x32xf32>) {
          %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, %arg8] [128, 8] [1, 1] : tensor<128x32xf32> to tensor<128x8xf32>
          %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[%arg8, 0] [8, 32] [1, 1] : tensor<32x32xf32> to tensor<8x32xf32>
          %32 = linalg.matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<128x8xf32>, tensor<8x32xf32>) outs(%arg9 : tensor<128x32xf32>) -> tensor<128x32xf32>
          scf.yield %32 : tensor<128x32xf32>
        } {done_tiling}
        %inserted_slice = tensor.insert_slice %31 into %arg7[0, %arg4] [128, 32] [1, 1] : tensor<128x32xf32> into tensor<128x128xf32>
        scf.yield %inserted_slice : tensor<128x128xf32>
      } {done_tiling}
      scf.yield %30 : tensor<128x128xf32>
    } {done_tiling}
    %4 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %1) -> (tensor<128x128xf32>) {
      %30 = scf.for %arg6 = %c0 to %c128 step %c32 iter_args(%arg7 = %arg5) -> (tensor<128x128xf32>) {
        %extracted_slice = tensor.extract_slice %collapsed[0, %arg6] [128, 32] [1, 1] : tensor<128x128xf32> to tensor<128x32xf32>
        %extracted_slice_12 = tensor.extract_slice %collapsed_4[%arg6, %arg4] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
        %extracted_slice_13 = tensor.extract_slice %arg7[0, %arg4] [128, 32] [1, 1] : tensor<128x128xf32> to tensor<128x32xf32>
        %31 = scf.for %arg8 = %c0 to %c32 step %c8 iter_args(%arg9 = %extracted_slice_13) -> (tensor<128x32xf32>) {
          %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, %arg8] [128, 8] [1, 1] : tensor<128x32xf32> to tensor<128x8xf32>
          %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[%arg8, 0] [8, 32] [1, 1] : tensor<32x32xf32> to tensor<8x32xf32>
          %32 = linalg.matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<128x8xf32>, tensor<8x32xf32>) outs(%arg9 : tensor<128x32xf32>) -> tensor<128x32xf32>
          scf.yield %32 : tensor<128x32xf32>
        } {done_tiling}
        %inserted_slice = tensor.insert_slice %31 into %arg7[0, %arg4] [128, 32] [1, 1] : tensor<128x32xf32> into tensor<128x128xf32>
        scf.yield %inserted_slice : tensor<128x128xf32>
      } {done_tiling}
      scf.yield %30 : tensor<128x128xf32>
    } {done_tiling}
    %expanded = tensor.expand_shape %2 [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %expanded_5 = tensor.expand_shape %3 [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %expanded_6 = tensor.expand_shape %4 [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %5 = tensor.empty() : tensor<1x4x128x32xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x128x4x32xf32>) outs(%5 : tensor<1x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x128x32xf32>
    %7 = tensor.empty() : tensor<1x4x32x128xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_5 : tensor<1x128x4x32xf32>) outs(%7 : tensor<1x4x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x32x128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<1x128x4x32xf32>) outs(%5 : tensor<1x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x128x32xf32>
    %collapsed_7 = tensor.collapse_shape %6 [[0, 1], [2], [3]] : tensor<1x4x128x32xf32> into tensor<4x128x32xf32>
    %collapsed_8 = tensor.collapse_shape %8 [[0, 1], [2], [3]] : tensor<1x4x32x128xf32> into tensor<4x32x128xf32>
    %collapsed_9 = tensor.collapse_shape %9 [[0, 1], [2], [3]] : tensor<1x4x128x32xf32> into tensor<4x128x32xf32>
    %10 = tensor.empty() : tensor<4x128x128xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %12 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %11) -> (tensor<4x128x128xf32>) {
      %30 = scf.for %arg6 = %c0 to %c32 step %c32 iter_args(%arg7 = %arg5) -> (tensor<4x128x128xf32>) {
        %extracted_slice = tensor.extract_slice %collapsed_7[0, 0, %arg6] [4, 128, 32] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x128x32xf32>
        %extracted_slice_12 = tensor.extract_slice %collapsed_8[0, %arg6, %arg4] [4, 32, 32] [1, 1, 1] : tensor<4x32x128xf32> to tensor<4x32x32xf32>
        %extracted_slice_13 = tensor.extract_slice %arg7[0, 0, %arg4] [4, 128, 32] [1, 1, 1] : tensor<4x128x128xf32> to tensor<4x128x32xf32>
        %31 = scf.for %arg8 = %c0 to %c32 step %c8 iter_args(%arg9 = %extracted_slice_13) -> (tensor<4x128x32xf32>) {
          %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, 0, %arg8] [4, 128, 8] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x128x8xf32>
          %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[0, %arg8, 0] [4, 8, 32] [1, 1, 1] : tensor<4x32x32xf32> to tensor<4x8x32xf32>
          %32 = linalg.batch_matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<4x128x8xf32>, tensor<4x8x32xf32>) outs(%arg9 : tensor<4x128x32xf32>) -> tensor<4x128x32xf32>
          scf.yield %32 : tensor<4x128x32xf32>
        } {done_tiling}
        %inserted_slice = tensor.insert_slice %31 into %arg7[0, 0, %arg4] [4, 128, 32] [1, 1, 1] : tensor<4x128x32xf32> into tensor<4x128x128xf32>
        scf.yield %inserted_slice : tensor<4x128x128xf32>
      } {done_tiling}
      scf.yield %30 : tensor<4x128x128xf32>
    } {done_tiling}
    %13 = tensor.empty() : tensor<4x128x128xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12 : tensor<4x128x128xf32>) outs(%13 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.mulf %in, %cst_1 : f32
      linalg.yield %30 : f32
    } -> tensor<4x128x128xf32>
    %15 = tensor.empty() : tensor<4x128xf32>
    %16 = linalg.fill ins(%cst_0 : f32) outs(%15 : tensor<4x128xf32>) -> tensor<4x128xf32>
    %17 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14 : tensor<4x128x128xf32>) outs(%16 : tensor<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.cmpf ogt, %in, %out : f32
      %31 = arith.select %30, %in, %out : f32
      linalg.yield %31 : f32
    } -> tensor<4x128xf32>
    %18 = tensor.empty() : tensor<4x128x128xf32>
    %19 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %17 : tensor<4x128x128xf32>, tensor<4x128xf32>) outs(%18 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %30 = arith.mulf %in, %cst_1 : f32
      %31 = arith.subf %30, %in_12 : f32
      %32 = math.exp %31 : f32
      linalg.yield %32 : f32
    } -> tensor<4x128x128xf32>
    %20 = tensor.empty() : tensor<4x128xf32>
    %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<4x128xf32>) -> tensor<4x128xf32>
    %22 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : tensor<4x128x128xf32>) outs(%21 : tensor<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.addf %in, %out : f32
      linalg.yield %30 : f32
    } -> tensor<4x128xf32>
    %23 = tensor.empty() : tensor<4x128x128xf32>
    %24 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %17, %22 : tensor<4x128x128xf32>, tensor<4x128xf32>, tensor<4x128xf32>) outs(%23 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %in_12: f32, %in_13: f32, %out: f32):
      %30 = arith.mulf %in, %cst_1 : f32
      %31 = arith.subf %30, %in_12 : f32
      %32 = math.exp %31 : f32
      %33 = arith.divf %32, %in_13 : f32
      linalg.yield %33 : f32
    } -> tensor<4x128x128xf32>
    %25 = tensor.empty() : tensor<4x128x32xf32>
    %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<4x128x32xf32>) -> tensor<4x128x32xf32>
    %27 = scf.for %arg4 = %c0 to %c32 step %c32 iter_args(%arg5 = %26) -> (tensor<4x128x32xf32>) {
      %30 = scf.for %arg6 = %c0 to %c128 step %c32 iter_args(%arg7 = %arg5) -> (tensor<4x128x32xf32>) {
        %extracted_slice = tensor.extract_slice %24[0, 0, %arg6] [4, 128, 32] [1, 1, 1] : tensor<4x128x128xf32> to tensor<4x128x32xf32>
        %extracted_slice_12 = tensor.extract_slice %collapsed_9[0, %arg6, %arg4] [4, 32, 32] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x32x32xf32>
        %extracted_slice_13 = tensor.extract_slice %arg7[0, 0, %arg4] [4, 128, 32] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x128x32xf32>
        %31 = scf.for %arg8 = %c0 to %c32 step %c8 iter_args(%arg9 = %extracted_slice_13) -> (tensor<4x128x32xf32>) {
          %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, 0, %arg8] [4, 128, 8] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x128x8xf32>
          %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[0, %arg8, 0] [4, 8, 32] [1, 1, 1] : tensor<4x32x32xf32> to tensor<4x8x32xf32>
          %32 = linalg.batch_matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<4x128x8xf32>, tensor<4x8x32xf32>) outs(%arg9 : tensor<4x128x32xf32>) -> tensor<4x128x32xf32>
          scf.yield %32 : tensor<4x128x32xf32>
        } {done_tiling}
        %inserted_slice = tensor.insert_slice %31 into %arg7[0, 0, %arg4] [4, 128, 32] [1, 1, 1] : tensor<4x128x32xf32> into tensor<4x128x32xf32>
        scf.yield %inserted_slice : tensor<4x128x32xf32>
      } {done_tiling}
      scf.yield %30 : tensor<4x128x32xf32>
    } {done_tiling}
    %expanded_10 = tensor.expand_shape %27 [[0, 1], [2], [3]] : tensor<4x128x32xf32> into tensor<1x4x128x32xf32>
    %28 = tensor.empty() : tensor<1x128x4x32xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_10 : tensor<1x4x128x32xf32>) outs(%28 : tensor<1x128x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x4x32xf32>
    %collapsed_11 = tensor.collapse_shape %29 [[0], [1], [2, 3]] : tensor<1x128x4x32xf32> into tensor<1x128x128xf32>
    return %collapsed_11 : tensor<1x128x128xf32>
  }
}


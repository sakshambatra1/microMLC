#map = affine_map<(d0, d1) -> (d0, 0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map7 = affine_map<(d0) -> (-d0 + 32, 4)>
module {
  func.func public @main(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.176776692 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_2 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_3 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_4 = tensor.collapse_shape %arg3 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %0) -> (tensor<128x128xf32>) {
      %extracted_slice = tensor.extract_slice %collapsed[%arg4, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
      %extracted_slice_12 = tensor.extract_slice %arg5[%arg4, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
      %23 = scf.for %arg6 = %c0 to %c128 step %c4 iter_args(%arg7 = %extracted_slice_12) -> (tensor<1x128xf32>) {
        %24 = scf.for %arg8 = %c0 to %c128 step %c4 iter_args(%arg9 = %arg7) -> (tensor<1x128xf32>) {
          %extracted_slice_13 = tensor.extract_slice %extracted_slice[0, %arg8] [1, 4] [1, 1] : tensor<1x128xf32> to tensor<1x4xf32>
          %extracted_slice_14 = tensor.extract_slice %collapsed_2[%arg8, %arg6] [4, 4] [1, 1] : tensor<128x128xf32> to tensor<4x4xf32>
          %extracted_slice_15 = tensor.extract_slice %arg9[0, %arg6] [1, 4] [1, 1] : tensor<1x128xf32> to tensor<1x4xf32>
          %25 = vector.transfer_read %extracted_slice_13[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map} : tensor<1x4xf32>, vector<1x4x4xf32>
          %26 = vector.transfer_read %extracted_slice_14[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map1} : tensor<4x4xf32>, vector<1x4x4xf32>
          %27 = vector.transfer_read %extracted_slice_15[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x4xf32>, vector<1x4xf32>
          %28 = vector.contract {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %25, %26, %27 : vector<1x4x4xf32>, vector<1x4x4xf32> into vector<1x4xf32>
          %29 = vector.transfer_write %28, %extracted_slice_15[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
          %inserted_slice_16 = tensor.insert_slice %29 into %arg9[0, %arg6] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<1x128xf32>
          scf.yield %inserted_slice_16 : tensor<1x128xf32>
        } {done_tiling}
        scf.yield %24 : tensor<1x128xf32>
      } {done_tiling}
      %inserted_slice = tensor.insert_slice %23 into %arg5[%arg4, 0] [1, 128] [1, 1] : tensor<1x128xf32> into tensor<128x128xf32>
      scf.yield %inserted_slice : tensor<128x128xf32>
    } {done_tiling}
    %2 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %0) -> (tensor<128x128xf32>) {
      %extracted_slice = tensor.extract_slice %collapsed[%arg4, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
      %extracted_slice_12 = tensor.extract_slice %arg5[%arg4, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
      %23 = scf.for %arg6 = %c0 to %c128 step %c4 iter_args(%arg7 = %extracted_slice_12) -> (tensor<1x128xf32>) {
        %24 = scf.for %arg8 = %c0 to %c128 step %c4 iter_args(%arg9 = %arg7) -> (tensor<1x128xf32>) {
          %extracted_slice_13 = tensor.extract_slice %extracted_slice[0, %arg8] [1, 4] [1, 1] : tensor<1x128xf32> to tensor<1x4xf32>
          %extracted_slice_14 = tensor.extract_slice %collapsed_3[%arg8, %arg6] [4, 4] [1, 1] : tensor<128x128xf32> to tensor<4x4xf32>
          %extracted_slice_15 = tensor.extract_slice %arg9[0, %arg6] [1, 4] [1, 1] : tensor<1x128xf32> to tensor<1x4xf32>
          %25 = vector.transfer_read %extracted_slice_13[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map} : tensor<1x4xf32>, vector<1x4x4xf32>
          %26 = vector.transfer_read %extracted_slice_14[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map1} : tensor<4x4xf32>, vector<1x4x4xf32>
          %27 = vector.transfer_read %extracted_slice_15[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x4xf32>, vector<1x4xf32>
          %28 = vector.contract {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %25, %26, %27 : vector<1x4x4xf32>, vector<1x4x4xf32> into vector<1x4xf32>
          %29 = vector.transfer_write %28, %extracted_slice_15[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
          %inserted_slice_16 = tensor.insert_slice %29 into %arg9[0, %arg6] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<1x128xf32>
          scf.yield %inserted_slice_16 : tensor<1x128xf32>
        } {done_tiling}
        scf.yield %24 : tensor<1x128xf32>
      } {done_tiling}
      %inserted_slice = tensor.insert_slice %23 into %arg5[%arg4, 0] [1, 128] [1, 1] : tensor<1x128xf32> into tensor<128x128xf32>
      scf.yield %inserted_slice : tensor<128x128xf32>
    } {done_tiling}
    %3 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %0) -> (tensor<128x128xf32>) {
      %extracted_slice = tensor.extract_slice %collapsed[%arg4, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
      %extracted_slice_12 = tensor.extract_slice %arg5[%arg4, 0] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
      %23 = scf.for %arg6 = %c0 to %c128 step %c4 iter_args(%arg7 = %extracted_slice_12) -> (tensor<1x128xf32>) {
        %24 = scf.for %arg8 = %c0 to %c128 step %c4 iter_args(%arg9 = %arg7) -> (tensor<1x128xf32>) {
          %extracted_slice_13 = tensor.extract_slice %extracted_slice[0, %arg8] [1, 4] [1, 1] : tensor<1x128xf32> to tensor<1x4xf32>
          %extracted_slice_14 = tensor.extract_slice %collapsed_4[%arg8, %arg6] [4, 4] [1, 1] : tensor<128x128xf32> to tensor<4x4xf32>
          %extracted_slice_15 = tensor.extract_slice %arg9[0, %arg6] [1, 4] [1, 1] : tensor<1x128xf32> to tensor<1x4xf32>
          %25 = vector.transfer_read %extracted_slice_13[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map} : tensor<1x4xf32>, vector<1x4x4xf32>
          %26 = vector.transfer_read %extracted_slice_14[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map1} : tensor<4x4xf32>, vector<1x4x4xf32>
          %27 = vector.transfer_read %extracted_slice_15[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x4xf32>, vector<1x4xf32>
          %28 = vector.contract {indexing_maps = [#map2, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %25, %26, %27 : vector<1x4x4xf32>, vector<1x4x4xf32> into vector<1x4xf32>
          %29 = vector.transfer_write %28, %extracted_slice_15[%c0, %c0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
          %inserted_slice_16 = tensor.insert_slice %29 into %arg9[0, %arg6] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<1x128xf32>
          scf.yield %inserted_slice_16 : tensor<1x128xf32>
        } {done_tiling}
        scf.yield %24 : tensor<1x128xf32>
      } {done_tiling}
      %inserted_slice = tensor.insert_slice %23 into %arg5[%arg4, 0] [1, 128] [1, 1] : tensor<1x128xf32> into tensor<128x128xf32>
      scf.yield %inserted_slice : tensor<128x128xf32>
    } {done_tiling}
    %expanded = tensor.expand_shape %1 [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %expanded_5 = tensor.expand_shape %2 [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %expanded_6 = tensor.expand_shape %3 [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %4 = tensor.empty() : tensor<1x4x128x32xf32>
    %5 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x128x4x32xf32>) outs(%4 : tensor<1x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x128x32xf32>
    %6 = tensor.empty() : tensor<1x4x32x128xf32>
    %7 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_5 : tensor<1x128x4x32xf32>) outs(%6 : tensor<1x4x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x32x128xf32>
    %8 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<1x128x4x32xf32>) outs(%4 : tensor<1x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x128x32xf32>
    %collapsed_7 = tensor.collapse_shape %5 [[0, 1], [2], [3]] : tensor<1x4x128x32xf32> into tensor<4x128x32xf32>
    %collapsed_8 = tensor.collapse_shape %7 [[0, 1], [2], [3]] : tensor<1x4x32x128xf32> into tensor<4x32x128xf32>
    %collapsed_9 = tensor.collapse_shape %8 [[0, 1], [2], [3]] : tensor<1x4x128x32xf32> into tensor<4x128x32xf32>
    %9 = tensor.empty() : tensor<4x128x128xf32>
    %10 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %9) -> (tensor<4x128x128xf32>) {
      %extracted_slice = tensor.extract_slice %collapsed_7[0, %arg4, 0] [4, 1, 32] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x1x32xf32>
      %extracted_slice_12 = tensor.extract_slice %arg5[0, %arg4, 0] [4, 1, 128] [1, 1, 1] : tensor<4x128x128xf32> to tensor<4x1x128xf32>
      %23 = scf.for %arg6 = %c0 to %c128 step %c4 iter_args(%arg7 = %extracted_slice_12) -> (tensor<4x1x128xf32>) {
        %24 = scf.for %arg8 = %c0 to %c32 step %c4 iter_args(%arg9 = %arg7) -> (tensor<4x1x128xf32>) {
          %25 = affine.min #map7(%arg8)
          %extracted_slice_13 = tensor.extract_slice %extracted_slice[0, 0, %arg8] [4, 1, %25] [1, 1, 1] : tensor<4x1x32xf32> to tensor<4x1x?xf32>
          %extracted_slice_14 = tensor.extract_slice %collapsed_8[0, %arg8, %arg6] [4, %25, 4] [1, 1, 1] : tensor<4x32x128xf32> to tensor<4x?x4xf32>
          %extracted_slice_15 = tensor.extract_slice %arg9[0, 0, %arg6] [4, 1, 4] [1, 1, 1] : tensor<4x1x128xf32> to tensor<4x1x4xf32>
          %26 = linalg.batch_matmul {done_tiling} ins(%extracted_slice_13, %extracted_slice_14 : tensor<4x1x?xf32>, tensor<4x?x4xf32>) outs(%extracted_slice_15 : tensor<4x1x4xf32>) -> tensor<4x1x4xf32>
          %inserted_slice_16 = tensor.insert_slice %26 into %arg9[0, 0, %arg6] [4, 1, 4] [1, 1, 1] : tensor<4x1x4xf32> into tensor<4x1x128xf32>
          scf.yield %inserted_slice_16 : tensor<4x1x128xf32>
        } {done_tiling}
        scf.yield %24 : tensor<4x1x128xf32>
      } {done_tiling}
      %inserted_slice = tensor.insert_slice %23 into %arg5[0, %arg4, 0] [4, 1, 128] [1, 1, 1] : tensor<4x1x128xf32> into tensor<4x128x128xf32>
      scf.yield %inserted_slice : tensor<4x128x128xf32>
    } {done_tiling}
    %11 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<4x128x128xf32>) outs(%9 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.mulf %in, %cst_1 : f32
      linalg.yield %23 : f32
    } -> tensor<4x128x128xf32>
    %12 = tensor.empty() : tensor<4x128xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<4x128xf32>) -> tensor<4x128xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11 : tensor<4x128x128xf32>) outs(%13 : tensor<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.cmpf ogt, %in, %out : f32
      %24 = arith.select %23, %in, %out : f32
      linalg.yield %24 : f32
    } -> tensor<4x128xf32>
    %15 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %14 : tensor<4x128x128xf32>, tensor<4x128xf32>) outs(%9 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %23 = arith.mulf %in, %cst_1 : f32
      %24 = arith.subf %23, %in_12 : f32
      %25 = math.exp %24 : f32
      linalg.yield %25 : f32
    } -> tensor<4x128x128xf32>
    %16 = linalg.fill ins(%cst : f32) outs(%12 : tensor<4x128xf32>) -> tensor<4x128xf32>
    %17 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : tensor<4x128x128xf32>) outs(%16 : tensor<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.addf %in, %out : f32
      linalg.yield %23 : f32
    } -> tensor<4x128xf32>
    %18 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %14, %17 : tensor<4x128x128xf32>, tensor<4x128xf32>, tensor<4x128xf32>) outs(%9 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %in_12: f32, %in_13: f32, %out: f32):
      %23 = arith.mulf %in, %cst_1 : f32
      %24 = arith.subf %23, %in_12 : f32
      %25 = math.exp %24 : f32
      %26 = arith.divf %25, %in_13 : f32
      linalg.yield %26 : f32
    } -> tensor<4x128x128xf32>
    %19 = tensor.empty() : tensor<4x128x32xf32>
    %20 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %19) -> (tensor<4x128x32xf32>) {
      %extracted_slice = tensor.extract_slice %18[0, %arg4, 0] [4, 1, 128] [1, 1, 1] : tensor<4x128x128xf32> to tensor<4x1x128xf32>
      %extracted_slice_12 = tensor.extract_slice %arg5[0, %arg4, 0] [4, 1, 32] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x1x32xf32>
      %23 = scf.for %arg6 = %c0 to %c32 step %c4 iter_args(%arg7 = %extracted_slice_12) -> (tensor<4x1x32xf32>) {
        %24 = scf.for %arg8 = %c0 to %c128 step %c4 iter_args(%arg9 = %arg7) -> (tensor<4x1x32xf32>) {
          %25 = affine.min #map7(%arg6)
          %extracted_slice_13 = tensor.extract_slice %extracted_slice[0, 0, %arg8] [4, 1, 4] [1, 1, 1] : tensor<4x1x128xf32> to tensor<4x1x4xf32>
          %extracted_slice_14 = tensor.extract_slice %collapsed_9[0, %arg8, %arg6] [4, 4, %25] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x4x?xf32>
          %extracted_slice_15 = tensor.extract_slice %arg9[0, 0, %arg6] [4, 1, %25] [1, 1, 1] : tensor<4x1x32xf32> to tensor<4x1x?xf32>
          %26 = linalg.batch_matmul {done_tiling} ins(%extracted_slice_13, %extracted_slice_14 : tensor<4x1x4xf32>, tensor<4x4x?xf32>) outs(%extracted_slice_15 : tensor<4x1x?xf32>) -> tensor<4x1x?xf32>
          %inserted_slice_16 = tensor.insert_slice %26 into %arg9[0, 0, %arg6] [4, 1, %25] [1, 1, 1] : tensor<4x1x?xf32> into tensor<4x1x32xf32>
          scf.yield %inserted_slice_16 : tensor<4x1x32xf32>
        } {done_tiling}
        scf.yield %24 : tensor<4x1x32xf32>
      } {done_tiling}
      %inserted_slice = tensor.insert_slice %23 into %arg5[0, %arg4, 0] [4, 1, 32] [1, 1, 1] : tensor<4x1x32xf32> into tensor<4x128x32xf32>
      scf.yield %inserted_slice : tensor<4x128x32xf32>
    } {done_tiling}
    %expanded_10 = tensor.expand_shape %20 [[0, 1], [2], [3]] : tensor<4x128x32xf32> into tensor<1x4x128x32xf32>
    %21 = tensor.empty() : tensor<1x128x4x32xf32>
    %22 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_10 : tensor<1x4x128x32xf32>) outs(%21 : tensor<1x128x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x4x32xf32>
    %collapsed_11 = tensor.collapse_shape %22 [[0], [1], [2, 3]] : tensor<1x128x4x32xf32> into tensor<1x128x128xf32>
    return %collapsed_11 : tensor<1x128x128xf32>
  }
}


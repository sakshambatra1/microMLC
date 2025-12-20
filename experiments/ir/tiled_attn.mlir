#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map3 = affine_map<(d0) -> (-d0 + 32, 128)>
#map4 = affine_map<(d0, d1) -> (d0 - d1, 16)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d0 - d1, 32)>
module {
  func.func public @main(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.176776692 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_2 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_3 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_4 = tensor.collapse_shape %arg3 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %0) -> (tensor<128x128xf32>) {
      %27 = scf.for %arg6 = %c0 to %c128 step %c128 iter_args(%arg7 = %arg5) -> (tensor<128x128xf32>) {
        %28 = scf.for %arg8 = %c0 to %c128 step %c128 iter_args(%arg9 = %arg7) -> (tensor<128x128xf32>) {
          %extracted_slice = tensor.extract_slice %collapsed[%arg4, %arg8] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
          %extracted_slice_12 = tensor.extract_slice %collapsed_2[%arg8, %arg6] [128, 128] [1, 1] : tensor<128x128xf32> to tensor<128x128xf32>
          %extracted_slice_13 = tensor.extract_slice %arg9[%arg4, %arg6] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
          %29 = scf.for %arg10 = %c0 to %c128 step %c32 iter_args(%arg11 = %extracted_slice_13) -> (tensor<1x128xf32>) {
            %30 = scf.for %arg12 = %c0 to %c128 step %c16 iter_args(%arg13 = %arg11) -> (tensor<1x128xf32>) {
              %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, %arg12] [1, 16] [1, 1] : tensor<1x128xf32> to tensor<1x16xf32>
              %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[%arg12, %arg10] [16, 32] [1, 1] : tensor<128x128xf32> to tensor<16x32xf32>
              %extracted_slice_16 = tensor.extract_slice %arg13[0, %arg10] [1, 32] [1, 1] : tensor<1x128xf32> to tensor<1x32xf32>
              %31 = linalg.matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<1x16xf32>, tensor<16x32xf32>) outs(%extracted_slice_16 : tensor<1x32xf32>) -> tensor<1x32xf32>
              %inserted_slice_17 = tensor.insert_slice %31 into %arg13[0, %arg10] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<1x128xf32>
              scf.yield %inserted_slice_17 : tensor<1x128xf32>
            } {done_tiling}
            scf.yield %30 : tensor<1x128xf32>
          } {done_tiling}
          %inserted_slice = tensor.insert_slice %29 into %arg9[%arg4, %arg6] [1, 128] [1, 1] : tensor<1x128xf32> into tensor<128x128xf32>
          scf.yield %inserted_slice : tensor<128x128xf32>
        } {done_tiling}
        scf.yield %28 : tensor<128x128xf32>
      } {done_tiling}
      scf.yield %27 : tensor<128x128xf32>
    } {done_tiling}
    %2 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %0) -> (tensor<128x128xf32>) {
      %27 = scf.for %arg6 = %c0 to %c128 step %c128 iter_args(%arg7 = %arg5) -> (tensor<128x128xf32>) {
        %28 = scf.for %arg8 = %c0 to %c128 step %c128 iter_args(%arg9 = %arg7) -> (tensor<128x128xf32>) {
          %extracted_slice = tensor.extract_slice %collapsed[%arg4, %arg8] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
          %extracted_slice_12 = tensor.extract_slice %collapsed_3[%arg8, %arg6] [128, 128] [1, 1] : tensor<128x128xf32> to tensor<128x128xf32>
          %extracted_slice_13 = tensor.extract_slice %arg9[%arg4, %arg6] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
          %29 = scf.for %arg10 = %c0 to %c128 step %c32 iter_args(%arg11 = %extracted_slice_13) -> (tensor<1x128xf32>) {
            %30 = scf.for %arg12 = %c0 to %c128 step %c16 iter_args(%arg13 = %arg11) -> (tensor<1x128xf32>) {
              %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, %arg12] [1, 16] [1, 1] : tensor<1x128xf32> to tensor<1x16xf32>
              %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[%arg12, %arg10] [16, 32] [1, 1] : tensor<128x128xf32> to tensor<16x32xf32>
              %extracted_slice_16 = tensor.extract_slice %arg13[0, %arg10] [1, 32] [1, 1] : tensor<1x128xf32> to tensor<1x32xf32>
              %31 = linalg.matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<1x16xf32>, tensor<16x32xf32>) outs(%extracted_slice_16 : tensor<1x32xf32>) -> tensor<1x32xf32>
              %inserted_slice_17 = tensor.insert_slice %31 into %arg13[0, %arg10] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<1x128xf32>
              scf.yield %inserted_slice_17 : tensor<1x128xf32>
            } {done_tiling}
            scf.yield %30 : tensor<1x128xf32>
          } {done_tiling}
          %inserted_slice = tensor.insert_slice %29 into %arg9[%arg4, %arg6] [1, 128] [1, 1] : tensor<1x128xf32> into tensor<128x128xf32>
          scf.yield %inserted_slice : tensor<128x128xf32>
        } {done_tiling}
        scf.yield %28 : tensor<128x128xf32>
      } {done_tiling}
      scf.yield %27 : tensor<128x128xf32>
    } {done_tiling}
    %3 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %0) -> (tensor<128x128xf32>) {
      %27 = scf.for %arg6 = %c0 to %c128 step %c128 iter_args(%arg7 = %arg5) -> (tensor<128x128xf32>) {
        %28 = scf.for %arg8 = %c0 to %c128 step %c128 iter_args(%arg9 = %arg7) -> (tensor<128x128xf32>) {
          %extracted_slice = tensor.extract_slice %collapsed[%arg4, %arg8] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
          %extracted_slice_12 = tensor.extract_slice %collapsed_4[%arg8, %arg6] [128, 128] [1, 1] : tensor<128x128xf32> to tensor<128x128xf32>
          %extracted_slice_13 = tensor.extract_slice %arg9[%arg4, %arg6] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
          %29 = scf.for %arg10 = %c0 to %c128 step %c32 iter_args(%arg11 = %extracted_slice_13) -> (tensor<1x128xf32>) {
            %30 = scf.for %arg12 = %c0 to %c128 step %c16 iter_args(%arg13 = %arg11) -> (tensor<1x128xf32>) {
              %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, %arg12] [1, 16] [1, 1] : tensor<1x128xf32> to tensor<1x16xf32>
              %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[%arg12, %arg10] [16, 32] [1, 1] : tensor<128x128xf32> to tensor<16x32xf32>
              %extracted_slice_16 = tensor.extract_slice %arg13[0, %arg10] [1, 32] [1, 1] : tensor<1x128xf32> to tensor<1x32xf32>
              %31 = linalg.matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<1x16xf32>, tensor<16x32xf32>) outs(%extracted_slice_16 : tensor<1x32xf32>) -> tensor<1x32xf32>
              %inserted_slice_17 = tensor.insert_slice %31 into %arg13[0, %arg10] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<1x128xf32>
              scf.yield %inserted_slice_17 : tensor<1x128xf32>
            } {done_tiling}
            scf.yield %30 : tensor<1x128xf32>
          } {done_tiling}
          %inserted_slice = tensor.insert_slice %29 into %arg9[%arg4, %arg6] [1, 128] [1, 1] : tensor<1x128xf32> into tensor<128x128xf32>
          scf.yield %inserted_slice : tensor<128x128xf32>
        } {done_tiling}
        scf.yield %28 : tensor<128x128xf32>
      } {done_tiling}
      scf.yield %27 : tensor<128x128xf32>
    } {done_tiling}
    %expanded = tensor.expand_shape %1 [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %expanded_5 = tensor.expand_shape %2 [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %expanded_6 = tensor.expand_shape %3 [[0, 1], [2, 3]] : tensor<128x128xf32> into tensor<1x128x4x32xf32>
    %4 = tensor.empty() : tensor<1x4x128x32xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x128x4x32xf32>) outs(%4 : tensor<1x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x128x32xf32>
    %6 = tensor.empty() : tensor<1x4x32x128xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_5 : tensor<1x128x4x32xf32>) outs(%6 : tensor<1x4x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x32x128xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<1x128x4x32xf32>) outs(%4 : tensor<1x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x128x32xf32>
    %collapsed_7 = tensor.collapse_shape %5 [[0, 1], [2], [3]] : tensor<1x4x128x32xf32> into tensor<4x128x32xf32>
    %collapsed_8 = tensor.collapse_shape %7 [[0, 1], [2], [3]] : tensor<1x4x32x128xf32> into tensor<4x32x128xf32>
    %collapsed_9 = tensor.collapse_shape %8 [[0, 1], [2], [3]] : tensor<1x4x128x32xf32> into tensor<4x128x32xf32>
    %9 = tensor.empty() : tensor<4x128x128xf32>
    %10 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %9) -> (tensor<4x128x128xf32>) {
      %27 = scf.for %arg6 = %c0 to %c128 step %c128 iter_args(%arg7 = %arg5) -> (tensor<4x128x128xf32>) {
        %28 = scf.for %arg8 = %c0 to %c32 step %c128 iter_args(%arg9 = %arg7) -> (tensor<4x128x128xf32>) {
          %29 = affine.min #map3(%arg8)
          %30 = affine.min #map3(%arg8)
          %extracted_slice = tensor.extract_slice %collapsed_7[0, %arg4, %arg8] [4, 1, %29] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x1x?xf32>
          %extracted_slice_12 = tensor.extract_slice %collapsed_8[0, %arg8, %arg6] [4, %30, 128] [1, 1, 1] : tensor<4x32x128xf32> to tensor<4x?x128xf32>
          %extracted_slice_13 = tensor.extract_slice %arg9[0, %arg4, %arg6] [4, 1, 128] [1, 1, 1] : tensor<4x128x128xf32> to tensor<4x1x128xf32>
          %31 = scf.for %arg10 = %c0 to %c128 step %c32 iter_args(%arg11 = %extracted_slice_13) -> (tensor<4x1x128xf32>) {
            %32 = scf.for %arg12 = %c0 to %29 step %c16 iter_args(%arg13 = %arg11) -> (tensor<4x1x128xf32>) {
              %33 = affine.min #map4(%29, %arg12)
              %34 = affine.min #map4(%29, %arg12)
              %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, 0, %arg12] [4, 1, %33] [1, 1, 1] : tensor<4x1x?xf32> to tensor<4x1x?xf32>
              %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[0, %arg12, %arg10] [4, %34, 32] [1, 1, 1] : tensor<4x?x128xf32> to tensor<4x?x32xf32>
              %extracted_slice_16 = tensor.extract_slice %arg13[0, 0, %arg10] [4, 1, 32] [1, 1, 1] : tensor<4x1x128xf32> to tensor<4x1x32xf32>
              %35 = linalg.batch_matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<4x1x?xf32>, tensor<4x?x32xf32>) outs(%extracted_slice_16 : tensor<4x1x32xf32>) -> tensor<4x1x32xf32>
              %inserted_slice_17 = tensor.insert_slice %35 into %arg13[0, 0, %arg10] [4, 1, 32] [1, 1, 1] : tensor<4x1x32xf32> into tensor<4x1x128xf32>
              scf.yield %inserted_slice_17 : tensor<4x1x128xf32>
            } {done_tiling}
            scf.yield %32 : tensor<4x1x128xf32>
          } {done_tiling}
          %inserted_slice = tensor.insert_slice %31 into %arg9[0, %arg4, %arg6] [4, 1, 128] [1, 1, 1] : tensor<4x1x128xf32> into tensor<4x128x128xf32>
          scf.yield %inserted_slice : tensor<4x128x128xf32>
        } {done_tiling}
        scf.yield %28 : tensor<4x128x128xf32>
      } {done_tiling}
      scf.yield %27 : tensor<4x128x128xf32>
    } {done_tiling}
    %11 = tensor.empty() : tensor<4x128x128xf32>
    %12 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<4x128x128xf32>) outs(%11 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.mulf %in, %cst_1 : f32
      linalg.yield %27 : f32
    } -> tensor<4x128x128xf32>
    %13 = tensor.empty() : tensor<4x128xf32>
    %14 = linalg.fill ins(%cst_0 : f32) outs(%13 : tensor<4x128xf32>) -> tensor<4x128xf32>
    %15 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12 : tensor<4x128x128xf32>) outs(%14 : tensor<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.cmpf ogt, %in, %out : f32
      %28 = arith.select %27, %in, %out : f32
      linalg.yield %28 : f32
    } -> tensor<4x128xf32>
    %16 = tensor.empty() : tensor<4x128x128xf32>
    %17 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %15 : tensor<4x128x128xf32>, tensor<4x128xf32>) outs(%16 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %27 = arith.subf %in, %in_12 : f32
      %28 = math.exp %27 : f32
      linalg.yield %28 : f32
    } -> tensor<4x128x128xf32>
    %18 = tensor.empty() : tensor<4x128xf32>
    %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<4x128xf32>) -> tensor<4x128xf32>
    %20 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : tensor<4x128x128xf32>) outs(%19 : tensor<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.addf %in, %out : f32
      linalg.yield %27 : f32
    } -> tensor<4x128xf32>
    %21 = tensor.empty() : tensor<4x128x128xf32>
    %22 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %20 : tensor<4x128x128xf32>, tensor<4x128xf32>) outs(%21 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %27 = arith.divf %in, %in_12 : f32
      linalg.yield %27 : f32
    } -> tensor<4x128x128xf32>
    %23 = tensor.empty() : tensor<4x128x32xf32>
    %24 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %23) -> (tensor<4x128x32xf32>) {
      %27 = scf.for %arg6 = %c0 to %c32 step %c128 iter_args(%arg7 = %arg5) -> (tensor<4x128x32xf32>) {
        %28 = scf.for %arg8 = %c0 to %c128 step %c128 iter_args(%arg9 = %arg7) -> (tensor<4x128x32xf32>) {
          %29 = affine.min #map3(%arg6)
          %30 = affine.min #map3(%arg6)
          %extracted_slice = tensor.extract_slice %22[0, %arg4, %arg8] [4, 1, 128] [1, 1, 1] : tensor<4x128x128xf32> to tensor<4x1x128xf32>
          %extracted_slice_12 = tensor.extract_slice %collapsed_9[0, %arg8, %arg6] [4, 128, %29] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x128x?xf32>
          %extracted_slice_13 = tensor.extract_slice %arg9[0, %arg4, %arg6] [4, 1, %30] [1, 1, 1] : tensor<4x128x32xf32> to tensor<4x1x?xf32>
          %31 = scf.for %arg10 = %c0 to %29 step %c32 iter_args(%arg11 = %extracted_slice_13) -> (tensor<4x1x?xf32>) {
            %32 = scf.for %arg12 = %c0 to %c128 step %c16 iter_args(%arg13 = %arg11) -> (tensor<4x1x?xf32>) {
              %33 = affine.min #map7(%29, %arg10)
              %34 = affine.min #map7(%29, %arg10)
              %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, 0, %arg12] [4, 1, 16] [1, 1, 1] : tensor<4x1x128xf32> to tensor<4x1x16xf32>
              %extracted_slice_15 = tensor.extract_slice %extracted_slice_12[0, %arg12, %arg10] [4, 16, %33] [1, 1, 1] : tensor<4x128x?xf32> to tensor<4x16x?xf32>
              %extracted_slice_16 = tensor.extract_slice %arg13[0, 0, %arg10] [4, 1, %34] [1, 1, 1] : tensor<4x1x?xf32> to tensor<4x1x?xf32>
              %35 = linalg.batch_matmul {done_tiling} ins(%extracted_slice_14, %extracted_slice_15 : tensor<4x1x16xf32>, tensor<4x16x?xf32>) outs(%extracted_slice_16 : tensor<4x1x?xf32>) -> tensor<4x1x?xf32>
              %inserted_slice_17 = tensor.insert_slice %35 into %arg13[0, 0, %arg10] [4, 1, %34] [1, 1, 1] : tensor<4x1x?xf32> into tensor<4x1x?xf32>
              scf.yield %inserted_slice_17 : tensor<4x1x?xf32>
            } {done_tiling}
            scf.yield %32 : tensor<4x1x?xf32>
          } {done_tiling}
          %inserted_slice = tensor.insert_slice %31 into %arg9[0, %arg4, %arg6] [4, 1, %30] [1, 1, 1] : tensor<4x1x?xf32> into tensor<4x128x32xf32>
          scf.yield %inserted_slice : tensor<4x128x32xf32>
        } {done_tiling}
        scf.yield %28 : tensor<4x128x32xf32>
      } {done_tiling}
      scf.yield %27 : tensor<4x128x32xf32>
    } {done_tiling}
    %expanded_10 = tensor.expand_shape %24 [[0, 1], [2], [3]] : tensor<4x128x32xf32> into tensor<1x4x128x32xf32>
    %25 = tensor.empty() : tensor<1x128x4x32xf32>
    %26 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_10 : tensor<1x4x128x32xf32>) outs(%25 : tensor<1x128x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x4x32xf32>
    %collapsed_11 = tensor.collapse_shape %26 [[0], [1], [2, 3]] : tensor<1x128x4x32xf32> into tensor<1x128x128xf32>
    return %collapsed_11 : tensor<1x128x128xf32>
  }
}


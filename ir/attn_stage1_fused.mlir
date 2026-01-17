#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func public @main(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.176776692 : f32
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_2 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_3 = tensor.collapse_shape %arg2 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %collapsed_4 = tensor.collapse_shape %arg3 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.matmul ins(%collapsed, %collapsed_2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = linalg.matmul ins(%collapsed, %collapsed_3 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %3 = linalg.matmul ins(%collapsed, %collapsed_4 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
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
    %10 = linalg.batch_matmul ins(%collapsed_7, %collapsed_8 : tensor<4x128x32xf32>, tensor<4x32x128xf32>) outs(%9 : tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<4x128x128xf32>) outs(%9 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.mulf %in, %cst_1 : f32
      linalg.yield %23 : f32
    } -> tensor<4x128x128xf32>
    %12 = tensor.empty() : tensor<4x128xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<4x128xf32>) -> tensor<4x128xf32>
    %14 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11 : tensor<4x128x128xf32>) outs(%13 : tensor<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.cmpf ogt, %in, %out : f32
      %24 = arith.select %23, %in, %out : f32
      linalg.yield %24 : f32
    } -> tensor<4x128xf32>
    %15 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %14 : tensor<4x128x128xf32>, tensor<4x128xf32>) outs(%9 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %23 = arith.mulf %in, %cst_1 : f32
      %24 = arith.subf %23, %in_12 : f32
      %25 = math.exp %24 : f32
      linalg.yield %25 : f32
    } -> tensor<4x128x128xf32>
    %16 = linalg.fill ins(%cst : f32) outs(%12 : tensor<4x128xf32>) -> tensor<4x128xf32>
    %17 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : tensor<4x128x128xf32>) outs(%16 : tensor<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.addf %in, %out : f32
      linalg.yield %23 : f32
    } -> tensor<4x128xf32>
    %18 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %14, %17 : tensor<4x128x128xf32>, tensor<4x128xf32>, tensor<4x128xf32>) outs(%9 : tensor<4x128x128xf32>) {
    ^bb0(%in: f32, %in_12: f32, %in_13: f32, %out: f32):
      %23 = arith.mulf %in, %cst_1 : f32
      %24 = arith.subf %23, %in_12 : f32
      %25 = math.exp %24 : f32
      %26 = arith.divf %25, %in_13 : f32
      linalg.yield %26 : f32
    } -> tensor<4x128x128xf32>
    %19 = tensor.empty() : tensor<4x128x32xf32>
    %20 = linalg.batch_matmul ins(%18, %collapsed_9 : tensor<4x128x128xf32>, tensor<4x128x32xf32>) outs(%19 : tensor<4x128x32xf32>) -> tensor<4x128x32xf32>
    %expanded_10 = tensor.expand_shape %20 [[0, 1], [2], [3]] : tensor<4x128x32xf32> into tensor<1x4x128x32xf32>
    %21 = tensor.empty() : tensor<1x128x4x32xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_10 : tensor<1x4x128x32xf32>) outs(%21 : tensor<1x128x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x4x32xf32>
    %collapsed_11 = tensor.collapse_shape %22 [[0], [1], [2, 3]] : tensor<1x128x4x32xf32> into tensor<1x128x128xf32>
    return %collapsed_11 : tensor<1x128x128xf32>
  }
}


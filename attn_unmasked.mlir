module @jit_multihead_attn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128x128xf32>) -> (tensor<1x128x128xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x128xf32>, tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %1 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x128xf32>, tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %2 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x128xf32>, tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %3 = stablehlo.reshape %0 : (tensor<1x128x128xf32>) -> tensor<1x128x4x32xf32>
    %4 = stablehlo.reshape %1 : (tensor<1x128x128xf32>) -> tensor<1x128x4x32xf32>
    %5 = stablehlo.reshape %2 : (tensor<1x128x128xf32>) -> tensor<1x128x4x32xf32>
    %6 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<1x128x4x32xf32>) -> tensor<1x4x128x32xf32>
    %7 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<1x128x4x32xf32>) -> tensor<1x4x128x32xf32>
    %8 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<1x128x4x32xf32>) -> tensor<1x4x128x32xf32>
    %9 = stablehlo.transpose %7, dims = [0, 1, 3, 2] : (tensor<1x4x128x32xf32>) -> tensor<1x4x32x128xf32>
    %10 = stablehlo.reshape %6 : (tensor<1x4x128x32xf32>) -> tensor<4x128x32xf32>
    %11 = stablehlo.dot_general %10, %9, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x32xf32>, tensor<1x4x32x128xf32>) -> tensor<4x128x1x128xf32>
    %12 = stablehlo.transpose %11, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf32>) -> tensor<1x4x128x128xf32>
    %cst = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %13 = stablehlo.sqrt %cst : tensor<f32>
    %14 = stablehlo.convert %13 : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %16 = stablehlo.divide %12, %15 : tensor<1x4x128x128xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %17 = stablehlo.reduce(%16 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x4x128xf32>
    %19 = stablehlo.maximum %18, %17 : tensor<1x4x128xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %22 = stablehlo.subtract %16, %21 : tensor<1x4x128x128xf32>
    %23 = stablehlo.exponential %22 : tensor<1x4x128x128xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %27 = stablehlo.divide %23, %26 : tensor<1x4x128x128xf32>
    %28 = stablehlo.reshape %27 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %29 = stablehlo.dot_general %28, %8, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<1x4x128x32xf32>) -> tensor<4x128x1x32xf32>
    %30 = stablehlo.transpose %29, dims = [2, 0, 1, 3] : (tensor<4x128x1x32xf32>) -> tensor<1x4x128x32xf32>
    %31 = stablehlo.transpose %30, dims = [0, 2, 1, 3] : (tensor<1x4x128x32xf32>) -> tensor<1x128x4x32xf32>
    %32 = stablehlo.reshape %31 : (tensor<1x128x4x32xf32>) -> tensor<1x128x128xf32>
    return %32 : tensor<1x128x128xf32>
  }
}

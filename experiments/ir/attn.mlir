module @jit_masked_multihead_attn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x4x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>, %arg4: tensor<4x4xi1>) -> (tensor<1x4x8xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8xf32>, tensor<8x8xf32>) -> tensor<1x4x8xf32>
    %1 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8xf32>, tensor<8x8xf32>) -> tensor<1x4x8xf32>
    %2 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4x8xf32>, tensor<8x8xf32>) -> tensor<1x4x8xf32>
    %3 = stablehlo.reshape %0 : (tensor<1x4x8xf32>) -> tensor<1x4x1x8xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1, 3] : (tensor<1x4x1x8xf32>) -> tensor<1x1x4x8xf32>
    %5 = stablehlo.reshape %1 : (tensor<1x4x8xf32>) -> tensor<1x4x1x8xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<1x4x1x8xf32>) -> tensor<1x1x4x8xf32>
    %7 = stablehlo.reshape %2 : (tensor<1x4x8xf32>) -> tensor<1x4x1x8xf32>
    %8 = stablehlo.transpose %7, dims = [0, 2, 1, 3] : (tensor<1x4x1x8xf32>) -> tensor<1x1x4x8xf32>
    %9 = stablehlo.transpose %6, dims = [0, 1, 3, 2] : (tensor<1x1x4x8xf32>) -> tensor<1x1x8x4xf32>
    %10 = stablehlo.reshape %4 : (tensor<1x1x4x8xf32>) -> tensor<4x8xf32>
    %11 = stablehlo.dot_general %10, %9, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x8xf32>, tensor<1x1x8x4xf32>) -> tensor<4x1x1x4xf32>
    %12 = stablehlo.transpose %11, dims = [1, 2, 0, 3] : (tensor<4x1x1x4xf32>) -> tensor<1x1x4x4xf32>
    %cst = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %13 = stablehlo.sqrt %cst : tensor<f32>
    %14 = stablehlo.convert %13 : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<1x1x4x4xf32>
    %16 = stablehlo.divide %12, %15 : tensor<1x1x4x4xf32>
    %17 = stablehlo.broadcast_in_dim %arg4, dims = [2, 3] : (tensor<4x4xi1>) -> tensor<1x1x4x4xi1>
    %cst_0 = stablehlo.constant dense<-1.000000e+09> : tensor<f32>
    %18 = call @_where(%17, %16, %cst_0) : (tensor<1x1x4x4xi1>, tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x4x4xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %19 = stablehlo.reduce(%18 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %cst_2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
    %21 = stablehlo.maximum %20, %19 : tensor<1x1x4xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x1x4x1xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2, 3] : (tensor<1x1x4x1xf32>) -> tensor<1x1x4x4xf32>
    %24 = stablehlo.subtract %18, %23 : tensor<1x1x4x4xf32>
    %25 = stablehlo.exponential %24 : tensor<1x1x4x4xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.reduce(%25 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x1x4x1xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1, 2, 3] : (tensor<1x1x4x1xf32>) -> tensor<1x1x4x4xf32>
    %29 = stablehlo.divide %25, %28 : tensor<1x1x4x4xf32>
    %30 = stablehlo.reshape %29 : (tensor<1x1x4x4xf32>) -> tensor<4x4xf32>
    %31 = stablehlo.dot_general %30, %8, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<1x1x4x8xf32>) -> tensor<4x1x1x8xf32>
    %32 = stablehlo.transpose %31, dims = [1, 2, 0, 3] : (tensor<4x1x1x8xf32>) -> tensor<1x1x4x8xf32>
    %33 = stablehlo.transpose %32, dims = [0, 2, 1, 3] : (tensor<1x1x4x8xf32>) -> tensor<1x4x1x8xf32>
    %34 = stablehlo.reshape %33 : (tensor<1x4x1x8xf32>) -> tensor<1x4x8xf32>
    return %34 : tensor<1x4x8xf32>
  }
  func.func private @_where(%arg0: tensor<1x1x4x4xi1>, %arg1: tensor<1x1x4x4xf32>, %arg2: tensor<f32>) -> tensor<1x1x4x4xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x1x4x4xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<1x1x4x4xi1>, tensor<1x1x4x4xf32>
    return %2 : tensor<1x1x4x4xf32>
  }
}

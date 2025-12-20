module @jit_masked_multihead_attn attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x4x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>, %arg4: tensor<4x4xi1>) -> (tensor<1x4x8xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x4x8xf32>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c4_0 = arith.constant 4 : index
    %1 = scf.for %arg5 = %c0 to %c4 step %c4_0 iter_args(%arg6 = %0) -> (tensor<1x4x8xf32>) {
      %c0_7 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c4_8 = arith.constant 4 : index
      %43 = scf.for %arg7 = %c0_7 to %c8 step %c4_8 iter_args(%arg8 = %arg6) -> (tensor<1x4x8xf32>) {
        %c0_9 = arith.constant 0 : index
        %c8_10 = arith.constant 8 : index
        %c4_11 = arith.constant 4 : index
        %44 = stablehlo.constant dense<0.000000e+00> : tensor<1x4x4xf32>
        %45 = scf.for %arg9 = %c0_9 to %c8_10 step %c4_11 iter_args(%arg10 = %44) -> (tensor<1x4x4xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg5, %arg9] [1, 4, 4] [1, 1, 1] : tensor<1x4x8xf32> to tensor<1x4x4xf32>
          %extracted_slice_12 = tensor.extract_slice %arg1[%arg9, %arg7] [4, 4] [1, 1] : tensor<8x8xf32> to tensor<4x4xf32>
          %46 = stablehlo.dot_general %extracted_slice, %extracted_slice_12, contracting_dims = [2] x [0], precision = [] : (tensor<1x4x4xf32>, tensor<4x4xf32>) -> tensor<1x4x4xf32>
          %47 = stablehlo.add %arg10, %46 : tensor<1x4x4xf32>
          scf.yield %47 : tensor<1x4x4xf32>
        }
        %inserted_slice = tensor.insert_slice %45 into %arg8[0, %arg5, %arg7] [1, 4, 4] [1, 1, 1] : tensor<1x4x4xf32> into tensor<1x4x8xf32>
        scf.yield %inserted_slice : tensor<1x4x8xf32>
      }
      scf.yield %43 : tensor<1x4x8xf32>
    }
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<1x4x8xf32>
    %c0_1 = arith.constant 0 : index
    %c4_2 = arith.constant 4 : index
    %c4_3 = arith.constant 4 : index
    %3 = scf.for %arg5 = %c0_1 to %c4_2 step %c4_3 iter_args(%arg6 = %2) -> (tensor<1x4x8xf32>) {
      %c0_7 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c4_8 = arith.constant 4 : index
      %43 = scf.for %arg7 = %c0_7 to %c8 step %c4_8 iter_args(%arg8 = %arg6) -> (tensor<1x4x8xf32>) {
        %c0_9 = arith.constant 0 : index
        %c8_10 = arith.constant 8 : index
        %c4_11 = arith.constant 4 : index
        %44 = stablehlo.constant dense<0.000000e+00> : tensor<1x4x4xf32>
        %45 = scf.for %arg9 = %c0_9 to %c8_10 step %c4_11 iter_args(%arg10 = %44) -> (tensor<1x4x4xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg5, %arg9] [1, 4, 4] [1, 1, 1] : tensor<1x4x8xf32> to tensor<1x4x4xf32>
          %extracted_slice_12 = tensor.extract_slice %arg2[%arg9, %arg7] [4, 4] [1, 1] : tensor<8x8xf32> to tensor<4x4xf32>
          %46 = stablehlo.dot_general %extracted_slice, %extracted_slice_12, contracting_dims = [2] x [0], precision = [] : (tensor<1x4x4xf32>, tensor<4x4xf32>) -> tensor<1x4x4xf32>
          %47 = stablehlo.add %arg10, %46 : tensor<1x4x4xf32>
          scf.yield %47 : tensor<1x4x4xf32>
        }
        %inserted_slice = tensor.insert_slice %45 into %arg8[0, %arg5, %arg7] [1, 4, 4] [1, 1, 1] : tensor<1x4x4xf32> into tensor<1x4x8xf32>
        scf.yield %inserted_slice : tensor<1x4x8xf32>
      }
      scf.yield %43 : tensor<1x4x8xf32>
    }
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<1x4x8xf32>
    %c0_4 = arith.constant 0 : index
    %c4_5 = arith.constant 4 : index
    %c4_6 = arith.constant 4 : index
    %5 = scf.for %arg5 = %c0_4 to %c4_5 step %c4_6 iter_args(%arg6 = %4) -> (tensor<1x4x8xf32>) {
      %c0_7 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c4_8 = arith.constant 4 : index
      %43 = scf.for %arg7 = %c0_7 to %c8 step %c4_8 iter_args(%arg8 = %arg6) -> (tensor<1x4x8xf32>) {
        %c0_9 = arith.constant 0 : index
        %c8_10 = arith.constant 8 : index
        %c4_11 = arith.constant 4 : index
        %44 = stablehlo.constant dense<0.000000e+00> : tensor<1x4x4xf32>
        %45 = scf.for %arg9 = %c0_9 to %c8_10 step %c4_11 iter_args(%arg10 = %44) -> (tensor<1x4x4xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg5, %arg9] [1, 4, 4] [1, 1, 1] : tensor<1x4x8xf32> to tensor<1x4x4xf32>
          %extracted_slice_12 = tensor.extract_slice %arg3[%arg9, %arg7] [4, 4] [1, 1] : tensor<8x8xf32> to tensor<4x4xf32>
          %46 = stablehlo.dot_general %extracted_slice, %extracted_slice_12, contracting_dims = [2] x [0], precision = [] : (tensor<1x4x4xf32>, tensor<4x4xf32>) -> tensor<1x4x4xf32>
          %47 = stablehlo.add %arg10, %46 : tensor<1x4x4xf32>
          scf.yield %47 : tensor<1x4x4xf32>
        }
        %inserted_slice = tensor.insert_slice %45 into %arg8[0, %arg5, %arg7] [1, 4, 4] [1, 1, 1] : tensor<1x4x4xf32> into tensor<1x4x8xf32>
        scf.yield %inserted_slice : tensor<1x4x8xf32>
      }
      scf.yield %43 : tensor<1x4x8xf32>
    }
    %6 = stablehlo.reshape %1 : (tensor<1x4x8xf32>) -> tensor<1x4x1x8xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1, 3] : (tensor<1x4x1x8xf32>) -> tensor<1x1x4x8xf32>
    %8 = stablehlo.reshape %3 : (tensor<1x4x8xf32>) -> tensor<1x4x1x8xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 1, 3] : (tensor<1x4x1x8xf32>) -> tensor<1x1x4x8xf32>
    %10 = stablehlo.reshape %5 : (tensor<1x4x8xf32>) -> tensor<1x4x1x8xf32>
    %11 = stablehlo.transpose %10, dims = [0, 2, 1, 3] : (tensor<1x4x1x8xf32>) -> tensor<1x1x4x8xf32>
    %12 = stablehlo.transpose %9, dims = [0, 1, 3, 2] : (tensor<1x1x4x8xf32>) -> tensor<1x1x8x4xf32>
    %13 = stablehlo.reshape %7 : (tensor<1x1x4x8xf32>) -> tensor<4x8xf32>
    %14 = stablehlo.dot_general %13, %12, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x8xf32>, tensor<1x1x8x4xf32>) -> tensor<4x1x1x4xf32>
    %15 = stablehlo.transpose %14, dims = [1, 2, 0, 3] : (tensor<4x1x1x4xf32>) -> tensor<1x1x4x4xf32>
    %16 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %17 = stablehlo.sqrt %16 : tensor<f32>
    %18 = stablehlo.convert %17 : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<1x1x4x4xf32>
    %20 = stablehlo.divide %15, %19 : tensor<1x1x4x4xf32>
    %21 = stablehlo.broadcast_in_dim %arg4, dims = [2, 3] : (tensor<4x4xi1>) -> tensor<1x1x4x4xi1>
    %22 = stablehlo.constant dense<-1.000000e+09> : tensor<f32>
    %23 = call @_where(%21, %20, %22) : (tensor<1x1x4x4xi1>, tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x4x4xf32>
    %24 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %25 = stablehlo.reduce(%23 init: %24) applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %26 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
    %28 = stablehlo.maximum %27, %25 : tensor<1x1x4xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x1x4x1xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 3] : (tensor<1x1x4x1xf32>) -> tensor<1x1x4x4xf32>
    %31 = stablehlo.subtract %23, %30 : tensor<1x1x4x4xf32>
    %32 = stablehlo.exponential %31 : tensor<1x1x4x4xf32>
    %33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.reduce(%32 init: %33) applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x4xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x1x4x1xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x1x4x1xf32>) -> tensor<1x1x4x4xf32>
    %37 = stablehlo.divide %32, %36 : tensor<1x1x4x4xf32>
    %38 = stablehlo.reshape %37 : (tensor<1x1x4x4xf32>) -> tensor<4x4xf32>
    %39 = stablehlo.dot_general %38, %11, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<1x1x4x8xf32>) -> tensor<4x1x1x8xf32>
    %40 = stablehlo.transpose %39, dims = [1, 2, 0, 3] : (tensor<4x1x1x8xf32>) -> tensor<1x1x4x8xf32>
    %41 = stablehlo.transpose %40, dims = [0, 2, 1, 3] : (tensor<1x1x4x8xf32>) -> tensor<1x4x1x8xf32>
    %42 = stablehlo.reshape %41 : (tensor<1x4x1x8xf32>) -> tensor<1x4x8xf32>
    return %42 : tensor<1x4x8xf32>
  }
  func.func private @_where(%arg0: tensor<1x1x4x4xi1>, %arg1: tensor<1x1x4x4xf32>, %arg2: tensor<f32>) -> tensor<1x1x4x4xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x1x4x4xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<1x1x4x4xi1>, tensor<1x1x4x4xf32>
    return %2 : tensor<1x1x4x4xf32>
  }
}


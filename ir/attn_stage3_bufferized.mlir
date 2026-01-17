#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map3 = affine_map<(d0) -> (-d0 + 32, 4)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func public @main(%arg0: memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>, %arg2: memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>, %arg3: memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>>) -> memref<1x128x128xf32> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.176776692 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x128xf32>
    memref.copy %arg0, %alloc : memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>> to memref<1x128x128xf32>
    %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2]] : memref<1x128x128xf32> into memref<128x128xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x128x128xf32>
    memref.copy %arg1, %alloc_2 : memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>> to memref<1x128x128xf32>
    %collapse_shape_3 = memref.collapse_shape %alloc_2 [[0, 1], [2]] : memref<1x128x128xf32> into memref<128x128xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x128x128xf32>
    memref.copy %arg2, %alloc_4 : memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>> to memref<1x128x128xf32>
    %collapse_shape_5 = memref.collapse_shape %alloc_4 [[0, 1], [2]] : memref<1x128x128xf32> into memref<128x128xf32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x128x128xf32>
    memref.copy %arg3, %alloc_6 : memref<1x128x128xf32, strided<[?, ?, ?], offset: ?>> to memref<1x128x128xf32>
    %collapse_shape_7 = memref.collapse_shape %alloc_6 [[0, 1], [2]] : memref<1x128x128xf32> into memref<128x128xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    scf.for %arg4 = %c0 to %c128 step %c64 {
      scf.for %arg5 = %c0 to %c128 step %c64 {
        scf.for %arg6 = %c0 to %c128 step %c64 {
          %subview = memref.subview %collapse_shape[%arg4, %arg6] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
          %subview_27 = memref.subview %collapse_shape_3[%arg6, %arg5] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
          %subview_28 = memref.subview %alloc_9[%arg4, %arg5] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
          scf.for %arg7 = %c0 to %c64 step %c4 {
            scf.for %arg8 = %c0 to %c64 step %c4 {
              scf.for %arg9 = %c0 to %c64 step %c4 {
                %subview_29 = memref.subview %subview[%arg7, %arg9] [4, 4] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
                %subview_30 = memref.subview %subview_27[%arg9, %arg8] [4, 4] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
                %subview_31 = memref.subview %subview_28[%arg7, %arg8] [4, 4] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
                linalg.matmul {done_tiling} ins(%subview_29, %subview_30 : memref<4x4xf32, strided<[128, 1], offset: ?>>, memref<4x4xf32, strided<[128, 1], offset: ?>>) outs(%subview_31 : memref<4x4xf32, strided<[128, 1], offset: ?>>)
                memref.copy %subview_31, %subview_31 : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
              } {done_tiling}
            } {done_tiling}
          } {done_tiling}
          memref.copy %subview_28, %subview_28 : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<64x64xf32, strided<[128, 1], offset: ?>>
        } {done_tiling}
      } {done_tiling}
    } {done_tiling}
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    scf.for %arg4 = %c0 to %c128 step %c64 {
      scf.for %arg5 = %c0 to %c128 step %c64 {
        scf.for %arg6 = %c0 to %c128 step %c64 {
          %subview = memref.subview %collapse_shape[%arg4, %arg6] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
          %subview_27 = memref.subview %collapse_shape_5[%arg6, %arg5] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
          %subview_28 = memref.subview %alloc_10[%arg4, %arg5] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
          scf.for %arg7 = %c0 to %c64 step %c4 {
            scf.for %arg8 = %c0 to %c64 step %c4 {
              scf.for %arg9 = %c0 to %c64 step %c4 {
                %subview_29 = memref.subview %subview[%arg7, %arg9] [4, 4] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
                %subview_30 = memref.subview %subview_27[%arg9, %arg8] [4, 4] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
                %subview_31 = memref.subview %subview_28[%arg7, %arg8] [4, 4] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
                linalg.matmul {done_tiling} ins(%subview_29, %subview_30 : memref<4x4xf32, strided<[128, 1], offset: ?>>, memref<4x4xf32, strided<[128, 1], offset: ?>>) outs(%subview_31 : memref<4x4xf32, strided<[128, 1], offset: ?>>)
                memref.copy %subview_31, %subview_31 : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
              } {done_tiling}
            } {done_tiling}
          } {done_tiling}
          memref.copy %subview_28, %subview_28 : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<64x64xf32, strided<[128, 1], offset: ?>>
        } {done_tiling}
      } {done_tiling}
    } {done_tiling}
    scf.for %arg4 = %c0 to %c128 step %c64 {
      scf.for %arg5 = %c0 to %c128 step %c64 {
        scf.for %arg6 = %c0 to %c128 step %c64 {
          %subview = memref.subview %collapse_shape[%arg4, %arg6] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
          %subview_27 = memref.subview %collapse_shape_7[%arg6, %arg5] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
          %subview_28 = memref.subview %alloc_8[%arg4, %arg5] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
          scf.for %arg7 = %c0 to %c64 step %c4 {
            scf.for %arg8 = %c0 to %c64 step %c4 {
              scf.for %arg9 = %c0 to %c64 step %c4 {
                %subview_29 = memref.subview %subview[%arg7, %arg9] [4, 4] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
                %subview_30 = memref.subview %subview_27[%arg9, %arg8] [4, 4] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
                %subview_31 = memref.subview %subview_28[%arg7, %arg8] [4, 4] [1, 1] : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
                linalg.matmul {done_tiling} ins(%subview_29, %subview_30 : memref<4x4xf32, strided<[128, 1], offset: ?>>, memref<4x4xf32, strided<[128, 1], offset: ?>>) outs(%subview_31 : memref<4x4xf32, strided<[128, 1], offset: ?>>)
                memref.copy %subview_31, %subview_31 : memref<4x4xf32, strided<[128, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>>
              } {done_tiling}
            } {done_tiling}
          } {done_tiling}
          memref.copy %subview_28, %subview_28 : memref<64x64xf32, strided<[128, 1], offset: ?>> to memref<64x64xf32, strided<[128, 1], offset: ?>>
        } {done_tiling}
      } {done_tiling}
    } {done_tiling}
    %expand_shape = memref.expand_shape %alloc_9 [[0, 1], [2, 3]] : memref<128x128xf32> into memref<1x128x4x32xf32>
    %expand_shape_11 = memref.expand_shape %alloc_10 [[0, 1], [2, 3]] : memref<128x128xf32> into memref<1x128x4x32xf32>
    %expand_shape_12 = memref.expand_shape %alloc_8 [[0, 1], [2, 3]] : memref<128x128xf32> into memref<1x128x4x32xf32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x4x128x32xf32>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x4x128x32xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expand_shape : memref<1x128x4x32xf32>) outs(%alloc_14 : memref<1x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x4x32x128xf32>
    linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expand_shape_11 : memref<1x128x4x32xf32>) outs(%alloc_15 : memref<1x4x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expand_shape_12 : memref<1x128x4x32xf32>) outs(%alloc_13 : memref<1x4x128x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %collapse_shape_16 = memref.collapse_shape %alloc_14 [[0, 1], [2], [3]] : memref<1x4x128x32xf32> into memref<4x128x32xf32>
    %collapse_shape_17 = memref.collapse_shape %alloc_15 [[0, 1], [2], [3]] : memref<1x4x32x128xf32> into memref<4x32x128xf32>
    %collapse_shape_18 = memref.collapse_shape %alloc_13 [[0, 1], [2], [3]] : memref<1x4x128x32xf32> into memref<4x128x32xf32>
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<4x128x128xf32>
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<4x128x128xf32>
    scf.for %arg4 = %c0 to %c128 step %c64 {
      scf.for %arg5 = %c0 to %c128 step %c64 {
        %subview = memref.subview %collapse_shape_16[0, %arg4, 0] [4, 64, 32] [1, 1, 1] : memref<4x128x32xf32> to memref<4x64x32xf32, strided<[4096, 32, 1], offset: ?>>
        %subview_27 = memref.subview %collapse_shape_17[0, 0, %arg5] [4, 32, 64] [1, 1, 1] : memref<4x32x128xf32> to memref<4x32x64xf32, strided<[4096, 128, 1], offset: ?>>
        %subview_28 = memref.subview %alloc_20[0, %arg4, %arg5] [4, 64, 64] [1, 1, 1] : memref<4x128x128xf32> to memref<4x64x64xf32, strided<[16384, 128, 1], offset: ?>>
        scf.for %arg6 = %c0 to %c64 step %c4 {
          scf.for %arg7 = %c0 to %c64 step %c4 {
            scf.for %arg8 = %c0 to %c32 step %c4 {
              %0 = affine.min #map3(%arg8)
              %subview_29 = memref.subview %subview[0, %arg6, %arg8] [4, 4, %0] [1, 1, 1] : memref<4x64x32xf32, strided<[4096, 32, 1], offset: ?>> to memref<4x4x?xf32, strided<[4096, 32, 1], offset: ?>>
              %subview_30 = memref.subview %subview_27[0, %arg8, %arg7] [4, %0, 4] [1, 1, 1] : memref<4x32x64xf32, strided<[4096, 128, 1], offset: ?>> to memref<4x?x4xf32, strided<[4096, 128, 1], offset: ?>>
              %subview_31 = memref.subview %subview_28[0, %arg6, %arg7] [4, 4, 4] [1, 1, 1] : memref<4x64x64xf32, strided<[16384, 128, 1], offset: ?>> to memref<4x4x4xf32, strided<[16384, 128, 1], offset: ?>>
              linalg.batch_matmul {done_tiling} ins(%subview_29, %subview_30 : memref<4x4x?xf32, strided<[4096, 32, 1], offset: ?>>, memref<4x?x4xf32, strided<[4096, 128, 1], offset: ?>>) outs(%subview_31 : memref<4x4x4xf32, strided<[16384, 128, 1], offset: ?>>)
              memref.copy %subview_31, %subview_31 : memref<4x4x4xf32, strided<[16384, 128, 1], offset: ?>> to memref<4x4x4xf32, strided<[16384, 128, 1], offset: ?>>
            } {done_tiling}
          } {done_tiling}
        } {done_tiling}
        memref.copy %subview_28, %subview_28 : memref<4x64x64xf32, strided<[16384, 128, 1], offset: ?>> to memref<4x64x64xf32, strided<[16384, 128, 1], offset: ?>>
      } {done_tiling}
    } {done_tiling}
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_20 : memref<4x128x128xf32>) outs(%alloc_19 : memref<4x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.mulf %in, %cst_1 : f32
      linalg.yield %0 : f32
    }
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<4x128xf32>
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<4x128xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_22 : memref<4x128xf32>)
    linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc_19 : memref<4x128x128xf32>) outs(%alloc_22 : memref<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf ogt, %in, %out : f32
      %1 = arith.select %0, %in, %out : f32
      linalg.yield %1 : f32
    }
    linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_20, %alloc_22 : memref<4x128x128xf32>, memref<4x128xf32>) outs(%alloc_19 : memref<4x128x128xf32>) {
    ^bb0(%in: f32, %in_27: f32, %out: f32):
      %0 = arith.mulf %in, %cst_1 : f32
      %1 = arith.subf %0, %in_27 : f32
      %2 = math.exp %1 : f32
      linalg.yield %2 : f32
    }
    linalg.fill ins(%cst : f32) outs(%alloc_21 : memref<4x128xf32>)
    linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc_19 : memref<4x128x128xf32>) outs(%alloc_21 : memref<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    }
    linalg.generic {indexing_maps = [#map4, #map5, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_20, %alloc_22, %alloc_21 : memref<4x128x128xf32>, memref<4x128xf32>, memref<4x128xf32>) outs(%alloc_19 : memref<4x128x128xf32>) {
    ^bb0(%in: f32, %in_27: f32, %in_28: f32, %out: f32):
      %0 = arith.mulf %in, %cst_1 : f32
      %1 = arith.subf %0, %in_27 : f32
      %2 = math.exp %1 : f32
      %3 = arith.divf %2, %in_28 : f32
      linalg.yield %3 : f32
    }
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<4x128x32xf32>
    scf.for %arg4 = %c0 to %c128 step %c64 {
      scf.for %arg5 = %c0 to %c128 step %c64 {
        %subview = memref.subview %alloc_19[0, %arg4, %arg5] [4, 64, 64] [1, 1, 1] : memref<4x128x128xf32> to memref<4x64x64xf32, strided<[16384, 128, 1], offset: ?>>
        %subview_27 = memref.subview %collapse_shape_18[0, %arg5, 0] [4, 64, 32] [1, 1, 1] : memref<4x128x32xf32> to memref<4x64x32xf32, strided<[4096, 32, 1], offset: ?>>
        %subview_28 = memref.subview %alloc_23[0, %arg4, 0] [4, 64, 32] [1, 1, 1] : memref<4x128x32xf32> to memref<4x64x32xf32, strided<[4096, 32, 1], offset: ?>>
        scf.for %arg6 = %c0 to %c64 step %c4 {
          scf.for %arg7 = %c0 to %c32 step %c4 {
            scf.for %arg8 = %c0 to %c64 step %c4 {
              %0 = affine.min #map3(%arg7)
              %subview_29 = memref.subview %subview[0, %arg6, %arg8] [4, 4, 4] [1, 1, 1] : memref<4x64x64xf32, strided<[16384, 128, 1], offset: ?>> to memref<4x4x4xf32, strided<[16384, 128, 1], offset: ?>>
              %subview_30 = memref.subview %subview_27[0, %arg8, %arg7] [4, 4, %0] [1, 1, 1] : memref<4x64x32xf32, strided<[4096, 32, 1], offset: ?>> to memref<4x4x?xf32, strided<[4096, 32, 1], offset: ?>>
              %subview_31 = memref.subview %subview_28[0, %arg6, %arg7] [4, 4, %0] [1, 1, 1] : memref<4x64x32xf32, strided<[4096, 32, 1], offset: ?>> to memref<4x4x?xf32, strided<[4096, 32, 1], offset: ?>>
              linalg.batch_matmul {done_tiling} ins(%subview_29, %subview_30 : memref<4x4x4xf32, strided<[16384, 128, 1], offset: ?>>, memref<4x4x?xf32, strided<[4096, 32, 1], offset: ?>>) outs(%subview_31 : memref<4x4x?xf32, strided<[4096, 32, 1], offset: ?>>)
              memref.copy %subview_31, %subview_31 : memref<4x4x?xf32, strided<[4096, 32, 1], offset: ?>> to memref<4x4x?xf32, strided<[4096, 32, 1], offset: ?>>
            } {done_tiling}
          } {done_tiling}
        } {done_tiling}
        memref.copy %subview_28, %subview_28 : memref<4x64x32xf32, strided<[4096, 32, 1], offset: ?>> to memref<4x64x32xf32, strided<[4096, 32, 1], offset: ?>>
      } {done_tiling}
    } {done_tiling}
    %expand_shape_24 = memref.expand_shape %alloc_23 [[0, 1], [2], [3]] : memref<4x128x32xf32> into memref<1x4x128x32xf32>
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<1x128x4x32xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expand_shape_24 : memref<1x4x128x32xf32>) outs(%alloc_25 : memref<1x128x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %collapse_shape_26 = memref.collapse_shape %alloc_25 [[0], [1], [2, 3]] : memref<1x128x4x32xf32> into memref<1x128x128xf32>
    memref.dealloc %alloc_8 : memref<128x128xf32>
    memref.dealloc %alloc_9 : memref<128x128xf32>
    memref.dealloc %alloc_10 : memref<128x128xf32>
    memref.dealloc %alloc_13 : memref<1x4x128x32xf32>
    memref.dealloc %alloc_14 : memref<1x4x128x32xf32>
    memref.dealloc %alloc_15 : memref<1x4x32x128xf32>
    memref.dealloc %alloc_19 : memref<4x128x128xf32>
    memref.dealloc %alloc_20 : memref<4x128x128xf32>
    memref.dealloc %alloc_21 : memref<4x128xf32>
    memref.dealloc %alloc_22 : memref<4x128xf32>
    memref.dealloc %alloc_23 : memref<4x128x32xf32>
    return %collapse_shape_26 : memref<1x128x128xf32>
  }
}


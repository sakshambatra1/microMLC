module {
  func.func @main(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>) -> tensor<8x8xf32> {
    // NOTICE THE QUOTES BELOW: "stablehlo.dot_general"
    %res = "stablehlo.dot_general"(%lhs, %rhs) <{
      dot_dimension_numbers = #stablehlo.dot_dimension_numbers<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = []
    }> : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    
    return %res : tensor<8x8xf32>
  }
}
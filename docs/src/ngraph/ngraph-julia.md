# nGraph Ops

## Add (`op_add`)

Elementwise addition operation. 

Arguments

- `arg0 - ngraph::Node` -- Node that produces the first input tensor.
- `arg1 - ngraph::Node` -- Node the produces the second input tensor.

Output
    
- `ngraph::Node`


## Avgpool (`op_avgpool`)

Arguments

- `arg - ngraph::Node` -- The node producing the input dat batch tensor.
- `window_shape - ngraph::Shape&` -- The window shape.
- `window_movement_strides - ngraph::Strides&` -- The window movement strides.
- `padding_below - ngraph::Shape&` -- The below-padding shape.
- `padding_above - ngraph::Shape&` -- The above-padding shape.

Output

- `ngraph::Node`


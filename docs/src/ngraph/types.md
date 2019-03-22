# Exposed nGraph Types

The header format is `C++ type name` -> `nGraph.Lib type name`.

## `element::Type`

**Julia Type**: `NGraphType`

Exposes the element type Julia. The ngraph compiler uses a static global reference for
each of the defined element types. The `gettype` function described below provides a
way of going from a string descriptor to the actual type reference.

### Methods

- `c_type_name`: Return the C type name of the element.

### Additional Functions

```c++
ngraph::element& gettype(std::string type)
```
Return the static reference for the element type reference for `type`. Valid values of 
type are: `boolean, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64`


## `CoordinateDiff`

**Julia Type**: `CoordinateDiff`

### Additional Functions

```c++
ngraph::CoordinateDiff CoordinateDiff(const jlcxx::ArrayRef<Int64_t, 1> vals)
```
Return a `ngraph::CoordinateDiff` constructed from the native Julia array.


## Shape

**Julia Type**: `Shape`

### Additional Functions

```c++
ngraph::Shape Shape(const jlcxx::ArrayRef<Int64_t, 1> vals)
```
Return a `ngraph::Shape` construct from a native Julia array.

```c++
int64_t _length(const ngraph::Shape shape)
```
Return the length of `shape`.

```c++
int64_t _getindex(const ngraph::Shape s, int64_t i)
```
Return the `i`th element of `s`.


## Strides

**Julia Type**: `Strides`

### Additional Functions

```c++
ngraph::Strides Strides(const jlcxx::ArrayRef<Int64_t, 1> vals)
```
Return a `ngraph::Strides` construct from a native Julia array.



## AxisSet

**Julia Type**: `AxisSet`

### Additional Functions

```c++
ngraph::AxisSet AxisSet(const jlcxx::ArrayRef<Int64_t, 1> vals)
```
Return a `ngraph::AxisSet` construct from a native Julia array.


## AxisVector

**Julia Type**: `AxisVector`

### Additional Functions

```c++
ngraph::AxisVector AxisVector(const jlcxx::ArrayRef<Int64_t, 1> vals)
```
Return a `ngraph::AxisVector` construct from a native Julia array.


## runtime::Tensor

**Julia Type**: `RuntimeTensor`

### Methods

```c++
ngraph::Shape get_shape(ngraph::runtime::Tensor T)
```
Return the `ngraph::Shape` of `T`

### Additional Functions

```c++
void tensor_write(std::shared_ptr<ngraph::runtime::Tensor> tensor,
    void* p,
    size_t offset,
    size_t n)
```
Write `n` bytes starting at `p` into `tensor`. Writes begin at `offset` in the tensor
storage.

```c++
void tensor_read(std::shared_ptr<ngraph::runtime::Tensor> tensor,
    void* p
    size_t offset,
    size_t n)
```
Read `n` bytes from `tensor` into memory starting at `p`. Read begins at `offset` in
tensor storage.


## descriptor::Tensor

**Julia Type**: `DescriptorTensor`

Descriptor of the Tensor, mainly contains information about the element type, offset
in pool, size, shape etc.

### Methods

```c++
size_t _sizeof(ngraph::descriptor::Tensor tensor)
```
Return the size of `tensor` in bytes.

```c++
std::string get_name(ngraph::descriptor::Tensor tensor)
```
Return the unique name of `tensor`.

## Node

**Julia Type**: `Node`

### Methods

```c++
std::string get_name(ngraph::Node node)
```
Return the unique name of `node`.

```c++
std::string description(ngraph::Node node)
```
Return the description of `node`. All nodes with the same `Op` will have the same
description.

```c++
size_t get_output_size(ngraph::Node node)
```
Return the number of outputs of `node`.

```c++
type::element& get_output_element_type(ngraph::Node node, size_t i)
```
Return the output element type for output `i` of `node`.

```c++
ngraph::Shape get_output_shape(ngraph::Node node, size_t i)
```
Return the shape of output `i` of `node`.

```c++
size_t get_input_size(ngraph::Node node)
```
Return the number of inputs of `node`.

```c++
type::element& get_input_element_type(ngraph::Node node, size_t i)
```
Return the element type for input `i` of `node`.

```c++
ngraph::Shape get_input_shape(ngraph::Node node, size_t i)
```
Return the shape of input `i` of `node`.

```c++
ngraph::Node get_input_node(ngraph::Node node, int64_t index)
```
Return the node who generates the `index` input tensor of `node`.

```c++
NodeWrapper get_output_nodes(ngraph::Node node, int64_t index)
```
Return the collection of nodes who use the `index` output tensor of `node` as an input.

### Additional Functions

```c++
descriptor::Tensor get_output_tensor_ptr(ngraph::Node node, int64_t index)
```
Return the descriptor for output `index` of `node`.

```c++
descriptor::Tensor get_input_tensor_ptr(ngraph::Node node, int64_t index)
```
Return the descriptor for input `index` of `node`.

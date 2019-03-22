# Custom Defined Types

## NodeWrapper

Transferring vectors of custom types from C++ to Julia is unreasonably tricky. To help
facilitate this, the `NodeWrapper` is just a wrapper type for
```
std::vector<std::shared_ptr<ngraph::Node>>
```
that exposes some iteration methods.

### Constructor

```c++
template <class Iter>
NodeWrapper(Iter start, Iter stop);

NodeWrapper(std::vector <std::shared_ptr <ngraph::Node> > nodes);
```

### Methods

- `std::shared_ptr<ngraph::Node> _getindex(int64_t i)` - Return the `i`th node in
    the vector.

- `int64_t _length()` - Return the length of the `NodeWrapper`.

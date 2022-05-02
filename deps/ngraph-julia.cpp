// stdlib
#include <vector>
#include <iostream>
#include <type_traits>

// jlcxx
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"

// ngraph
#include "ngraph/ngraph.hpp"

// Convert a Julia array of UInt8 to a std::vector of type `T`.
//
// Used for the construction of Constants.
template<typename T>
std::vector<T> castcollect(jlcxx::ArrayRef<uint8_t> x)
{
    // Get the raw data from the array and reinterpret it to T*
    const T* ptr = reinterpret_cast<const T*>(x.data());
    size_t len = x.size() / sizeof(T);

    std::vector<T> rc;
    for (size_t i = 0; i < len; i++)
    {
        rc.push_back(ptr[i]);
    }
    return rc;
}

template<typename T, typename U>
std::vector<T> convertcollect(jlcxx::ArrayRef<U> x)
{
    std::vector<T> rc;
    for (auto i: x)
    {
        rc.push_back(i);
    }
    return rc;
}

// Map the random nGraph Vector Types to normal std::vectors
//
// This way, we only need to deal with `std::vectors` on the Julia size.
template<typename T>
auto tovector(T& x)
{
    return std::vector<typename T::value_type>(x.begin(), x.end());
}

template<>
auto tovector(ngraph::AxisSet& x)
{
    return x.to_vector();
}

// Since C++ does not allow partial function template specialization, we use this
// intermediate `type` struct to propagate type information into function arguments
// in order to correctly dispatch between the various ngraph `std::vector` derived
// classes and other constructs like `ngraph::AxisSet`.
template <typename T>
struct type{};

template<typename T, typename U>
T construct(type<T>, U x)
{
    return T(x.begin(), x.end());
}

template<typename U>
ngraph::AxisSet construct(type<ngraph::AxisSet>, U x)
{
    return ngraph::AxisSet(convertcollect<size_t>(x));
}

template<typename T, typename U>
T construct(U x)
{
    return construct(type<T>(), x);
}

// Convert ops into basic `nodes`
template<typename T>
auto tonode(std::shared_ptr<T> x)
{
    return std::dynamic_pointer_cast<ngraph::Node>(x);
}

/////
///// Module Wrapping
/////

#define TYPEIF(v,t) \
    if (v == ngraph::element::Type_t::t) {return &ngraph::element::t;}

// Forward to node creation and auto-cast to `ngraph::Node`
#define makeop(name,...) tonode(std::make_shared<ngraph::op::name>(__VA_ARGS__))

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    /////
    ///// Elements
    /////

    // The basic strategy is to map back and forth using enum values.
    mod.add_type<ngraph::element::Type>("Element")
        .method("c_type_name", &ngraph::element::Type::c_type_string);

    mod.method("Type_t", [](const ngraph::element::Type& x){
        return static_cast<int32_t>(ngraph::element::Type_t(x));
    });

    // Each of the ngraph types has a Enum value.
    // Here, we take the integer value from the Julia shadow of the type selector enum,
    // cast it to the C++ enum and use that to return the reference to the correct type.
    mod.method("ngraph_type", [](int32_t enum_value){
        auto type_t = static_cast<ngraph::element::Type_t>(enum_value);

        // TODO: Switch?
        TYPEIF(type_t, boolean)
        else TYPEIF(type_t, f32)
        else TYPEIF(type_t, f64)
        else TYPEIF(type_t, i8)
        else TYPEIF(type_t, i16)
        else TYPEIF(type_t, i32)
        else TYPEIF(type_t, i64)
        else TYPEIF(type_t, u8)
        else TYPEIF(type_t, u16)
        else TYPEIF(type_t, u32)
        else TYPEIF(type_t, u64)
        else {return &ngraph::element::dynamic;}
    });

    /////
    ///// Node
    /////

    mod.add_type<ngraph::Node>("Node")
        .method("get_output_size", &ngraph::Node::get_output_size)
        .method("get_output_shape", [](
            const std::shared_ptr<ngraph::Node> node,
            int64_t index)
        {
            return tovector(node->output(index).get_shape());
        })
        .method("get_output_element_type", &ngraph::Node::get_output_element_type)
        .method("get_name", &ngraph::Node::get_name)
        .method("description", &ngraph::Node::description);

    /////
    ///// NodeOutput
    /////

    mod.add_type<ngraph::Output<ngraph::Node>>("NodeOutput")
        .constructor<const std::shared_ptr<ngraph::Node>&,size_t>()
        .method("get_node", &ngraph::Output<ngraph::Node>::get_node_shared_ptr)
        .method("get_index", &ngraph::Output<ngraph::Node>::get_index)
        .method("get_shape", [](const ngraph::Output<ngraph::Node> output)
        {
            return tovector(output.get_shape());
        });

    // We occaisionally hand back `std::vectors` of `Node` shared_pointers.
    // Here, we opt into the stl in CxxWrap.
    jlcxx::stl::apply_stl<std::shared_ptr<ngraph::Node>>(mod);

    /////
    ///// Function
    /////

    mod.add_type<ngraph::Function>("NGFunction")
        .method("get_name", &ngraph::Function::get_name)
        .method("get_temporary_pool_size", &ngraph::Function::get_temporary_pool_size)
        // Parameters and Ops get converted to std::vectors
        .method("get_parameters", [](const std::shared_ptr<ngraph::Function>& fn)
        {
            // Need to convert `ngraph::op::Parameter` to `ngraph::Node`
            auto parameters = fn->get_parameters();
            auto x = std::vector<std::shared_ptr<ngraph::Node>>(parameters.size());
            std::transform(
                parameters.begin(),
                parameters.end(),
                x.begin(),
                &tonode<ngraph::op::Parameter>
            );
            return x;
        })
        .method("get_results", [](const std::shared_ptr<ngraph::Function>& fn)
        {
            // Need to convert `ngraph::op::Result` to just `ngraph::Node`
            auto results = fn->get_results();
            auto x = std::vector<std::shared_ptr<ngraph::Node>>(results.size());
            std::transform(
                results.begin(),
                results.end(),
                x.begin(),
                &tonode<ngraph::op::Result>
            );
            return x;
        });

    mod.method("make_function", [](
        const jlcxx::ArrayRef<std::shared_ptr<ngraph::Node>,1> jl_results,
        const jlcxx::ArrayRef<std::shared_ptr<ngraph::Node>,1> jl_parameters)
    {
        // Convert the Julia Arrays of results and parameters to the correct types
        auto results = ngraph::OutputVector(jl_results.begin(), jl_results.end());

        // For the Parameters, we have to cast the nodes to `Parameters`
        auto op = [](std::shared_ptr<ngraph::Node> x)
        {
            return std::dynamic_pointer_cast<ngraph::op::Parameter>(x);
        };
        auto parameters = ngraph::ParameterVector(jl_parameters.size());
        std::transform(jl_parameters.begin(), jl_parameters.end(), parameters.begin(), op);

        return std::make_shared<ngraph::Function>(results, parameters);
    });

    /////
    ///// runtime::Tensor
    /////

    // Define before `Executable`, since these come as arguments to the `Executable`.
    mod.add_type<ngraph::runtime::Tensor>("RuntimeTensor")
        .method("get_shape", [](const std::shared_ptr<ngraph::runtime::Tensor> tensor)
        {
            return tovector(tensor->get_shape());
        })
        .method("get_size_in_bytes", &ngraph::runtime::Tensor::get_size_in_bytes)
        .method("get_element_type", &ngraph::runtime::Tensor::get_element_type)
        .method("get_name", &ngraph::runtime::Tensor::get_name);

    /////
    ///// Executable
    /////

    // Needs to be defined before `backend->compile` because `compile` returns an Executable.
    mod.add_type<ngraph::runtime::Executable>("Executable")
        // TODO: We might be able to optimize this by pre-creating the `std::vector`
        // and just passing those.
        .method("call", [](const std::shared_ptr<ngraph::runtime::Executable> executable,
                    const jlcxx::ArrayRef<std::shared_ptr<ngraph::runtime::Tensor>> jl_outputs,
                    const jlcxx::ArrayRef<std::shared_ptr<ngraph::runtime::Tensor>> jl_inputs)
        {
            auto inputs = std::vector<std::shared_ptr<ngraph::runtime::Tensor>>(
                jl_inputs.begin(),
                jl_inputs.end()
            );

            auto outputs = std::vector<std::shared_ptr<ngraph::runtime::Tensor>>(
                jl_outputs.begin(),
                jl_outputs.end()
            );

            executable->call(outputs, inputs);
        });

    /////
    ///// Backend
    /////

    mod.add_type<ngraph::runtime::Backend>("Backend")
        .method("compile", [](
            const std::shared_ptr<ngraph::runtime::Backend>& backend,
            const std::shared_ptr<ngraph::Function>& func,
            bool enable_performance_data)
        {
            return backend->compile(func, enable_performance_data);
        })
        .method("remove_compiled_function", &ngraph::runtime::Backend::remove_compiled_function)
        .method("get_version", &ngraph::runtime::Backend::get_version);

    mod.method("create", [](const std::string& type){
        return ngraph::runtime::Backend::create(type);
    });


    /////
    ///// Misc Methods
    /////
    //
    // Methods that require the above types to be first declared.

    mod.method("create_tensor", [](
        const std::shared_ptr<ngraph::runtime::Backend> backend,
        const ngraph::element::Type& element_type,
        const jlcxx::ArrayRef<int64_t> jl_shape,
        void* ptr)
    {
        return backend->create_tensor(
            element_type,
            construct<ngraph::Shape>(jl_shape),
            ptr
        );
    });

    /////
    ///// Ops
    /////

    mod.method("op_add", [](
        const std::shared_ptr<ngraph::Node>& arg0,
        const std::shared_ptr<ngraph::Node>& arg1)
    {
        return makeop(v1::Add, arg0, arg1);
    });

    mod.method("op_avgpool", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const jlcxx::ArrayRef<int64_t> strides,
        const jlcxx::ArrayRef<int64_t> pads_begin,
        const jlcxx::ArrayRef<int64_t> pads_end,
        const jlcxx::ArrayRef<int64_t> kernel,
        bool exclude_pad)
    {
        return makeop(
            v1::AvgPool,
            arg,
            construct<ngraph::Strides>(strides),
            construct<ngraph::Shape>(pads_begin),
            construct<ngraph::Shape>(pads_end),
            construct<ngraph::Shape>(kernel),
            exclude_pad,
            ngraph::op::RoundingType::FLOOR,
            ngraph::op::PadType::EXPLICIT
        );
    });

    // mod.method("op_batchnorm_training", [](
    //     const std::shared_ptr<ngraph::Node> input,
    //     const std::shared_ptr<ngraph::Node> gamma,
    //     const std::shared_ptr<ngraph::Node> beta,
    //     double epsilon)
    // {
    //     auto a = std::make_shared<ngraph::op::BatchNormTraining>(input, gamma, beta, epsilon);
    //     return std::dynamic_pointer_cast<ngraph::Node>(a);
    // });

    mod.method("op_broadcast", [](
        const std::shared_ptr<ngraph::Node> &arg,
        const std::shared_ptr<ngraph::Node> &target_shape)
    {
        return makeop(v1::Broadcast, arg, target_shape);
    });

    mod.method("op_concat", [](
        const jlcxx::ArrayRef<std::shared_ptr<ngraph::Node>> jl_nodes,
        int64_t axis)
    {
        return makeop(
            v0::Concat,
            construct<ngraph::OutputVector>(jl_nodes),
            axis
        );
    });

    // Strategy for constants,
    // pass the julia array as an array of UInt8s - then we can use reinterpret-cast to
    // convert this to the type we want.
    mod.method("op_constant", [](
        const ngraph::element::Type& type,
        const jlcxx::ArrayRef<int64_t> jl_shape,
        const jlcxx::ArrayRef<uint8_t> jl_values)
    {
        ngraph::Shape shape = construct<ngraph::Shape>(jl_shape);
        ngraph::element::Type_t type_enum = ngraph::element::Type_t(type);

        // TODO: Finish up constant construction
        switch (type_enum)
        {
            case ngraph::element::Type_t::f32:
                return makeop(v0::Constant, type, shape, castcollect<float>(jl_values));
            case ngraph::element::Type_t::f64:
                return makeop(v0::Constant, type, shape, castcollect<double>(jl_values));
            case ngraph::element::Type_t::i64:
                return makeop(v0::Constant, type, shape, castcollect<int64_t>(jl_values));
            default:
                throw std::runtime_error("Unsupported type");
        }
    });

    mod.method("op_convert", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const ngraph::element::Type& element_type)
    {
        return makeop(v0::Convert, arg, element_type);
    });


    mod.method("op_convolution", [](
        const std::shared_ptr<ngraph::Node>& data_batch,
        const std::shared_ptr<ngraph::Node>& filters,
        const jlcxx::ArrayRef<int64_t> strides,
        const jlcxx::ArrayRef<int64_t> pads_begin,
        const jlcxx::ArrayRef<int64_t> pads_end,
        const jlcxx::ArrayRef<int64_t> dilations)
    {
        return makeop(
            v1::Convolution,
            data_batch,
            filters,
            construct<ngraph::Strides>(strides),
            construct<ngraph::CoordinateDiff>(pads_begin),
            construct<ngraph::CoordinateDiff>(pads_end),
            construct<ngraph::Strides>(dilations)
        );
    });

    mod.method("op_divide", [](
        const std::shared_ptr<ngraph::Node>& arg0,
        const std::shared_ptr<ngraph::Node>& arg1)
    {
        return makeop(v0::Divide, arg0, arg1);
    });

    mod.method("op_dot", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1,
        size_t reduction_axes_count)
    {
        return makeop(v0::Dot, arg0, arg1, reduction_axes_count);
    });

    // mod.method("op_goe", [](
    //     const std::shared_ptr<ngraph::Node>& arg,
    //     uint64_t n)
    // {
    //     return makeop(v1::GetOutputElement, arg, n);
    // });

    mod.method("op_log", [](const std::shared_ptr<ngraph::Node>& arg)
    {
        return makeop(v0::Log, arg);
    });

    mod.method("op_maximum", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1)
    {
        return makeop(v0::Maximum, arg0, arg1);
    });

    mod.method("op_maxpool", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const jlcxx::ArrayRef<int64_t> strides,
        const jlcxx::ArrayRef<int64_t> pads_begin,
        const jlcxx::ArrayRef<int64_t> pads_end,
        const jlcxx::ArrayRef<int64_t> kernel)
    {
        return makeop(
            v1::MaxPool,
            arg,
            construct<ngraph::Strides>(strides),
            construct<ngraph::Shape>(pads_begin),
            construct<ngraph::Shape>(pads_end),
            construct<ngraph::Shape>(kernel),
            ngraph::op::RoundingType::FLOOR,
            ngraph::op::PadType::EXPLICIT
        );
    });

    mod.method("op_minimum", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1)
    {
        return makeop(v0::Minimum, arg0, arg1);
    });


    mod.method("op_mul", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1)
    {
        return makeop(v0::Multiply, arg0, arg1);
    });

    mod.method("op_negative", [](const std::shared_ptr<ngraph::Node>& arg)
    {
        return makeop(v0::Negative, arg);
    });

    mod.method("op_parameter", [](
        const ngraph::element::Type &element_type,
        // NOTE: don't pass ArrayRef by reference ...
        const jlcxx::ArrayRef<int64_t,1> shape)
    {
        return makeop(
            Parameter,
            element_type,
            construct<ngraph::Shape>(shape)
        );
     });

    mod.method("op_power", [](
        const std::shared_ptr<ngraph::Node>& arg0,
        const std::shared_ptr<ngraph::Node>& arg1)
    {
        return makeop(v1::Power, arg0, arg1);
    });

    mod.method("op_relu", [](const std::shared_ptr<ngraph::Node> &arg){
        return makeop(v0::Relu, arg);
    });

    mod.method("op_reshape", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const jlcxx::ArrayRef<int64_t,1> input_order,
        const jlcxx::ArrayRef<int64_t,1> output_shape)
    {
        return makeop(
            v0::Reshape,
            arg,
            construct<ngraph::AxisVector>(input_order),
            construct<ngraph::Shape>(output_shape)
        );
    });

    mod.method("op_sigmoid", [](const std::shared_ptr<ngraph::Node>& arg)
    {
        return makeop(v0::Sigmoid, arg);
    });

    // mod.method("op_slice", [](
    //     const std::shared_ptr<ngraph::Node>& arg,
    //     const jlcxx::ArrayRef<int64_t,1> lower_bounds,
    //     const jlcxx::ArrayRef<int64_t,1> upper_bounds)
    // {
    //     return makeop(
    //         v0::Slice,
    //         arg,
    //         construct<ngraph::Coordinate>(lower_bounds),
    //         construct<ngraph::Coordinate>(upper_bounds)
    //     );
    // });

    mod.method("op_softmax", [](
        const std::shared_ptr<ngraph::Node>& arg,
        int64_t axis)
    {
        return makeop(v1::Softmax, arg, axis);
    });

    mod.method("op_sqrt", [](
        const std::shared_ptr<ngraph::Node>& arg)
    {
        return makeop(v0::Sqrt, arg);
    });

    mod.method("op_subtract", [](
        const std::shared_ptr<ngraph::Node>& arg0,
        const std::shared_ptr<ngraph::Node>& arg1)
    {
        return makeop(v1::Subtract, arg0, arg1);
    });

    mod.method("op_sum", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const jlcxx::ArrayRef<int64_t,1> reduction_axes)
    {
        return makeop(v0::Sum, arg, construct<ngraph::AxisSet>(reduction_axes));
    });
}

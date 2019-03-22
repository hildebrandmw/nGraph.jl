#include "jlcxx/jlcxx.hpp"
#include "jlcxx/tuple.hpp"

#include "ngraph/ngraph.hpp"
#include "ngraph/util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/descriptor/tensor.hpp"

#include "ngraph/type/element_type.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"

#include "ngraph/runtime/cpu/cpu_backend.hpp"

//#include "ngraph/descriptor/layout/tensor_layout.hpp"
//#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
//#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"

#include "ngraph/frontend/onnx_import/onnx.hpp"

#ifdef NGRAPH_PMDK_ENABLE
#include "ngraph/pmem.hpp"
#endif

/////
///// Struct for wrapping Node vectors
/////

struct NodeWrapper
{
    public:
        template <class Iter>
        NodeWrapper(Iter start, Iter stop);

        NodeWrapper(std::vector <std::shared_ptr <ngraph::Node> > nodes);

        std::shared_ptr<ngraph::Node> _getindex(int64_t i);
        int64_t _length();

    private:
        std::vector< std::shared_ptr <ngraph::Node> > m_nodes;
};

// Implementation
template <class Iter>
NodeWrapper::NodeWrapper(Iter start, Iter stop)
{
    m_nodes = std::vector< std::shared_ptr<ngraph::Node> >(start, stop);
}

NodeWrapper::NodeWrapper(std::vector<std::shared_ptr<ngraph::Node>> nodes)
{
    m_nodes = nodes;
}

std::shared_ptr<ngraph::Node> NodeWrapper::_getindex(int64_t i)
{
    return m_nodes[i];
}

int64_t NodeWrapper::_length()
{
    return m_nodes.size();
}

/////
///// Module Wrapping
/////


JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    /////
    ///// nGraph Types
    /////

    ///// AxisSet
    mod.add_type<ngraph::AxisSet>("AxisSet");
    mod.method("AxisSet", [](jlcxx::ArrayRef<int64_t, 1> arr){
        return ngraph::AxisSet(std::set<size_t>(arr.begin(), arr.end()));
    });

    ///// AxisVector
    mod.add_type<ngraph::AxisVector>("AxisVector");
    mod.method("AxisVector", [](jlcxx::ArrayRef<int64_t, 1> arr){
        return ngraph::AxisVector(arr.begin(), arr.end());
    });

    ///// CoordinateDiff
    mod.add_type<ngraph::CoordinateDiff>("CoordinateDiff");
    mod.method("CoordinateDiff", [](const jlcxx::ArrayRef<int64_t, 1> vals){
        return ngraph::CoordinateDiff(vals.begin(), vals.end());
    });

    ///// Elements
    mod.add_type<ngraph::element::Type>("NGraphType")
        .method("c_type_name", &ngraph::element::Type::c_type_string);

    mod.method("gettype", [](const std::string& type){
        if (type == "boolean") {return &ngraph::element::boolean;}
        else if (type == "bf16") {return &ngraph::element::bf16;}
        else if (type == "f32") {return &ngraph::element::f32;}
        else if (type == "f64") {return &ngraph::element::f64;}
        else if (type == "i8" ) {return &ngraph::element::i8; }
        else if (type == "i16") {return &ngraph::element::i16;}
        else if (type == "i32") {return &ngraph::element::i32;}
        else if (type == "i64") {return &ngraph::element::i64;}
        else if (type == "u8" ) {return &ngraph::element::u8; }
        else if (type == "u16") {return &ngraph::element::u16;}
        else if (type == "u32") {return &ngraph::element::u32;}
        else if (type == "u64") {return &ngraph::element::u64;}
        else{return &ngraph::element::dynamic;}
    });


    ///// Shapes
    mod.add_type<ngraph::Shape>("Shape");
    mod.method("Shape", [](const jlcxx::ArrayRef<int64_t, 1> vals){
        return ngraph::Shape(vals.begin(), vals.end());
    });
    // Methods to facilitate instantiating a shape in Julia
    mod.method("_length", [](const ngraph::Shape s){return (int64_t) s.size();});
    mod.method("_getindex", [](const ngraph::Shape s, int64_t i){return (int64_t) s[i];});

    ////// Strides
    mod.add_type<ngraph::Strides>("Strides");
    mod.method("Strides", [](const jlcxx::ArrayRef<int64_t, 1> vals){
        return ngraph::Strides(vals.begin(), vals.end());
    });

    ///// descriptor::Tensor
    mod.add_type<ngraph::descriptor::Tensor>("DescriptorTensor")
        .method("_sizeof", &ngraph::descriptor::Tensor::size)
        .method("get_name", &ngraph::descriptor::Tensor::get_name);

    ///// runtime::Tensor
    mod.add_type<ngraph::runtime::Tensor>("RuntimeTensor")
        .method("get_shape", &ngraph::runtime::Tensor::get_shape);


    // Read/write wrappers for tensorw
    mod.method("tensor_write", [](
        std::shared_ptr<ngraph::runtime::Tensor> tensor,
        void* p,
        size_t offset,
        size_t n)
    {
        tensor->write(p, offset, n);
    });

    mod.method("tensor_read", [](
        std::shared_ptr<ngraph::runtime::Tensor> tensor,
        void* p,
        size_t offset,
        size_t n)
    {
        tensor->read(p, offset, n);
    });

    ///// Node
    mod.add_type<ngraph::Node>("Node")
        .method("get_name", &ngraph::Node::get_name)
        .method("description", &ngraph::Node::description)
        ///// Outputs
        // Return the number of outputs for the op
        .method("get_output_size", &ngraph::Node::get_output_size)
        // Return the element type for output i
        .method("get_output_element_type", &ngraph::Node::get_output_element_type)
        // Return the shape of output i
        .method("get_output_shape", &ngraph::Node::get_output_shape)
        ///// Inputs
        // Return the number of inputs for the op
        .method("get_input_size", &ngraph::Node::get_input_size)
        // Return the element type of input i
        .method("get_input_element_type", &ngraph::Node::get_input_element_type)
        // Return the shape of input i
        .method("get_input_shape", &ngraph::Node::get_input_shape)
        .method("get_input_node", [](
                const std::shared_ptr<ngraph::Node>& node,
                const int64_t index)
            {
                // Node
                // -> Deque of input descriptors
                // -> Input Descriptor
                // -> Output connected to Input
                // -> Node
                ngraph::descriptor::Output& output = node->get_inputs().at(index).get_output();
                auto output_node = output.get_node();

                return output_node;
            })

         .method("get_output_nodes", [](
                 const std::shared_ptr<ngraph::Node>& node,
                 const int64_t index)
             {
                 std::set<ngraph::descriptor::Input*> output_inputs = node->get_output_inputs(index);
                 std::vector<std::shared_ptr<ngraph::Node>> nodes;
                 for (auto input : output_inputs)
                 {
                    nodes.push_back(input->get_node());
                 }
                 return NodeWrapper(nodes);
             });

    // Give me a node, I'll give yah a tensor!
    mod.method("get_output_tensor_ptr", [](
            const std::shared_ptr<ngraph::Node> node,
            int64_t index)
        {
            return node->get_output_tensor_ptr(index);
        });

    mod.method("get_input_tensor_ptr", [](
            const std::shared_ptr<ngraph::Node> node,
            int64_t index)
        {
            return node->get_inputs().at(index).get_output().get_tensor_ptr();
        });

    // Check if the given output is only connected to a "Result" node.
    mod.method("output_is_result", [](
            const std::shared_ptr<ngraph::Node> node,
            int64_t index)
        {
            // Get the set of nodes that use this output
            const std::set<ngraph::descriptor::Input*>& users = node->get_output_inputs(index);

            // Predicate to check if a node is a Result
            auto predicate = [](ngraph::descriptor::Input* input)
            {
                return input->get_node()->description() == "Result";
            };
            return std::all_of(users.begin(), users.end(), predicate);
        });

    mod.add_type<ngraph::NodeVector>("NodeVector")
        .method("push!", [](ngraph::NodeVector& nodes, std::shared_ptr<ngraph::Node> node)
        {
            nodes.push_back(node);
        });

    mod.add_type<ngraph::ParameterVector>("ParameterVector")
        .method("push!", [](ngraph::ParameterVector& params, std::shared_ptr<ngraph::Node> parameter)
        {
            auto a = std::dynamic_pointer_cast<ngraph::op::Parameter>(parameter);
            params.push_back(a);
        });

    /////
    ///// NodeWrapper
    /////

    mod.add_type<NodeWrapper>("NodeWrapper")
        .method("_length", &NodeWrapper::_length)
        .method("_getindex", &NodeWrapper::_getindex);

    /////
    ///// Function
    /////
    mod.add_type<ngraph::Function>("NFunction")
        .method("get_name", &ngraph::Function::get_name);

    mod.method("make_function", [](
            const ngraph::NodeVector& nodes,
            const ngraph::ParameterVector& parameters)
    {
        return std::make_shared<ngraph::Function>(nodes, parameters);
    });

    mod.method("get_ordered_ops", [](const std::shared_ptr<ngraph::Function> func)
    {
        std::list<std::shared_ptr<ngraph::Node>> node_list = func->get_ordered_ops();
        return NodeWrapper(node_list.begin(), node_list.end());
    });

    /////
    ///// Adjoint
    /////

    mod.add_type<ngraph::autodiff::Adjoints>("Adjoints")
        .constructor<const ngraph::NodeVector&, const ngraph::NodeVector&>()
        .method("backprop_node", &ngraph::autodiff::Adjoints::backprop_node);

    /////
    ///// Ops
    /////
    mod.method("op_add", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1)
    {
        auto a = std::make_shared<ngraph::op::Add>(arg0, arg1);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_avgpool", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const ngraph::Shape& window_shape,
        const ngraph::Strides& window_movement_strides,
        const ngraph::Shape& padding_below,
        const ngraph::Shape& padding_above)
    {
        auto a = std::make_shared<ngraph::op::AvgPool>(
            arg, window_shape, window_movement_strides, padding_below, padding_above, false);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_broadcast", [](
        const std::shared_ptr<ngraph::Node> &arg,
        const ngraph::Shape& shape,
        const ngraph::AxisSet& broadcast_axes)
    {
        auto a = std::make_shared<ngraph::op::Broadcast>(arg, shape, broadcast_axes);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_concat", [](
        const ngraph::NodeVector& args,
        int64_t concatenation_axis)
    {
        auto a = std::make_shared<ngraph::op::Concat>(args, concatenation_axis);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_constant", [](
        const ngraph::element::Type& type,
        ngraph::Shape shape,
        const jlcxx::ArrayRef<float, 1>& jl_values)
    {
        std::vector<float> values = std::vector<float>(jl_values.begin(), jl_values.end());
        auto a = std::make_shared<ngraph::op::Constant>(type, shape, values);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_convolution", [](
        const std::shared_ptr<ngraph::Node>& data_batch,
        const std::shared_ptr<ngraph::Node>& filters,
        const ngraph::Strides& window_movement_strides,
        const ngraph::Strides& window_dilation_strides,
        const ngraph::CoordinateDiff& padding_below,
        const ngraph::CoordinateDiff& padding_above)
    {
        auto a = std::make_shared<ngraph::op::Convolution>(
            data_batch,
            filters,
            window_movement_strides,
            window_dilation_strides,
            padding_below,
            padding_above);

        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_divide", [](
        const std::shared_ptr<ngraph::Node>& arg0,
        const std::shared_ptr<ngraph::Node>& arg1)
    {
        auto a = std::make_shared<ngraph::op::Divide>(arg0, arg1);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_dot", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1,
        size_t reduction_axes_count)
    {
        auto a = std::make_shared<ngraph::op::Dot>(arg0, arg1, reduction_axes_count);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_log", [](const std::shared_ptr<ngraph::Node>& arg)
    {
        auto a = std::make_shared<ngraph::op::Log>(arg);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_maxpool", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const ngraph::Shape& window_shape,
        const ngraph::Strides& window_movement_strides,
        const ngraph::Shape& padding_below,
        const ngraph::Shape& padding_above)
    {
        auto a = std::make_shared<ngraph::op::MaxPool>(
                arg, window_shape, window_movement_strides, padding_below, padding_above);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_minimum", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1)
    {
        auto a = std::make_shared<ngraph::op::Minimum>(arg0, arg1);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });


    mod.method("op_mul", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1)
    {
        auto a = std::make_shared<ngraph::op::Multiply>(arg0, arg1);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_negative", [](const std::shared_ptr<ngraph::Node>& arg)
    {
        auto a = std::make_shared<ngraph::op::Negative>(arg);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_parameter", [](
        const ngraph::element::Type &element_type,
        const ngraph::Shape &shape)
    {
        // Do the casting shape -> partial shape
        auto a =  std::make_shared<ngraph::op::Parameter>(
            element_type,
            ngraph::PartialShape(shape)
        );
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_power", [](
        const std::shared_ptr<ngraph::Node>& arg0,
        const std::shared_ptr<ngraph::Node>& arg1)
    {
        auto a = std::make_shared<ngraph::op::Power>(arg0, arg1);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_relu", [](const std::shared_ptr<ngraph::Node> &arg){
        auto a = std::make_shared<ngraph::op::Relu>(arg);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_reshape", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const ngraph::AxisVector& input_order,
        const ngraph::Shape& output_shape)
    {
        auto a = std::make_shared<ngraph::op::Reshape>(arg, input_order, output_shape);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_softmax", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const ngraph::AxisSet& axes)
    {
        auto a = std::make_shared<ngraph::op::Softmax>(arg, axes);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_subtract", [](
        const std::shared_ptr<ngraph::Node>& arg0,
        const std::shared_ptr<ngraph::Node>& arg1)
    {
        auto a = std::make_shared<ngraph::op::Subtract>(arg0, arg1);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_sum", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const ngraph::AxisSet& reduction_axes)
    {
        auto a = std::make_shared<ngraph::op::Sum>(arg, reduction_axes);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    /////
    ///// Executable
    /////
    mod.add_type<ngraph::runtime::Executable>("Executable")
        // Due to the limitations of CxxWrap, we have to play some schenenigans with
        // jl_value_t* to pass wrapped types into a function.
        //
        // See: https://github.com/JuliaInterop/CxxWrap.jl/issues/112
        .method("call", [](const std::shared_ptr<ngraph::runtime::Executable> executable,
                    const jlcxx::ArrayRef<jl_value_t*, 1>& jl_outputs,
                    const jlcxx::ArrayRef<jl_value_t*, 1>& jl_inputs)
            {
                std::vector<std::shared_ptr<ngraph::runtime::Tensor>> inputs = {};
                std::vector<std::shared_ptr<ngraph::runtime::Tensor>> outputs = {};
                // TODO: validate this
                for (auto i: jl_outputs)
                    outputs.push_back(
                        *jlcxx::unbox_wrapped_ptr< std::shared_ptr<ngraph::runtime::Tensor> >(i)
                    );

                for (auto i: jl_inputs)
                    inputs.push_back(
                        *jlcxx::unbox_wrapped_ptr< std::shared_ptr<ngraph::runtime::Tensor> >(i)
                    );

                executable->call(outputs, inputs);
            }
        );

    // Backend
    mod.add_type<ngraph::runtime::Backend>("Backend")
        .method("create", &ngraph::runtime::Backend::create)
        .method("compile", &ngraph::runtime::Backend::compile)
        .method("remove_compiled_function", &ngraph::runtime::Backend::remove_compiled_function);

    mod.method("create_tensor", [](
        ngraph::runtime::Backend* backend,
        const ngraph::element::Type& element_type,
        const ngraph::Shape& shape)
    {
        return backend->create_tensor(element_type, shape);
    });


    // PMDK Stuff
#ifdef NGRAPH_PMDK_ENABLE

    // Query if a tensor is in persistent memory
    mod.method("is_persistent", [](const std::shared_ptr<ngraph::descriptor::Tensor> tensor)
    {
        return tensor->get_pool_number() == 1;
    });

    // Mark that a tensor should be placed in persistent memory
    mod.method("make_persistent", [](const std::shared_ptr<ngraph::descriptor::Tensor> tensor)
    {
        tensor->set_pool_number(1);
    });

    // Mark that a tensor should not be placed in persistent memory
    mod.method("make_volatile", [](const std::shared_ptr<ngraph::descriptor::Tensor> tensor)
    {
        tensor->set_pool_number(0);
    });

    mod.method("create_persistent_tensor", [](
        ngraph::runtime::Backend* backend,
        const ngraph::element::Type& element_type,
        const ngraph::Shape& shape)
    {
        auto cpu_backend = static_cast<ngraph::runtime::cpu::CPU_Backend*>(backend);
        return cpu_backend->create_persistent_tensor(element_type, shape);
    });

    // PMDK stuff
    mod.add_type<ngraph::pmem::PMEMManager>("PMEMManager")
        .method("getinstance", &ngraph::pmem::PMEMManager::getinstance)
        .method("create_pool", &ngraph::pmem::PMEMManager::create_pool);

#endif

}


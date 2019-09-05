#include "jlcxx/jlcxx.hpp"
#include "jlcxx/tuple.hpp"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/move.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"

#include "ngraph/type/element_type.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"

#include "ngraph/runtime/allocator.hpp"

// CPU Related Stuff
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/cpu_helper.hpp"

// CPU ops
#include "ngraph/runtime/cpu/op/batch_dot.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"

// GPU Related Stuff
#ifdef NGRAPH_GPU_ENABLE
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/runtime/gpu/gpu_helper.hpp"
#include "ngraph/runtime/gpu/op/sync.hpp"
#endif

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
///// Class for extracting performance data from GPU Backend
/////

class PerfCounterTranslator
{
    public:
        PerfCounterTranslator(std::shared_ptr<ngraph::runtime::Executable>& exe);
        int64_t _length();
        std::tuple<std::string, size_t>  _getindex(int64_t i);

    private:
        std::vector<ngraph::runtime::PerformanceCounter> m_counters;
};

// Implementation
PerfCounterTranslator::PerfCounterTranslator(std::shared_ptr<ngraph::runtime::Executable>& exe)
{
    m_counters = exe->get_performance_data();
}

int64_t PerfCounterTranslator::_length()
{
    return m_counters.size();
}

std::tuple<std::string, size_t> PerfCounterTranslator::_getindex(int64_t i)
{
    ngraph::runtime::PerformanceCounter counter = m_counters.at(i);
    return make_tuple(counter.get_node()->get_name(), counter.microseconds());
}

/////
///// Module Wrapping
/////

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    /////
    ///// NodeWrapper
    /////

    mod.add_type<PerfCounterTranslator>("PerfCounterTranslator")
        .constructor<std::shared_ptr<ngraph::runtime::Executable>&>()
        .method("_length", &PerfCounterTranslator::_length)
        .method("_getindex", &PerfCounterTranslator::_getindex);

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

    ///// Coordinate
    mod.add_type<ngraph::Coordinate>("Coordinate");
    mod.method("Coordinate", [](const jlcxx::ArrayRef<int64_t, 1> vals){
        return ngraph::Coordinate(vals.begin(), vals.end());
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
        .method("get_name", &ngraph::descriptor::Tensor::get_name)
        .method("set_pool_offset", &ngraph::descriptor::Tensor::set_pool_offset)
        .method("get_pool_offset", &ngraph::descriptor::Tensor::get_pool_offset);

    ///// runtime::Tensor
    mod.add_type<ngraph::runtime::Tensor>("RuntimeTensor")
        .method("get_shape", &ngraph::runtime::Tensor::get_shape)
        .method("get_size_in_bytes", &ngraph::runtime::Tensor::get_size_in_bytes)
        .method("get_element_type", &ngraph::runtime::Tensor::get_element_type)
        .method("get_name", &ngraph::runtime::Tensor::get_name);

    // Read/write wrappers for tensor
    mod.method("tensor_write", [](
        std::shared_ptr<ngraph::runtime::Tensor> tensor,
        void* p,
        size_t n)
    {
        tensor->write(p, n);
    });

    mod.method("tensor_read", [](
        std::shared_ptr<ngraph::runtime::Tensor> tensor,
        void* p,
        size_t n)
    {
        tensor->read(p, n);
    });

    ///// Node
    mod.add_type<ngraph::Node>("Node")
        .method("get_name", &ngraph::Node::get_name)
        .method("description", &ngraph::Node::description)
        .method("add_control_dependency", &ngraph::Node::add_control_dependency)
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
                 //std::set<ngraph::descriptor::Input*> output_inputs = node->get_output_inputs(index);
                 auto output_inputs = node->output(index).get_target_inputs();
                 std::vector<std::shared_ptr<ngraph::Node>> nodes;
                 for (auto input : output_inputs)
                 {
                    // Input<Node> just store the raw pointers.
                    //
                    // Find the canonical shared_ptr from the node and use that.
                    nodes.push_back(input.get_node()->shared_from_this());
                 }
                 return NodeWrapper(nodes);
             })
         .method("get_control_deps", [](const std::shared_ptr<ngraph::Node>& node)
             {
                 auto deps = node->get_control_dependencies();
                 std::vector<std::shared_ptr<ngraph::Node>> nodes;
                 for (auto node : deps)
                 {
                    nodes.push_back(node);
                 }
                 return NodeWrapper(nodes);
             });

    mod.method("copy_with_new_args", [](
                const std::shared_ptr<ngraph::Node>& node,
                const ngraph::NodeVector& args)
    {
        // Create an OutputVector from the arguments 
        ngraph::OutputVector ov = ngraph::OutputVector(args.size());    
        auto op = [](std::shared_ptr<ngraph::Node> node){ return node->output(0); }; 
        std::transform(args.begin(), args.end(), ov.begin(), op);
        return node->copy_with_new_inputs(ov);
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
            //const std::set<ngraph::descriptor::Input*>& users = node->get_output_inputs(index);
            auto users = node->output(index).get_target_inputs();

            // Predicate to check if a node is a Result
            auto predicate = [](ngraph::Input<ngraph::Node> input)
            {
                return input.get_node()->description() == "Result";
            };
            return std::all_of(users.begin(), users.end(), predicate);
        });

    mod.add_type<ngraph::NodeVector>("NodeVector")
        .method("push!", [](ngraph::NodeVector& nodes, std::shared_ptr<ngraph::Node>& node)
        {
            nodes.push_back(node);
        });

    mod.add_type<ngraph::ParameterVector>("ParameterVector")
        .method("push!", [](ngraph::ParameterVector& params, std::shared_ptr<ngraph::Node> parameter)
        {
            auto a = std::dynamic_pointer_cast<ngraph::op::Parameter>(parameter);
            params.push_back(a);
        })
        .method("_length", [](ngraph::ParameterVector& params)
        {
            return params.size();
        })
        .method("_getindex", [](ngraph::ParameterVector& params, int64_t i)
        {
            return std::dynamic_pointer_cast<ngraph::Node>(params[i]);
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
        .method("get_name", &ngraph::Function::get_name)
        .method("get_parameters", &ngraph::Function::get_parameters)
        .method("get_temporary_pool_size", &ngraph::Function::get_temporary_pool_size)
        .method("get_remote_pool_size", &ngraph::Function::get_remote_pool_size)
        .method("set_jl_callback", &ngraph::Function::set_jl_callback)
        .method("clear_jl_callback", &ngraph::Function::clear_jl_callback)
        .method("get_jl_callback", &ngraph::Function::get_jl_callback)
        .method("has_jl_callback", &ngraph::Function::has_jl_callback);


    mod.method("get_results", [](const std::shared_ptr<ngraph::Function> fn)
    {
        ngraph::ResultVector results = fn->get_results();
        return NodeWrapper(results.begin(), results.end());
    });


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

    mod.method("clone_function", [](const std::shared_ptr<ngraph::Function> func)
    {
        return ngraph::clone_function(*func.get());
    });

    /////
    ///// Adjoint
    /////

    mod.add_type<ngraph::autodiff::Adjoints>("Adjoints");
    mod.method("make_adjoints", [](const ngraph::NodeVector& y, const ngraph::NodeVector& c)
    {
        ngraph::OutputVector oy = ngraph::OutputVector(y.size());
        ngraph::OutputVector oc = ngraph::OutputVector(c.size());

        auto op = [](std::shared_ptr<ngraph::Node> node){ 
            return node->output(0);
        };

        std::transform(y.begin(), y.end(), oy.begin(), op);
        std::transform(c.begin(), c.end(), oc.begin(), op);

        return ngraph::autodiff::Adjoints(oy, oc);
    });

    mod.method("backprop_node", [](
        ngraph::autodiff::Adjoints& adjoints,
        const std::shared_ptr<ngraph::Node>& x)
    {
        return adjoints.backprop_node(x);
    });

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

    mod.method("op_batchdot", [](
        const std::shared_ptr<ngraph::Node>& x,
        const std::shared_ptr<ngraph::Node>& y,
        bool transpose_x,
        bool transpose_y)
    {
        auto a = std::make_shared<ngraph::op::BatchDot>(x, y, transpose_x, transpose_y);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_batchnorm_training", [](
        const std::shared_ptr<ngraph::Node> input,
        const std::shared_ptr<ngraph::Node> gamma,
        const std::shared_ptr<ngraph::Node> beta,
        double epsilon)
    {
        auto a = std::make_shared<ngraph::op::BatchNormTraining>(input, gamma, beta, epsilon);
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

    mod.method("op_convolution_backprop_data", [](
        const ngraph::Shape& data_batch_shape,
        const std::shared_ptr<ngraph::Node>& filters,
        const std::shared_ptr<ngraph::Node>& output_delta,
        const ngraph::Strides& window_movement_strides_forward,
        const ngraph::Strides& window_dilation_strides_forward,
        const ngraph::CoordinateDiff& padding_below_forward,
        const ngraph::CoordinateDiff& padding_above_forward,
        const ngraph::Strides& data_dilation_strides_forward)
    {
        auto a = std::make_shared<ngraph::op::ConvolutionBackpropData>(
            data_batch_shape,
            filters,
            output_delta,
            window_movement_strides_forward,
            window_dilation_strides_forward,
            padding_below_forward,
            padding_above_forward,
            data_dilation_strides_forward);
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

    mod.method("op_embedding", [](
        const std::shared_ptr<ngraph::Node>& data,
        const std::shared_ptr<ngraph::Node>& weights)
    {
        auto a = std::make_shared<ngraph::op::EmbeddingLookup>(data, weights);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_get_output_element", [](
        const std::shared_ptr<ngraph::Node>& arg,
        size_t n)
    {
        auto a = std::make_shared<ngraph::op::GetOutputElement>(arg, n);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_log", [](const std::shared_ptr<ngraph::Node>& arg)
    {
        auto a = std::make_shared<ngraph::op::Log>(arg);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_lstm", [](
            const std::shared_ptr<ngraph::Node> src_layer,
            const std::shared_ptr<ngraph::Node> src_iter,
            const std::shared_ptr<ngraph::Node> weights_layer,
            const std::shared_ptr<ngraph::Node> weights_iter,
            std::shared_ptr<ngraph::Node> bias)
    {
        auto a = std::make_shared<ngraph::op::Lstm>(
                src_layer, 
                src_iter, 
                weights_layer, 
                weights_iter, 
                bias, 
                ngraph::runtime::cpu::rnn_utils::vanilla_lstm
            );
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    // Some notes regarding direction
    //
    // nGraph's documentation regarding direction is a little spotty (i.e. it doesn't exist.)
    // I'm pretty sure 1 means single directional while 2 is bidirectional.
    //
    // Would be nice if they had an actual API for this.
    mod.method("op_lstm_rnn", [](
            const std::shared_ptr<ngraph::Node> src_layer,
            const std::shared_ptr<ngraph::Node> src_iter,
            const std::shared_ptr<ngraph::Node> weights_layer,
            const std::shared_ptr<ngraph::Node> weights_iter,
            std::shared_ptr<ngraph::Node> bias,
            size_t num_timesteps,
            size_t direction,
            size_t num_fused_layers)
    {
        auto a = std::make_shared<ngraph::op::Rnn>(
                src_layer,
                src_iter,
                weights_layer,
                weights_iter,
                bias,
                num_timesteps,
                4, // 4 gates for LSTM cell
                num_timesteps,
                2, // 2 cell stats for LSTM
                direction,
                num_fused_layers,
                ngraph::runtime::cpu::rnn_utils::vanilla_lstm);

        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_maximum", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1)
    {
        auto a = std::make_shared<ngraph::op::Maximum>(arg0, arg1);
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

    mod.method("op_result", [](const std::shared_ptr<ngraph::Node>& arg)
    {
        auto a = std::make_shared<ngraph::op::Result>(arg);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_sigmoid", [](const std::shared_ptr<ngraph::Node>& arg)
    {
        auto a = std::make_shared<ngraph::op::Sigmoid>(arg);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_slice", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const ngraph::Coordinate& lower_bounds,
        const ngraph::Coordinate& upper_bounds)
    {
        auto a = std::make_shared<ngraph::op::Slice>(arg, lower_bounds, upper_bounds);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_softmax", [](
        const std::shared_ptr<ngraph::Node>& arg,
        const ngraph::AxisSet& axes)
    {
        auto a = std::make_shared<ngraph::op::Softmax>(arg, axes);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_sqrt", [](
        const std::shared_ptr<ngraph::Node>& arg)
    {
        auto a = std::make_shared<ngraph::op::Sqrt>(arg);
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
        auto a = std::make_shared<ngraph::op::Sum>(arg->output(0), reduction_axes);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_tanh", [](const std::shared_ptr<ngraph::Node>& arg)
    {
        auto a = std::make_shared<ngraph::op::Tanh>(arg);
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
        .method("remove_compiled_function", &ngraph::runtime::Backend::remove_compiled_function);
        //.method("set_host_memory_allocator", &ngraph::runtime::Backend::set_host_memory_allocator);

    mod.method("create", [](const std::string& type)
        {
            return ngraph::runtime::Backend::create(type, false);
        });

    mod.method("compile", [](std::shared_ptr<ngraph::runtime::Backend> backend,
                             std::shared_ptr<ngraph::Function> func,
                             bool enable_performance_data)
        {
            return backend->compile(func, enable_performance_data);
        });

    mod.method("create_tensor", [](
        ngraph::runtime::Backend* backend,
        const ngraph::element::Type& element_type,
        const ngraph::Shape& shape)
    {
        return backend->create_tensor(element_type, shape);
    });

    /////
    ///// CPU Ops
    /////

    mod.method("op_cpu_convert_layout_to", [](
        const std::shared_ptr<ngraph::Node> &arg,
        const std::shared_ptr<ngraph::Node> &target,
        int64_t input_index)
    {
        // Get the LayoutDescriptor for `input_index` of `target`.
        auto a = std::make_shared<ngraph::runtime::cpu::op::ConvertLayout>(
            arg,
            target,
            input_index
        );
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    // Need to wrap this in a lambda instead of just directly referencing the method
    // because in the direct reference case, Julia cannot resolve the symbol
    mod.method("input_needs_conversion", [](
        const std::shared_ptr<ngraph::Node> &node,
        size_t input_index)
    {
        return ngraph::runtime::cpu::input_needs_conversion(node, input_index);
    });


    // The following two methods were taken from mkldnn_utils.cpp in the runtime/cpu
    mod.method("node_is_mkldnn_op", [](const std::shared_ptr<ngraph::Node>& node)
    {
        // Convert to an "Op"
        auto op_node = std::dynamic_pointer_cast<ngraph::op::Op>(node);
        if (!op_node)
        {
            return false;
        }

        // Can be null
        auto op_annotations = op_node->get_op_annotations();
        if (!op_annotations)
        {
            return false;
        }

        // Might not be a CPU op
        auto cpu_annotations = std::dynamic_pointer_cast<ngraph::runtime::cpu::CPUOpAnnotations>(op_annotations);
        if (!cpu_annotations)
        {
            return false;
        }

        return cpu_annotations->is_mkldnn_op();
    });

    // Assumes this is actually a mkldnn_op
    mod.method("node_set_mkldnn_op", [](const std::shared_ptr<ngraph::Node>& node)
    {
        auto ngraph_op = std::static_pointer_cast<ngraph::op::Op>(node);
        auto op_annotations = std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
        op_annotations->set_mkldnn_op(true);
        ngraph_op->set_op_annotations(op_annotations);
    });

    mod.method("get_input_format_int", [](
        const std::shared_ptr<ngraph::Node>& node,
        size_t index)
    {
        return ngraph::runtime::cpu::get_input_format_int(node, index);
    });

    mod.method("get_mkldnn_string", [](int64_t enum_int)
    {
        return ngraph::runtime::cpu::get_mkldnn_string(enum_int);
    });

    // Graph Utils
    mod.method("special_insert_new_node_between", [](
        const std::shared_ptr<ngraph::Node>& src_node,
        size_t src_output_index,
        const std::shared_ptr<ngraph::Node>& dst_node,
        size_t dst_input_index,
        const std::shared_ptr<ngraph::Node>& new_node) 
    {
        return ngraph::special_insert_new_node_between(
            src_node,
            src_output_index,
            dst_node,
            dst_input_index,
            new_node);
    });

    mod.method("op_move", [](const std::shared_ptr<ngraph::Node> &arg, size_t n){
        auto a = std::make_shared<ngraph::op::Move>(arg, n);
        return std::dynamic_pointer_cast<ngraph::Node>(a);
    });

    mod.method("op_moveasync", [](
        const std::shared_ptr<ngraph::Node> &arg,
        size_t n,
        const std::shared_ptr<ngraph::Node> &across)
    {
        auto a = std::make_shared<ngraph::op::MoveAsync>(arg, n, across);
        auto b = std::dynamic_pointer_cast<ngraph::Node>(a);

        // Across now has a control dependency on the async move node, which means that
        // the async move node MUST be scheduled before `across`
        //across->add_control_dependency(b);
        b->add_control_dependency(across);
        return b;
    });

    mod.method("set_priority", [](const std::shared_ptr<ngraph::Node>& node, int64_t p)
    {
        node->set_priority(p);
    });

    mod.method("get_priority", [](const std::shared_ptr<ngraph::Node>& node)
    {
        node->get_priority();
    });

    /////
    ///// GPU Ops
    /////

#ifdef NGRAPH_GPU_ENABLE
    mod.method("can_select_algo", [](const std::shared_ptr<ngraph::Node>& node)
    {
        return ngraph::runtime::gpu::can_select_algo(node);
    });
#else
// Dummy fallback because this method can still be called.
    mod.method("can_select_algo", [](const std::shared_ptr<ngraph::Node>& node)
    {
        return false;
    });
#endif

#ifdef NGRAPH_GPU_ENABLE
    mod.method("get_algo_options", [](
        const std::shared_ptr<ngraph::Node>& node,
        jlcxx::ArrayRef<uint32_t> algo_numbers,
        jlcxx::ArrayRef<float> timings,
        jlcxx::ArrayRef<size_t> memories)
    {
        // Constructing a julia array directly did not seem to work super well ...
        //
        // The simplest way would bw to just construct a vector of tuples, but it gets
        // horribly mangled when transferring over into Julia, and I don't really feel
        // like debugging that.
        auto options = ngraph::runtime::gpu::get_algo_options(node);
        bool alloc_failed = false;
        for (auto i: options)
        {
            // Unpack tuple
            uint32_t algo;
            float time;
            size_t memory;
            bool failed;
            std::tie(algo, time, memory, failed) = i;

            // Only add the non-failed convolutions.
            if (failed)
            {
                alloc_failed = true;
            } else {
                algo_numbers.push_back(algo);
                timings.push_back(time);
                memories.push_back(memory);
            }
        }
        return alloc_failed;
    });

    mod.method("set_algo", [](
        const std::shared_ptr<ngraph::Node>& node,
        size_t algo,
        size_t workspace_size)
    {
        ngraph::runtime::gpu::set_algo(node, algo, workspace_size); 
    });
#endif

    /////
    ///// These now have realy bad names now that we're switching over to doing CPU and GPU
    /////

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

    mod.method("create_cpu_persistent_tensor", [](
        ngraph::runtime::Backend* backend,
        const ngraph::element::Type& element_type,
        const ngraph::Shape& shape)
    {
        auto cpu_backend = static_cast<ngraph::runtime::cpu::CPU_Backend*>(backend);
        return cpu_backend->create_persistent_tensor(element_type, shape);
    });

#ifdef NGRAPH_GPU_ENABLE
    mod.method("create_gpu_persistent_tensor", [](
        ngraph::runtime::Backend* backend,
        const ngraph::element::Type& element_type,
        const ngraph::Shape& shape)
    {
        auto gpu_backend = static_cast<ngraph::runtime::gpu::GPU_Backend*>(backend);
        return gpu_backend->create_remote_tensor(element_type, shape);
    });
#endif

    // PMDK stuff
#ifdef NGRAPH_PMDK_ENABLE
    mod.add_type<ngraph::pmem::PMEMManager>("PMEMManager")
        .method("getinstance", &ngraph::pmem::PMEMManager::getinstance)
        .method("set_pool_dir", &ngraph::pmem::PMEMManager::set_pool_dir);

    mod.method("set_pmm_allocator", [](const std::shared_ptr<ngraph::runtime::Backend> backend)
    {
        std::cout << "Setting PMM Allocator" << std::endl;
        backend->set_host_memory_allocator(ngraph::runtime::get_pmm_allocator());
    });
#endif
}


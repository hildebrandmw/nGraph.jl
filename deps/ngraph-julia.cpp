#include "jlcxx/jlcxx.hpp"

#include "ngraph/ngraph.hpp"
#include "ngraph/util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"

#include "ngraph/runtime/cpu/cpu_backend.hpp"

#include "ngraph/frontend/onnx_import/onnx.hpp"

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    // Types
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

    /////
    ///// Shapes
    /////
    mod.add_type<ngraph::Shape>("Shape");
    mod.method("make_shape", [](const jlcxx::ArrayRef<int64_t, 1> vals){
        return ngraph::Shape(vals.begin(), vals.end());
    });
    // Methods to facilitate instantiating a shape in Julia
    mod.method("shape_length", [](const ngraph::Shape s){return (int64_t) s.size();});
    mod.method("shape_getindex", [](const ngraph::Shape s, int64_t i){return (int64_t) s[i];});

    /////
    ///// AxisSet
    /////
    mod.add_type<ngraph::AxisSet>("AxisSet"); 
    mod.method("make_axisset", [](jlcxx::ArrayRef<int64_t, 1> arr){
        return ngraph::AxisSet(std::set<size_t>(arr.begin(), arr.end()));
    });

    /////
    ///// Tensor
    /////
    mod.add_type<ngraph::runtime::Tensor>("Tensor")
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

    /////
    ///// Node
    /////
    mod.add_type<ngraph::Node>("Node")
        .method("get_output_size", &ngraph::Node::get_output_size)
        .method("get_output_element_type", &ngraph::Node::get_output_element_type)
        .method("get_output_shape", &ngraph::Node::get_output_shape);

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
    ///// Function
    /////
    mod.add_type<ngraph::Function>("NFunction");
    mod.method("make_function", [](
            const ngraph::NodeVector& nodes,
            const ngraph::ParameterVector& parameters)
    {
        return std::make_shared<ngraph::Function>(nodes, parameters);
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

    mod.method("op_broadcast", [](
        const std::shared_ptr<ngraph::Node> &arg,
        const ngraph::Shape& shape,
        const ngraph::AxisSet& broadcast_axes)
    {
        auto a = std::make_shared<ngraph::op::Broadcast>(arg, shape, broadcast_axes);
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

    mod.method("op_mul", [](
        const std::shared_ptr<ngraph::Node> &arg0,
        const std::shared_ptr<ngraph::Node> &arg1)
    {
        auto a = std::make_shared<ngraph::op::Multiply>(arg0, arg1);
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

    mod.method("op_relu", [](const std::shared_ptr<ngraph::Node> &arg){
        auto a = std::make_shared<ngraph::op::Relu>(arg);
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
                    outputs.push_back(*jlcxx::unbox_wrapped_ptr< std::shared_ptr<ngraph::runtime::Tensor> >(i));

                for (auto i: jl_inputs)
                    inputs.push_back(*jlcxx::unbox_wrapped_ptr< std::shared_ptr<ngraph::runtime::Tensor> >(i));

                executable->call(outputs, inputs);
            }
        );


   // Onnx models
    mod.method("import_onnx_model", [](std::string file){
        return ngraph::onnx_import::import_onnx_model(file);
    });

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



    // PMDK stuff
    //mod.add_type<ngraph::PoolManager>("PoolManager")
    //    .method("getinstance", &ngraph::PoolManager::getinstance)
    //    .method("setpool", &ngraph::PoolManager::setpool)
    //    .method("createpool", &ngraph::PoolManager::createpool)
    //    .method("openpool", &ngraph::PoolManager::openpool)
    //    .method("closepool", &ngraph::PoolManager::closepool)
    //    .method("enablepmem", &ngraph::PoolManager::enablepmem)
    //    .method("disablepmem", &ngraph::PoolManager::disablepmem)
    //    .method("isenabled", &ngraph::PoolManager::isenabled);
}

# Steps for importing ONNX graphs
function onnx_import(filepath)
    fn = NFunction(Lib.read_onnx(filepath))
    return fn
end

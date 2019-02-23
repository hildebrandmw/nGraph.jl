using BenchmarkTools
using Glob

# Model URLs from https://github.com/onnx/models
const model_urls = [
    # Modilenet
    "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz",
    # Resnet
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.tar.gz",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.tar.gz",
    "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.tar.gz", 
    # Squeezenet
    "https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz",
    # Vgg 19
    "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.tar.gz"
]

function fetchmodels()
    mkpath(joinpath(MODELDIR))
    # Download if it doesn't exist
    for url in model_urls
        localfile = joinpath(MODELDIR, basename(url))
        if !ispath(localfile)
            download(url, localfile)
        end
    end
end

function unpack()
    for file in readdir(MODELDIR)
        if endswith(file, ".tar.gz")
            fullpath = joinpath(MODELDIR, file)
            run(`tar -xvf $fullpath -C $MODELDIR`)
        end
    end
end

# Random benchmarking stuff
function setup(pool = "/mnt/everyone/test.pool", size = 2 ^ 34)
    ispath(pool) && rm(pool)
    # Get the pool manager and create the pool
    manager = getinstance()
    setpool(manager, pool)
    disablepmem(manager)
    createpool(manager, UInt(size))
    return nothing
end

function benchmark()
    files = filter(isdir, joinpath.(MODELDIR, readdir(MODELDIR)))
    backend = create("CPU")
    for file in files
        println(file)
        benchmark(basename(file), backend)
    end
    finalize(backend)
end

function benchmark(model, backend = create("CPU"))
    dir = joinpath(MODELDIR, model)
    onnx = glob("*.onnx", dir)
    @info "Importing Model"
    model = import_onnx_model(joinpath(MODELDIR, model, first(onnx)))

    # Compile without performance counters
    @info "Compiling Model" 
    executable = compile(backend, model, false)

    # Make tensors for input and output 
    element_type = gettype("f32") 

    @info "Creating Tensors" 
    input = create_tensor(backend, element_type, UInt.([1, 3, 224, 224]))
    output = create_tensor(backend, element_type, UInt.([1, 1000]))

    input_vector = Vector{Any}([input])
    output_vector = Vector{Any}([output])

    @info "Calling Function"
    b = @benchmark call($executable, $output_vector, $input_vector)
    display(b)

    # The CPU backend keeps a list of functions that have been compiled. Here, we delete
    # the compiled function to avoid a memory leak.
    remove_compiled_function(backend, executable)
    #finalize(executable)

    return nothing
end

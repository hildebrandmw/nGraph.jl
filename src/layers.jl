# Custom Layers mapping to some deeper ngraph stuff
#
# ngraph wants to know the direction of a cell, so we use an enum.
# Set the values of the enum to be the corresponding values requested by the ngraph
# constructor.
@enum LSTMDirection::UInt64 FORWARD=1 BIDIRECTIONAL=2
struct RnnLSTM{T <: Flux.LSTMCell}
    # LSTM Layer from flux
    lstm::T
    direction::LSTMDirection
    num_timesteps::Int64
    num_fused_layers::Int64
end

function RnnLSTM(
        in::Integer, 
        out::Integer, 
        direction::LSTMDirection, 
        num_timesteps, 
        num_fused_layers = 1; 
        init = Flux.glorot_uniform
    )

    # Much (pretty much all) of the numbers here are reverse engineered from the ngraph 
    # source code.
    direction_multiplier = Int(direction)
    cell = Flux.LSTMCell(
        # Wi:Flux - weights_layer:ngraph
        Flux.param(init(4 * out, out * direction_multiplier * num_fused_layers)),
        # Wh:Flux - weights_iter:ngraph
        Flux.param(init(4 * out, in * direction_multiplier * num_fused_layers)),
        # bias
        Flux.param(init(4 * out * direction_multiplier * num_fused_layers) ),
        # initial states
        Flux.param(zeros(Float32, out)),
        Flux.param(zeros(Float32, out)),
    )

    # Taken from Flux.jl
    cell.b.data[Flux.gate(out, 2)] .= 1
    return RnnLSTM(
        cell,
        direction,
        num_timesteps,
        num_fused_layers
    )
end

function (L::RnnLSTM)(x)
    # Construct `src_iter`, `weights_layer`, and `weights_iter` and `bias`, from the
    # members of LSTM
    src_iter = cat(Node(L.lstm.h), Node(L.lstm.c); dims = 1)
    src_iter = broadcast(src_iter, (size(src_iter, 1), size(x, 2)))
    weights_layer = Node(L.lstm.Wh)
    weights_iter = Node(L.lstm.Wi)
    bias = Node(L.lstm.b)

    @show size(x)
    @show size(src_iter)
    @show size(weights_layer)
    @show size(weights_iter)

    op = Node(Lib.op_lstm_rnn(
        getpointer(x),
        getpointer(src_iter),
        getpointer(weights_layer),
        getpointer(weights_iter),
        getpointer(bias),
        convert(UInt, L.num_timesteps),
        UInt(L.direction),
        convert(UInt, L.num_fused_layers),
    ))

    # This node has two outputs.
    #
    # Usually, we will just want the first and the second must become an implicit output
    # (hence the use of __register)
    out = get_output_element(op, 1)
    hidden_out = get_output_element(op, 2)
    __register(hidden_out)

    return out, hidden_out
end


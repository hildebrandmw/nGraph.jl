# Toggles for nGraph environmental variables
enable_codegen() = ENV["NGRAPH_CODEGEN"] = 1
disable_codegen() = delete!(ENV, "NGRAPH_CODEGEN")

"""
    codegen_debug()

Set environmental variables to allow internal codegeneration to display debug information
to `stdout`.
"""
function codegen_debug()
    # The actual values that are assigned don't actually matter. It just matters that these
    # variables are assigned to begin with.
    ENV["NGRAPH_COMPILER_DEBUGINFO_ENABLE"] = true
    ENV["NGRAPH_COMPILER_DIAG_ENABLE"] = true
    ENV["NGRAPH_COMPILER_REPORT_ENABLE"] = true
end

enable_timing() = ENV["NGRAPH_CPU_TRACING"] = true
disable_timing() = delete!(ENV, "NGRAPH_CPU_TRACING")

# Pass Enables
abstract type AbstractPassAttribute end
struct ReuseMemory <: AbstractPassAttribute end
express(::ReuseMemory) = "ReuseMemory=1"

set_pass_attributes(x::AbstractPassAttribute...) = (ENV["NGRAPH_PASS_ATTRIBUTES"] = join(express.(x), ";"))

# Build options

const DEFAULTS = (
    # Build nGraph with PMDK
    PMDK = EXPERIMENTAL,
    GPU = false,
    # Build nGraph in debug mode
    DEBUG = false,
    NUMA = false,
)

function _default_build_parameters()
    open(joinpath(DEPSDIR, "build.json"); write = true) do io
        JSON.print(io, DEFAULTS, 4)
    end
end

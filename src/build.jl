# Build options

const DEFAULTS = (
    # Build nGraph with PMDK
    PMDK = EXPERIMENTAL,
    # Build nGraph in debug mode
    DEBUG = false,
)

function _default_build_parameters()
    open(joinpath(DEPSDIR, "build.json"); write = true) do io
        JSON.print(io, DEFAULTS, 4)
    end
end

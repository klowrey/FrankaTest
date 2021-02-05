# FrankaTest

```bash
$> julia --project
```

```julia
julia> ] 
FrankaTest> instantiate
FrankaTest> ctrl-c
julia> include("src/testfranka.jl")
julia> env = FrankaPickup();
julia> jacctrl(env);

```

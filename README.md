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

To move around the object you can double click on it, then ctrl click and drag it to move.

You can press ctrl-right arrow to switch from the passive visualizer to the controller mode to activate the controller.

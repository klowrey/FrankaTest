# FrankaTest


## Interactive Control

```bash
$> export JULIA_NUM_THREADS=4
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

## Policy Learning

Currently the rewards function and NPG parameters probably need some tuning, as "pick-up" does not happen. The kinematic structure of the robot also needs some consideration as it's simply not great and it hits joint singularities all the time. Lastly, given that it is position controlled, that may effect how the policy learning comences; `setaction!` may need to add a difference to the controls rather than setting the controls directly.

Assuming you've instantiated the project as above (and will visualize with a single thread)...

```bash
$> export JULIA_NUM_THREADS=8 # or however many CPU cores you have
$> julia --project
```
```julia
julia> include("src/npg.jl")
julia> env = FrankaPickup();
julia> runNPG(; niters=200);
```

In your single threaded visualization REPL:
```julia
julia> vizpolicy(env, "/tmp/frankapickup.jlso");
```


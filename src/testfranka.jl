using LinearAlgebra
using LyceumAI, LyceumBase, LyceumMuJoCo, LyceumMuJoCoViz, Shapes
using LyceumBase.Tools
using UniversalLogger
using Base: @propagate_inbounds
using Random
import LyceumBase: tconstruct, _rollout
using LyceumMuJoCo: fastreset_nofwd!
using Statistics, StaticArrays, UnsafeArrays
using Base.Iterators
using Distributions, Distances
using MuJoCo

#using Optim

include("franka.jl")

function randrollouts(env::FrankaPickup, T=300, K=10)
    s = length(statespace(env))
    states = [ zeros(s, T) for i=1:K ]

    nu = length(actionspace(env))
    
    for k=1:K
        ctrl = zeros(nu)
        state = zeros(s)
        randreset!(env)
        for t=1:T
            ctrl .+= randn(nu)*0.05
            setaction!(env, ctrl)
            step!(env)
            getstate!(state, env)
            states[k][:,t] .= state
        end
    end

    visualize(env, trajectories=states)
end

function jacctrl(env::FrankaPickup, gain=10.0)
    jacp = zeros(env.sim.m.nv, 3)
    jacr = zeros(env.sim.m.nv, 3)
    ctrl = zeros(env.sim.m.nu)
    positiondelta = zeros(3)

    function ctrlfn(env)
        m, d = env.sim.m, env.sim.d
        MuJoCo.MJCore.mj_jacSite(m, d, vec(jacp), vec(jacr), 0)

        # have the object as the target site through it's body_xpos
        #positiondelta .= gain .* (SPoint3D(d.xpos, m.nbody) - SPoint3D(d.site_xpos, 1))
        diff = SPoint3D(d.xpos, m.nbody) - SPoint3D(d.site_xpos, 1)
        positiondelta .+= 0.8 * (diff - positiondelta)
        
        # already transposed as mujoco is row-major; julia is col-major
        #ctrl[1:7] .= gain * jacp[1:7,:] * positiondelta
        ctrl[1:7] .= gain * jacp[1:7,:] / positiondelta'

        ctrl[8] = 0.04 # maximally open

        setaction!(env, ctrl)

        forward!(env.sim)
    end
    visualize(env, controller=ctrlfn)
end


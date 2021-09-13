using LinearAlgebra
using LyceumAI, LyceumBase, LyceumMuJoCo, LyceumMuJoCoViz, Shapes
using LyceumBase.Tools
using UniversalLogger
using Base: @propagate_inbounds
using Random
import LyceumBase: tconstruct, _rollout
using LyceumMuJoCo: fastreset_nofwd!
using Statistics, StaticArrays
using Base.Iterators
#using Distributions, Distances
using MuJoCo
using JLSO
using Flux

#using Optim

function INTEL_DRIVER()
   ENV["MESA_LOADER_DRIVER_OVERRIDE"] = "i965"
end

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

   visualize(env, trajectories=states, windowsize=LyceumMuJoCoViz.RES_HD)
end

function jacctrl(env::FrankaPickup, gain=10.0)
   jacp = zeros(env.sim.m.nv, 3)
   jacr = zeros(env.sim.m.nv, 3)
   ctrl = zeros(length(actionspace(env)))
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
   visualize(env, controller=ctrlfn, windowsize=LyceumMuJoCoViz.RES_HD)
end

function mpcctrl(env::FrankaPickup; H=10, K=4, σ=0.3, λ=0.1)
   state = zeros(length(statespace(env)))
   action = zeros(length(actionspace(env)))

   mppi = MPPI(env_tconstructor = n -> tconstruct(FrankaPickup, n),
               covar = Diagonal(σ * I, size(actionspace(env), 1)),
               lambda = λ,
               H = H,
               K = K,
               gamma = 1.0,
               # The following is for position control based MPC
               #initfn! = (meantraj) -> @inbounds @views meantraj[:, end] .= meantraj[:, end-1]
              )
   mppi.meantrajectory .= env.sim.d.qpos[1:8]
   env.sim.d.ctrl[1:8] .= env.sim.d.qpos[1:8]

   function mpcctrlfn(env)
      getstate!(state, env)
      getaction!(action, state, mppi) # gets action from mppi for state

      setaction!(env, action)
      forward!(env.sim)
   end
   visualize(env, controller=mpcctrlfn, windowsize=LyceumMuJoCoViz.RES_HD)
end


function vizpolicy(env::FrankaPickup, f)
   d = JLSO.load(f)
   states = d[:stocstates][:states][1]
   policy = d[:policy]

   obs = getobs(env) # pre-allocate

   function policyctrl(env)
      getobs!(obs, env)
      setaction!(env, policy(obs))
      forward!(env.sim)
   end

   visualize(env, controller=policyctrl, trajectories=states,
             windowsize=LyceumMuJoCoViz.RES_HD)
end

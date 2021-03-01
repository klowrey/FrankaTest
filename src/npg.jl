using LinearAlgebra, Random, Statistics

using Flux
using Flux: glorot_uniform, paramtype

using UnicodePlots, StaticArrays, JLSO, UnsafeArrays, Distributions

using LyceumBase, LyceumAI, LyceumMuJoCo, MuJoCo, UniversalLogger, Shapes
using LyceumBase.Tools
using LyceumBase.Tools: filter_nt
using Base: @propagate_inbounds
#using Shapes: AbstractVectorShape

include("franka.jl")

function runNPG(; niters=200, plotiter=div(niters,10), seed = Random.make_seed())
    etype = FrankaPickup
    BLAS.set_num_threads(Threads.nthreads())
    seed_threadrngs!(seed)

    e = etype()
    dobs, dact = length(obsspace(e)), length(actionspace(e))

    DT = Float32
    Hmax, K = 500, 32
    N = Hmax * K

    policy = DiagGaussianPolicy(
        multilayer_perceptron(dobs, 64, 64, dact, σ=tanh), #initb=zeros, initb_final=glorot_uniform),
        ones(dact) .*= -0.5
    )
    policy = paramtype(DT, policy)
    policy.meanNN[end].W .*= 1e-2
    policy.meanNN[end].b .*= 1e-2

    value = multilayer_perceptron(dobs, 64, 64, 1, σ=Flux.relu) #, initb=glorot_uniform, initb_final=glorot_uniform)
    valueloss(bl, X, Y) = mse(vec(bl(X)), vec(Y))

    valuetrainer = FluxTrainer(
        optimiser = ADAM(1e-3),
        szbatch = 64,
        lossfn = valueloss,
        stopcb = s->s.nepochs > 2
    )
    value = Flux.paramtype(DT, value)

    timefeatureizer = LyceumAI.TimeFeatures{DT}(
        [1, 2, 3],
        [1, 1, 1],
        1 / 1000
    )

    npg = NaturalPolicyGradient(
        n -> tconstruct(etype, n),
        policy,
        value,
        gamma = 0.995,
        gaelambda = 0.97,
        valuetrainer,
        Hmax=Hmax,
        norm_step_size=0.1,
        N=N,
        value_feature_op = timefeatureizer
    )

    envname = lowercase(string(nameof(etype)))
    savepath = "/tmp/$envname.jlso"

    lg = ULogger()

    for (i, state) in enumerate(npg)
        if i > niters
            break
        end

        push!(lg, :algstate, filter_nt(state, exclude=(:meanbatch, :stocbatch)))

        if mod(i, plotiter) == 0
            x = lg[:algstate]
            display(expplot(
                Line(x[:stoctraj_reward], "StocR"),
                title="NPG Iteration=$i", width=60, height=10
            ))

            display(expplot(
                Line(x[:meantraj_reward], "MeanR"),
                title="NPG Iteration=$i", width=60, height=10
            ))
        end
    end

    JLSO.save(savepath,
              :policy => npg.policy,
              :value => npg.value,
              :etype => etype,
              :meanstates => state.meanbatch,
              :stocstates => state.stocbatch,
              :prng_seed => seed,
              :algstate => lg[:algstate])

    npg
end

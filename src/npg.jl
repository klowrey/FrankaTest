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
    #BLAS.set_num_threads(Threads.nthreads())
    BLAS.set_num_threads(1)
    seed_threadrngs!(seed)

    e = etype()
    dobs, dact = length(obsspace(e)), length(actionspace(e))

    DT = Float64
    Hmax, K = 500, 32
    N = Hmax * K

    policy = DiagGaussianPolicy(
        multilayer_perceptron(dobs, 64, 64, dact, σ=tanh),
        ones(dact) .*= -0.5
    )
    policy = paramtype(DT, policy)
    policy.meanNN[end].W .*= 1e-1
    policy.meanNN[end].b .*= 1e-1

    timefeaturizer = LyceumAI.TimeFeatures{DT}(
                                                [1, 2, 3, 4],
                                                [1, 1, 1, 1],
                                                1 / 1000
                                               )
    value = multilayer_perceptron(dobs+length(timefeaturizer.orders),
                                  64, 64, 1, σ=Flux.relu)
    valueloss(bl, X, Y) =  mse(vec(bl(X)), vec(Y))

    valuetrainer = FluxTrainer(
        optimiser = ADAM(1e-3),
        szbatch = 64,
        lossfn = valueloss,
        stopcb = s->s.nepochs > 2
    )
    value = Flux.paramtype(DT, value)


    npg = NaturalPolicyGradient(
        n -> tconstruct(etype, n),
        policy,
        value,
        gamma = 0.995,
        gaelambda = 0.97,
        valuetrainer,
        Hmax=Hmax,
        norm_step_size=0.05,
        N=N,
        value_feature_op = timefeaturizer
    )

    envname = lowercase(string(nameof(etype)))
    savepath = "/tmp/$envname.jlso"

    lg = ULogger()

    log = Dict()
    for (i, state) in enumerate(npg)
        if i > niters
            push!(lg, :meanbatch, state.meanbatch)
            push!(lg, :stocbatch, state.stocbatch)
            break
        end

        push!(lg, :algstate, filter_nt(state, exclude=(:meanbatch, :stocbatch)))

        if mod(i, plotiter) == 0
            x = lg[:algstate]
            display(expplot(
                Line(x[:stoctraj_reward], "StocR"),
                Line(x[:meantraj_reward], "MeanR"),
                title="NPG Iteration=$i", width=50, height=6
            ))
            display(expplot(
                Line(x[:stoctraj_eval], "Stoc Eval"),
                Line(x[:meantraj_eval], "Mean Eval"),
                width=50, height=6
            ))
            println("elapsed_sampled  = $(state.elapsed_sampled)")
            println("elapsed_gradll   = $(state.elapsed_gradll)")
            println("elapsed_vpg      = $(state.elapsed_vpg)")
            println("elapsed_cg       = $(state.elapsed_cg)")
            println("elapsed_valuefit = $(state.elapsed_valuefit)")
        end
    end

    JLSO.save(savepath,
              :policy => npg.policy,
              :value => npg.value,
              :etype => etype,
              :meanstates => lg[:meanbatch],
              :stocstates => lg[:stocbatch],
              :prng_seed => seed,
              :algstate => lg[:algstate])

    npg
end

module NeuralNetworkModule
    export NeuralNetworkStruct
    mutable struct NeuralNetworkStruct
        layerSizes :: Vector{Int64}
        weightMatricies :: Vector{Matrix{Float64}}
        biasVectors :: Vector{Vector{Float64}}

        function NeuralNetworkStruct(layerSizes::Int64...) :: NeuralNetworkStruct
            return (NeuralNetworkStruct ∘ collect)(layerSizes)
        end

        function NeuralNetworkStruct(layerSizes :: Vector{Int64}) :: NeuralNetworkStruct
            amountOfLayers = length(layerSizes)
            @assert amountOfLayers ≥ 2 "A neural network must have at most 2 layers, an input and an output layer"

            W = Matrix{Float64}[[;;]]
            b = Vector{Float64}[[]]

            for i ∈ 2:amountOfLayers
                currentLayerSize, previousLayerSize = layerSizes[i], layerSizes[i-1]
                @assert currentLayerSize > 0 && previousLayerSize > 0 "Layer sizes must be bigger then 0!!"

                push!(W, rand(Float64,currentLayerSize, previousLayerSize))
                push!(b, rand(Float64, currentLayerSize))
            end

            return new(copy(layerSizes), W, b)
        end
    end

    σ(x :: Real) :: Real = 1/(1 + exp(-x))

    function δσ(x :: Real) :: Real
        SigmoidValue = σ(x)
        return SigmoidValue*(1 - SigmoidValue)
    end

    function (network :: NeuralNetworkStruct)(X::Float64...)
        return network(collect(X))
    end

    function (network :: NeuralNetworkStruct)(X :: Vector{Float64})
        @assert length(X) == network.layerSizes[1] "Input vector ($(length(X))) has to be the same length as the input layer ($(network.layerSizes[1]))"

        a,z = Vector{Float64}[], Vector{Float64}[[]]
        push!(a, copy(X))

        for l ∈ 2:length(network.layerSizes)
            push!(z, network.weightMatricies[l] * a[l-1] + network.biasVectors[l])
            push!(a, σ.(z[l]) )            
        end

        return a, z
    end

    function TrainNetwork(network :: NeuralNetworkStruct, trainigData)

    end
end
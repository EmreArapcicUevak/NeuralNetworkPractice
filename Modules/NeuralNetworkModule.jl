module NeuralNetworkModule
    using Term.Progress

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

        a,z = Vector{Float64}[copy(X)], Vector{Float64}[[]]

        for l ∈ 2:length(network.layerSizes)
            push!(z, network.weightMatricies[l] * a[l-1] + network.biasVectors[l])
            push!(a, σ.(z[l]) )            
        end

        return a, z
    end

    function (network :: NeuralNetworkStruct)(X :: Matrix{Float64})
        featureSize, numberOf = size(X)
        @assert featureSize == network.layerSizes[1]  "Number of features has to be same as the input layer"
        A,Z = Matrix{Float64}[copy(X)], Matrix{Float64}[[;;]]

        for l ∈ 2:length(network.layerSizes)
            push!(Z, network.weightMatricies[l] * A[l-1] .+ network.biasVectors[l])
            push!(A, σ.(Z[l]) )            
        end
    
        return A, Z
    end

    function TrainNetwork(network :: NeuralNetworkStruct; trainingData :: Matrix{Float64}, correctAnswers :: Matrix{Float64}, epoch :: Int, η::Float64 = 0.01)
        @assert epoch > 0 "Epochs have to be greater then 0"
        @assert η > 0 && η < 1 "Eta has to be between 0 and 1 excluding the boundaries"

        numOfCorrectSolutions, amountOfData = size(correctAnswers)
        @assert numOfCorrectSolutions == network.layerSizes[end] "Number of correct solutions has to be equal to the number of neurons in the output layer"

        coeficient = η / amountOfData
        @track for i ∈ 1:epoch
            A, Z = network(trainingData)

            ΔC = 2 * (A[end] - correctAnswers)
            Δ = [ΔC .* σ.(Z[end])]
            
            for l ∈ length(network.layerSizes)-1:-1:2
                pushfirst!(Δ,network.weightMatricies[l+1]' * Δ[begin] .* δσ.(Z[l]))
            end

            pushfirst!(Δ, [;;])

            δCδW = Matrix{Float64}[[;;]]

            for l ∈ 2:length(network.layerSizes)
                push!(δCδW, Δ[l] * (A[l-1])')
            end 

            δCδb = Vector{Float64}[[]]

            for l ∈ 2:length(network.layerSizes)
                push!(δCδb, Δ[l] * ones(size(Δ[l])[2]))
            end 

            for l ∈ 2:length(network.layerSizes)
                network.weightMatricies[l] -= coeficient * δCδW[l]
                network.biasVectors[l] -=  coeficient * δCδb[l]
            end
        end
    end
end
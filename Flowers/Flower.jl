using Plots, Revise
Plots.plotlyjs()

# each point is length, width, type (0, 1)

data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1,  1],
        [1,    1,  0]]

mystery_flower = [4.5, 1]

Sigmoid(x :: Real) :: Real = 1/(1 + exp(-x))
function δSigmoid(x :: Real) :: Real
    σ = Sigmoid(x)
    return σ*(1 - σ)
end

plotRange = LinRange(-20,20,9000)
plot(Sigmoid, plotRange)
plot!(δSigmoid,plotRange)

function Predict(input :: Vector{T}, weigthVector :: Vector{Float64}) :: Real where T <: Real
    @assert length(input) == 2 "Input vector has to be of dimention 2"
    input = [input; 1]

    return Sigmoid(input' * weigthVector)
end

function Train(epoch :: Int, weigthVector = rand(3) :: Vector{Float64}; η = 0.001) :: Vector{Float64}
    @assert epoch > 0 "You can't have less then 1 epoch"
    @assert η > 0 "Eta must be positive"

    for itteration ∈ 1:epoch
        println("Itteration: $itteration")
        gradient = zeros(Float64, 3)
        for dataPoint ∈ data
            input = dataPoint[1:2]
            yₛ = dataPoint[3]

            input = [input; 1]'
            a = input * weigthVector
            yₚ = Sigmoid(a)
            common = 2 * (yₚ - yₛ) * δSigmoid(a) 
            gradient += [common *dataPoint[1],common*dataPoint[2], common]
        end

        weigthVector = weigthVector - η*gradient
    end

    return weigthVector
end

function GetColor(prediction :: Float64) :: Symbol
    @assert (prediction ≤ 1 && prediction ≥ 0) "Prediction has to be between 0 and 1"
    if prediction == 0.5
        return :gray
    elseif prediction > 0.5
        return :red
    else
        return :blue
    end
end

weightVector = Train(999_999)
Predict(mystery_flower, weightVector)

# Initialize an empty 3D scatter plot
plt = scatter(zeros(0), zeros(0), label="")

for flowerPoint in data
    scatter!(plt, [flowerPoint[1]], [flowerPoint[2]], color=GetColor(flowerPoint[3]), label="")
end

scatter!(plt, [mystery_flower[1]], [mystery_flower[2]], color= :orange, label = "unknown flower")

xRange = LinRange(1,6,40)
yRange = LinRange(0,2,40)

colorOfPredicition = GetColor ∘ Predict

for x ∈ xRange
    for y ∈ yRange
        scatter!(plt, [x], [y], color = colorOfPredicition([x,y], weightVector), label = "",alpha = 0.2)
    end
end

xlabel!(plt, "Pedal Length")
ylabel!(plt, "Pedal Width")
savefig("Flower Predicition.png")

println(weightVector)
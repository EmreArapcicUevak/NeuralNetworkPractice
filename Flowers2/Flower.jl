modulePath = pwd() * "/Modules/"
if modulePath ∉ LOAD_PATH
    push!(LOAD_PATH, modulePath);
end

using Revise, Plots
using NeuralNetworkModule

Plots.plotlyjs()


# Generate circular data
function generate_circular_data(inner_radius, outer_radius, num_points)
    data = []
    solutions = []

    for _ in 1:num_points
        angle = 2 * π * rand()
        radius = inner_radius  + (outer_radius - inner_radius) * rand()

        x = radius * cos(angle)
        y = radius * sin(angle)

        if radius <= (inner_radius + outer_radius) / 2
            push!(solutions, 1)  # red
        else
            push!(solutions, 0)  # blue
        end

        push!(data, [x, y])
    end

    data = hcat(data...)
    solutions = Float64[solutions;]  # Convert solutions to a row vector
    return data, solutions
end

inner_radius = 1.0
outer_radius = 2.0
num_points = 500

data, solutions = generate_circular_data(inner_radius, outer_radius, num_points)
solutions = (Matrix ∘ transpose)(solutions)

flowerNetwork = NeuralNetworkStruct(2,5,5,1)
NeuralNetworkModule.TrainNetwork(flowerNetwork; trainingData = data, correctAnswers = solutions, epoch = 999_999)


mystery_flower = [1.5, 0.5]
flowerNetwork(mystery_flower)[1][end]

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

plt = scatter(zeros(0), zeros(0), label="")
for i in 1:size(data)[2]
    x,y,color = data[1,i], data[2,i], GetColor(solutions[i])
    scatter!(plt, [x], [y], color=color, label="")
end

scatter!(plt, [mystery_flower[1]], [mystery_flower[2]], color= :orange, label = "unknown flower")

xRange = LinRange(-3,3,100)
yRange = LinRange(-3,3,100)
colorOfPredicition(X) = GetColor(flowerNetwork(X)[1][end][1])

for x ∈ xRange
    for y ∈ yRange
        scatter!(plt, [x], [y], color = colorOfPredicition([x,y]), label = "",alpha = 0.2)
    end
end

xlabel!(plt, "Pedal Length")
ylabel!(plt, "Pedal Width")

savefig("AdvanceFlower.png")
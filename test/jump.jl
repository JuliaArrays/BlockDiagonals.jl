@testset "JuMP" begin
    num_nodes = 2
    num_targets = 2
    nodes = [randstring(3) for _ in 1:num_nodes]
    targets = [DateTime(2020, 1, 1, h) for h in 1:num_targets]

    dists = Vector{MvNormal}()
    for k in targets
        mu = randn(num_nodes * num_targets)
        X = rand(num_nodes * num_targets, num_nodes * num_targets)
        sigma = X * X' + I
        push!(dists, MvNormal(mu, sigma))
    end

    covs = [Matrix(cov(d)) for d in dists]
    means = [mean(d) for d in dists]

    preds = (mean=vcat(means...), cov=BlockDiagonal(covs), target=targets, nodes=nodes)

    @testset "Multiplication" begin
        model = JuMP.Model()

        n = length(preds.mean)
        v = (
            supply_mwh=@variable(model, supply_mwh[1:n] >= 0),
            demand_mwh=@variable(model, demand_mwh[1:n] <= 0),
        )

        volume = v.supply_mwh + v.demand_mwh
        normalized_sqrt_cov = cholesky(preds.cov).U / 24

        @test normalized_sqrt_cov * volume == Matrix(normalized_sqrt_cov) * volume
    end
end

### A Pluto.jl notebook ###
# v0.12.15

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 6c73a490-34b5-11eb-1684-4f51731475d2
using Plots, PlutoUI, Flux, Pipe, Statistics, Random, GLM, Distributions; gr();

# ╔═╡ 3f68d22c-34b5-11eb-2d3c-79097d4d2cc1
md"# Impact of coefficients on logistic regression function"

# ╔═╡ 6158c360-34b5-11eb-29a1-798000b3f693
begin
dfd = md"
β₀ $(@bind β0 Slider(-3:0.1:3;show_value=true, default=0.))

β₁$(@bind β1 Slider(-3:0.1:3; show_value=true, default=1.))
	
xᵢ $(@bind xᵢ Slider(-5:.01:5, show_value=true, default=0.))
"
dfd
end

# ╔═╡ 43916b20-3541-11eb-3445-0771204171b3
md"
Show derivative of logistic function?
$(@bind ok1 PlutoUI.CheckBox( default=false))
"

# ╔═╡ 1f0f1a14-354a-11eb-0fe2-a58ce120ecd7
md"
Show loss function of logistic function?
$(@bind ok2 PlutoUI.CheckBox( default=false))
"

# ╔═╡ 662563b0-34b7-11eb-341d-993ab0413dd2
md"###### Side calculation"

# ╔═╡ 07996034-34b6-11eb-0a63-7ff27a61b07c
# Define pi and z function and use @bind-parameters as function parameters
begin
	π(z) = 1 / (1 + exp(-z))
	z(x, β0, β1) = β0 + β1*x 
end;

# ╔═╡ 19eaed72-34b6-11eb-35c1-05ac47ece5a4
x = -5:0.1:5;

# ╔═╡ 41f56c0c-34b6-11eb-23cc-0fd238ac690e
begin
plot(x, π.(z.(x, β0, β1)), 
		linewidth=3, 
		labels="β0=$(β0), β1=$(β1)", 
		title="Logistic function", 
		framestyle=:origin, 
		ylabel="p", 
		xlabel="x",
		#titlefontsize=12,
	)
plot!(x, π.(z.(x, 0, 1)), color="black", linestyle=:dash, label="β0=0, β1=1")
plot!(legend=:left,foreground_color_legend = nothing)
a = scatter!([xᵢ], [π.(z.(xᵢ, β0, β1))], markersize=7, label="", color="red", alpha=1)	
end

# ╔═╡ 33a5b4ac-353a-11eb-3947-17fa6482a9a8
# calculate derivative of loss function
begin
	pifunc(x) = 1 ./ (1 .+ exp(-(β0 .+ β1 .* x)))
	grads = gradient.(pifunc, x)
	dx = [grad[1] for grad in grads]
end;

# ╔═╡ 8edd9dd0-353a-11eb-08ca-8ffc04c462f0
if ok1
	plot(x, dx, 
		title="Derivative of logistic function",
		legend=false, 
		alpha=0.8,
		linewidth=3,
		color="gray",
		framestyle = :origin)
	scatter!([xᵢ], [gradient.(pifunc,xᵢ)[1]], markersize=7, label="", color="red", alpha=1)
end

# ╔═╡ 96152dda-3543-11eb-3eeb-63e8f5fab9d2
# calculate loss function
begin
	b0true, b1true = -2, 2
	y = ifelse.((1 ./ (1 .+ exp.(-(b0true .+ b1true .* x)))) .> 0.5,1,0)
	pi(x,b0, b1) = 1 ./ (1 .+ exp.(.-(b0 .+ b1 .* x)))
	loss(y,x, β0,β1) = sum(log.(pi(x,β0,β1)) .*y .+ log.((1 .- pi(x,β0,β1))) .* (1 .- y))
	
	losses = []
	for (i, b) in enumerate(-3:0.1:3)
		push!(losses, loss(y,x,β0,b))
	end
end

# ╔═╡ 15f6f968-3545-11eb-2e21-cfedbd8fe1ff
begin
	if ok2
		plot(-3:0.1:3, losses ,label="", 
		title="Loss function for β₁ for given β₀\n (True values: β₀ = $(b0true), β₁ = $(b1true))",
		ylabel="Loss", xlabel="β₁")
		xlims!((-3,3))
		ylims!((minimum(losses),50))
		scatter!([β1], [loss(y,x,β0,β1)], markersize=7, label="")
	end
end

# ╔═╡ 08757f1c-3547-11eb-1a8b-45a97cccdce7
l = loss(y,x,β0,β1);

# ╔═╡ 4dd9f33e-354d-11eb-0e15-4b0da599dc6c
if ok2
md"
Loss for given parameter is $(round(l,digits=3)).  
	
The -2LL is $(-2*round(l,digits=3))	
"	
end


# ╔═╡ 2924011a-356d-11eb-35d4-edba543f2ad0
glm((y,x)

# ╔═╡ Cell order:
# ╟─3f68d22c-34b5-11eb-2d3c-79097d4d2cc1
# ╟─6158c360-34b5-11eb-29a1-798000b3f693
# ╟─41f56c0c-34b6-11eb-23cc-0fd238ac690e
# ╟─43916b20-3541-11eb-3445-0771204171b3
# ╟─8edd9dd0-353a-11eb-08ca-8ffc04c462f0
# ╟─1f0f1a14-354a-11eb-0fe2-a58ce120ecd7
# ╟─15f6f968-3545-11eb-2e21-cfedbd8fe1ff
# ╟─4dd9f33e-354d-11eb-0e15-4b0da599dc6c
# ╟─662563b0-34b7-11eb-341d-993ab0413dd2
# ╠═6c73a490-34b5-11eb-1684-4f51731475d2
# ╠═07996034-34b6-11eb-0a63-7ff27a61b07c
# ╠═19eaed72-34b6-11eb-35c1-05ac47ece5a4
# ╠═33a5b4ac-353a-11eb-3947-17fa6482a9a8
# ╠═96152dda-3543-11eb-3eeb-63e8f5fab9d2
# ╠═08757f1c-3547-11eb-1a8b-45a97cccdce7
# ╠═2924011a-356d-11eb-35d4-edba543f2ad0

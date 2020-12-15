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

# ╔═╡ eabc2f5e-3998-11eb-0bc4-fd4806f8ab0d
using DataFrames, CSV, Statistics, Pipe, Plots, PlutoUI

# ╔═╡ cbe65d34-39a4-11eb-2a49-a35066fc9ff9
md"# K-Means clustering algorithm"

# ╔═╡ 86e04224-399d-11eb-1a11-c974f563a245
md"

No. of Clusters (K)
$(@bind K Slider(1:10; show_value=true))

No. of iterations
$(@bind iteration Slider(0:20; show_value=true))
"
#No. of observations
#$(@bind els Slider(4:1000; show_value=true))


# ╔═╡ e26ae6e0-399c-11eb-0571-83a09cf40077
md"## Side calculations"

# ╔═╡ bc00f890-3998-11eb-3287-d9fd1885f7a4
begin


	# Load data
	fpath = "./data/cluster.csv"
	df = CSV.read(fpath, DataFrame) |> Matrix
	data, cluster = df[:,1:2], df[:,3]
	cluster .+= 1 

	euclidean(p,q) = sum((p - q).^2)

	function calc_centroids(data, cluster)
		ks, J = unique(cluster), size(data,2)
		centroids = []
		for (i,k) in enumerate(ks)
			push!(centroids, mean(data[cluster .== k,:],dims=1))
		end
		vcat(centroids...)
	end

	function find_cluster(data, centroids)
		distances = []
		dist = []
		for obs in eachrow(data)
			for centroid in eachrow(centroids)
				push!(dist, euclidean(obs, centroid))
			end
			k = argmin(dist)
			dist = []
			push!(distances, k)
		end
		distances
	end

end;

# ╔═╡ 5825f666-399a-11eb-2d31-57b97255a30b
function kmeans(data, initial, K; iter=10)
	n = size(data,1)
	
	current_cluster = rand(1:1:K,n)

	my_clusters = []
	my_centroids = []
	for i in 1:iter
		new_centroids = calc_centroids(data, current_cluster)
		new_cluster = find_cluster(data,new_centroids) 
		push!(my_centroids, new_centroids)
		push!(my_clusters, new_cluster)
		current_cluster = new_cluster
	end
	my_clusters, my_centroids
end;

# ╔═╡ 0f0b31c0-399b-11eb-3a4c-399d5ada55df
function initialize(data) 
	n  = size(data,1)
	rand(1:1:K,n)
end;

# ╔═╡ 1c041aac-39a4-11eb-1bbe-81efd4642d83
initial = initialize(data);

# ╔═╡ c5491f84-399a-11eb-17f8-77a30f096ec9
all_clusters, all_centroids = kmeans(data, initial, K; iter=20);

# ╔═╡ d4853f76-399c-11eb-145b-6d96407d86e6
begin
	colors = palette(:tab10)
	colors = [c for c in colors]#
	els = size(data,1)
	x1, x2 = data[1:els,1], data[1:els,2]
	scatter(x1,x2,alpha=0.4,legend=false, size=(680,500), xlabel="X1", ylabel="X2")	
	title!("Visualization of cluster formation")
	ylims!(-2,6)
	xlims!(-12,1)
	if iteration == 0
		scatter!(x1,x2,alpha=0.4,markercolor=[colors[i] for i in initial])#initial)
	else
	
		cs = all_centroids[iteration]
		scatter!(x1,x2,alpha=0.4,markercolor=[colors[i] for i in all_clusters[iteration]])#all_clusters[iteration])
		
		scatter!(cs[:,1], cs[:,2],markercolor=colors, alpha=0.9, markersize=10)

	end

end

# ╔═╡ Cell order:
# ╟─cbe65d34-39a4-11eb-2a49-a35066fc9ff9
# ╟─86e04224-399d-11eb-1a11-c974f563a245
# ╟─d4853f76-399c-11eb-145b-6d96407d86e6
# ╟─e26ae6e0-399c-11eb-0571-83a09cf40077
# ╟─eabc2f5e-3998-11eb-0bc4-fd4806f8ab0d
# ╟─bc00f890-3998-11eb-3287-d9fd1885f7a4
# ╟─5825f666-399a-11eb-2d31-57b97255a30b
# ╟─0f0b31c0-399b-11eb-3a4c-399d5ada55df
# ╟─1c041aac-39a4-11eb-1bbe-81efd4642d83
# ╟─c5491f84-399a-11eb-17f8-77a30f096ec9

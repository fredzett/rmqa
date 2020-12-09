using DataFrames
using CSV
using Statistics
using Pipe: @pipe


# Load data
df = CSV.read("./data/cluster.csv", DataFrame) |> Matrix
data, cluster = df[:,1:2], df[:,3]
cluster .+= 1 

euclidean(p,q) = sum((p - q).^2)

function calc_centroids(data, cluster)
	ks, J = unique(cluster), size(data,2)
	#K = length(ks)
	centroids = []#zeros(K, J)
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


K, n = 5, size(data,1)
current_cluster = rand(1:1:K,n)

my_clusters = []
for i in 1:5
    new_cluster = @pipe calc_centroids(data, current_cluster) |> find_cluster(data, _)
    push!(my_clusters, new_cluster)
    current_cluster = new_cluster
end


begin
	foo = randn(3)
	all_foos = []
	for i in 1:5
		new_foo = @pipe sum(foo) |> repeat([_], 3)
		push!(all_foos, new_foo)
		foo = new_foo
	end
end
all_foos

m = zeros(10,10)
m[1,2] = 103
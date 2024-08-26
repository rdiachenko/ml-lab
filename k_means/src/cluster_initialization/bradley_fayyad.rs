use super::macqueen;
use crate::run_k_means;
use rand::seq::SliceRandom;

pub fn init_centroids(data: &Vec<Vec<f64>>, k: usize, j: usize, max_iters: usize, eps: f64) -> Vec<Vec<f64>> {
    // randomly partition data onto j subsets
    let mut rng = rand::thread_rng();
    let mut points: Vec<usize> = (0..data.len()).collect();
    points.shuffle(&mut rng);
    let mut partitions = vec![Vec::new(); j];
    let partition_size = (data.len() + j - 1) / j;
    let mut partition_index = 0usize;
    for p in points {
        let feature = data[p].clone();
        partitions[partition_index].push(feature);
        if partitions[partition_index].len() == partition_size {
            partition_index += 1;
        }
    }

    // cluster each j subset using k-means with MacQueeen, that gives j sets of intermediate centers each with k points
    let mut centroids_per_partition = vec![];
    let mut all_centroids = vec![];
    for partition in partitions {
        let initial_centroids = macqueen::init_centroids(&partition, k);
        let (centroids, _, _, _) = run_k_means(&partition, &initial_centroids, max_iters, eps);
        // combine all j center sets into a single superset
        for centroid in &centroids {
            all_centroids.push(centroid.clone());
        }
        centroids_per_partition.push(centroids);
    }

    // cluster this superset using k-means j times, each time initialized with a different center set j
    // return the initialization subset j that gives the least SSE
    let mut result = vec![];
    let mut min_sse = f64::MAX;
    for initial_centroids in centroids_per_partition {
        let (_, _, sse, _) = run_k_means(&all_centroids, &initial_centroids, max_iters, eps);
        if sse < min_sse {
            min_sse = sse;
            result = initial_centroids;
        }
    }
    result
}

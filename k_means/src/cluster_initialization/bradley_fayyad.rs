//! Bradley-Fayyad Initialization Method
//!
//! This module implements the Bradley-Fayyad method for initializing centroids in k-means clustering.

use super::macqueen;
use crate::run_k_means;
use rand::seq::SliceRandom;

/// Initialize centroids using the Bradley-Fayyad method.
///
/// The Bradley-Fayyad method, also known as the "Refined Start" algorithm, aims to find a good
/// initial set of centroids by running k-means on multiple subsets of the data and then
/// finding the best set of centroids among these results.
///
/// # Algorithm Steps:
/// 1. Randomly partition the data into `j` subsets.
/// 2. Run k-means (using MacQueen's method for initialization) on each subset to get `j` sets of `k` centroids.
/// 3. Combine all these centroids into a superset.
/// 4. Run k-means `j` times on this superset, each time initialized with a different set of centroids from step 2.
/// 5. Return the set of centroids that resulted in the lowest Sum of Squared Errors (SSE).
///
/// # Arguments
///
/// * `data` - A slice of vectors, where each vector represents a data point.
/// * `k` - The number of clusters to create.
/// * `j` - The number of subsets to partition the data into.
/// * `max_iters` - The maximum number of iterations for each k-means run.
/// * `eps` - The convergence threshold for k-means.
///
/// # Returns
///
/// A vector of `k` centroids, where each centroid is a vector of the same
/// dimensionality as the input data points.
///
/// # Example
///
/// ```ignore
/// let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]];
/// let k = 2;
/// let j = 2;
/// let max_iters = 100;
/// let eps = 1e-4;
/// let centroids = init_centroids(&data, k, j, max_iters, eps);
/// assert_eq!(centroids.len(), k);
/// ```
pub fn init_centroids(
    data: &[Vec<f64>],
    k: usize,
    j: usize,
    max_iters: usize,
    eps: f64,
) -> Vec<Vec<f64>> {
    // Randomly partition data into j subsets
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

    // Cluster each j subset using k-means with MacQueen,
    // giving j sets of intermediate centers each with k points
    let mut centroids_per_partition = vec![];
    let mut all_centroids = vec![];
    for partition in partitions {
        let initial_centroids = macqueen::init_centroids(&partition, k);
        let (centroids, _, _, _) = run_k_means(&partition, &initial_centroids, max_iters, eps);

        // Combine all j center sets into a single superset
        for centroid in &centroids {
            all_centroids.push(centroid.clone());
        }
        centroids_per_partition.push(centroids);
    }

    // Cluster this superset using k-means j times,
    // each time initialized with a different center set j.
    // Return the initialization subset j that gives the least SSE
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

//! K-Means++ Initialization Method with Greedy Approach
//!
//! This module implements the k-means++ method for initializing centroids in k-means clustering,
//! combined with a greedy approach for improved quality of the chosen centroids.

use crate::squared_euclidean_dist;
use rand::distributions::Uniform;
use rand::Rng;

/// Initialize centroids using the k-means++ method with a greedy approach.
///
/// This method implements the k-means++ algorithm for initializing centroids, which aims to
/// choose centroids that are both spread out and representative of the data distribution.
/// It also incorporates a greedy approach to potentially improve the quality of the chosen centroids.
///
/// # Algorithm Steps:
/// 1. Choose the first centroid uniformly at random from the data points.
/// 2. For each subsequent centroid:
///    a. Calculate the distance from each point to its nearest existing centroid.
///    b. Choose `samples_per_iter` candidate points, with probability proportional to their squared distance.
///    c. For each candidate, calculate the sum of squared distances if it were chosen as the next centroid.
///    d. Select the candidate that minimizes the sum of squared distances.
/// 3. Repeat step 2 until `k` centroids are chosen.
///
/// # Arguments
///
/// * `data` - A slice of vectors, where each vector represents a data point.
/// * `k` - The number of clusters to create.
/// * `samples_per_iter` - The number of candidate points to consider in each iteration.
///   If set to 0, it uses a default value based on `k`: `floor(2 + log2(k))`.
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
/// let samples_per_iter = 5;
/// let centroids = init_centroids(&data, k, samples_per_iter);
/// assert_eq!(centroids.len(), k);
/// ```
pub fn init_centroids(data: &[Vec<f64>], k: usize, samples_per_iter: usize) -> Vec<Vec<f64>> {
    // Determine the number of local centers to consider in each iteration
    let local_centers_n = if samples_per_iter > 0 {
        samples_per_iter
    } else {
        (2.0 + (k as f64).log2()).floor() as usize
    };

    let mut rng = rand::thread_rng();

    // Choose the first centroid randomly
    let initial_centroid = rng.gen_range(0..data.len());
    let mut result = vec![data[initial_centroid].clone()];

    // Initialize distances and sum of squared errors
    let mut min_dists = vec![f64::MAX; data.len()];
    let mut sse = 0f64;

    // Calculate initial distances
    for p in 0..data.len() {
        min_dists[p] = squared_euclidean_dist(&data[p], &data[initial_centroid]);
        sse += min_dists[p];
    }

    let range = Uniform::new(0.0, 1.0);

    // Choose remaining k-1 centroids
    for _ in 1..k {
        // Calculate cumulative sum of distances
        let dists_cumsum: Vec<f64> = min_dists
            .iter()
            .scan(0.0, |acc, &dist| {
                *acc += dist;
                Some(*acc)
            })
            .collect();

        // Choose candidate centroids
        let centroid_candidates: Vec<usize> = (0..local_centers_n)
            .map(|_| {
                let rand_val = rng.sample(range) * sse;
                dists_cumsum
                    .iter()
                    .position(|&dist| dist > rand_val)
                    .unwrap_or(0)
            })
            .collect();

        // Calculate distances to candidate centroids
        let mut dist_to_candidates = Vec::new();
        for candidate_index in 0..centroid_candidates.len() {
            dist_to_candidates.push(Vec::new());
            let c = centroid_candidates[candidate_index];
            for p in 0..data.len() {
                let dist = squared_euclidean_dist(&data[p], &data[c]);
                dist_to_candidates[candidate_index].push(dist);
            }
        }

        // Find the best candidate centroid
        let mut best_centroid_candidate: usize = 0;
        let mut best_sse = f64::MAX;
        let mut best_min_dists = vec![];

        for candidate_index in 0..centroid_candidates.len() {
            let mut new_min_dists = vec![];
            let mut new_sse = 0.0;
            for (dist_index, min_dist) in min_dists.iter().enumerate() {
                let dist = f64::min(*min_dist, dist_to_candidates[candidate_index][dist_index]);
                new_min_dists.push(dist);
                new_sse += dist;
            }
            if new_sse < best_sse {
                best_sse = new_sse;
                best_min_dists = new_min_dists;
                best_centroid_candidate = centroid_candidates[candidate_index];
            }
        }

        // Add the best candidate to the result
        result.push(data[best_centroid_candidate].clone());
        sse = best_sse;
        min_dists = best_min_dists;
    }

    result
}

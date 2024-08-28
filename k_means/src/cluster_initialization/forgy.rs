//! Forgy Initialization Method
//!
//! This module implements the Forgy method for initializing centroids in k-means clustering.

use crate::compute_centroids;
use rand::distributions::{Distribution, Uniform};

/// Initialize centroids using the Forgy method.
///
/// The Forgy method randomly assigns each data point to one of k clusters, then
/// computes the centroids of the resulting clusters.
///
/// # Arguments
///
/// * `data` - A slice of vectors, where each vector represents a data point.
/// * `k` - The number of clusters to create.
///
/// # Returns
///
/// A vector of `k` centroids, where each centroid is a vector of the same
/// dimensionality as the input data points.
///
/// # Example
///
/// ```ignore
/// let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
/// let k = 2;
/// let centroids = init_centroids(&data, k);
/// assert_eq!(centroids.len(), k);
/// ```
pub fn init_centroids(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let range = Uniform::from(0..k);
    let mut rng = rand::thread_rng();

    let clusters: Vec<usize> = data.iter().map(|_| range.sample(&mut rng)).collect();

    compute_centroids(data, &clusters, k)
}

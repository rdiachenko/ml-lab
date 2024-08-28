//! MacQueen Initialization Method
//!
//! This module implements the MacQueen method for initializing centroids in k-means clustering.

use rand::prelude::SliceRandom;

/// Initialize centroids using the MacQueen method.
///
/// The MacQueen method, also known as random sampling, simply selects k random
/// data points from the dataset to serve as initial centroids. This method is
/// computationally efficient but can sometimes lead to poor initial centroids
/// if unlucky selections are made.
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
/// let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]];
/// let k = 2;
/// let centroids = init_centroids(&data, k);
/// assert_eq!(centroids.len(), k);
/// ```
///
/// # Note
///
/// While this method is fast, it doesn't guarantee an optimal starting position
/// for the k-means algorithm. In some cases, especially with small datasets or
/// unlucky selections, it might lead to slower convergence or suboptimal final
/// clustering results compared to more sophisticated initialization methods.
pub fn init_centroids(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    data.choose_multiple(&mut rng, k).cloned().collect()
}

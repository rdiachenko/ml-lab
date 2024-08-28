//! Maximin Initialization Method
//!
//! This module implements the Maximin method for initializing centroids in k-means clustering.

use crate::squared_euclidean_dist;
use rand::Rng;
use std::collections::HashSet;

/// Initialize centroids using the Maximin method.
///
/// The Maximin method aims to spread out the initial centroids by iteratively
/// selecting points that are farthest from the already chosen centroids. This
/// approach tends to provide a good spread of initial centroids across the dataset.
///
/// # Algorithm Steps:
/// 1. Choose the first centroid randomly from the dataset.
/// 2. For each subsequent centroid:
///    a. For each data point not yet chosen as a centroid, find its distance to the nearest existing centroid.
///    b. Choose the data point with the maximum distance as the next centroid.
/// 3. Repeat step 2 until `k` centroids are chosen.
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
/// While this method generally provides a good spread of initial centroids,
/// it can be sensitive to outliers in the dataset. In datasets with significant
/// outliers, this method might choose some of those outliers as initial centroids.
pub fn init_centroids(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut centroids = HashSet::new();
    let mut rng = rand::thread_rng();

    // Choose the first centroid randomly
    centroids.insert(rng.gen_range(0..data.len()));

    // Choose remaining k-1 centroids
    while centroids.len() < k {
        let mut max_dist = f64::MIN;
        let mut centroid = 0usize;

        // Find the point with maximum distance from existing centroids
        for p in 0..data.len() {
            if !centroids.contains(&p) {
                let mut min_dist = f64::MAX;
                let mut min_dist_point = p;

                // Find the nearest centroid to this point
                for c in &centroids {
                    let dist = squared_euclidean_dist(&data[p], &data[*c]);
                    if dist < min_dist {
                        min_dist = dist;
                        min_dist_point = p;
                    }
                }

                // Update the max distance if this point is farther
                if min_dist > max_dist {
                    max_dist = min_dist;
                    centroid = min_dist_point;
                }
            }
        }

        // Add the point with max distance as the next centroid
        centroids.insert(centroid);
    }

    // Convert centroid indices to actual data points
    centroids.iter().map(|c| data[*c].clone()).collect()
}

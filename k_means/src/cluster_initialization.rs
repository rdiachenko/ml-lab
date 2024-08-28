//! Centroid Initialization Strategies
//!
//! This module provides various strategies for initializing centroids in k-means clustering.
//! It includes several well-known methods, each implemented in its own submodule.

pub mod bradley_fayyad;
pub mod forgy;
pub mod kmeanspp;
pub mod macqueen;
pub mod maximin;

/// Enum representing different centroid initialization strategies
///
/// This enum allows users to select which initialization method to use
/// when running the k-means algorithm.
///
/// # Variants
///
/// * `Forgy` - Randomly assigns each data point to one of k clusters
/// * `MacQueen` - Selects k unique data points at random as initial centroids
/// * `Maximin` - Iteratively selects points farthest from existing centroids
/// * `BradleyFayyad` - Runs k-means on subsets of data to find good initial centroids
/// * `KmeansPP` - Selects initial centroids with probability proportional to their squared distance from existing centroids
/// * `GreedyKmeansPP` - A variant of KmeansPP that considers multiple candidates at each step
///
/// # Example
///
/// ```
/// # use k_means::CentroidInitStrategy;
/// let strategy = CentroidInitStrategy::KmeansPP;
/// // Use 'strategy' when initializing your k-means algorithm
/// ```
#[derive(Debug)]
pub enum CentroidInitStrategy {
    Forgy,
    MacQueen,
    Maximin,
    BradleyFayyad,
    KmeansPP,
    GreedyKmeansPP,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forgy_init_centroids() {
        let k: usize = 1;
        let centroids = forgy::init_centroids(&data(), k);
        assert(&centroids, k);
    }

    #[test]
    fn test_macqueen_init_centroids() {
        let k: usize = 5;
        let centroids = macqueen::init_centroids(&data(), k);
        assert(&centroids, k);
    }

    #[test]
    fn test_maximin_init_centroids() {
        let k: usize = 4;
        let centroids = maximin::init_centroids(&data(), k);
        assert(&centroids, k);
    }

    #[test]
    fn test_bradley_fayyad_init_centroids() {
        let k: usize = 3;
        let centroids = bradley_fayyad::init_centroids(&data(), k, 2, 10, 1e-3);
        assert(&centroids, k);
    }

    #[test]
    fn test_kmeanspp_init_centroids() {
        let k: usize = 6;
        let centroids = kmeanspp::init_centroids(&data(), k, 1);
        assert(&centroids, k);
    }

    #[test]
    fn test_greedy_kmeanspp_init_centroids() {
        let k: usize = 2;
        let centroids = kmeanspp::init_centroids(&data(), k, 0);
        assert(&centroids, k);
    }

    fn data() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![2.0, 3.0],
            vec![5.0, 6.0],
            vec![5.5, 6.5],
            vec![6.0, 7.0],
        ]
    }

    fn assert(centroids: &[Vec<f64>], k: usize) {
        assert_eq!(centroids.len(), k);
        assert!(
            centroids
                .iter()
                .all(|centroid| centroid.iter().all(|&x| (1.0..=7.0).contains(&x))),
            "Some centroids have non-positive values: {:?}",
            centroids
        );
    }
}

mod cluster_initialization;

pub use crate::cluster_initialization::CentroidInitStrategy;
use crate::cluster_initialization::CentroidInitStrategy::*;
use crate::cluster_initialization::forgy;
use crate::cluster_initialization::macqueen;
use crate::cluster_initialization::maximin;
use crate::cluster_initialization::bradley_fayyad;
use crate::cluster_initialization::kmeanspp;

const MAX_ITERATIONS: usize = 100;
const EPS: f64 = 1e-6;

pub fn cluster(data: &Vec<Vec<f64>>, k: usize, init_strategy: &CentroidInitStrategy) -> (Vec<Vec<f64>>, Vec<usize>, f64, usize) {
    println!("finding initial clusters");
    let initial_centroids = match init_strategy {
        Forgy => forgy::init_centroids(&data, k),
        MacQueen => macqueen::init_centroids(&data, k),
        Maximin => maximin::init_centroids(&data, k),
        BradleyFayyad => bradley_fayyad::init_centroids(&data, k, 10, MAX_ITERATIONS, EPS),
        KmeansPP => kmeanspp::init_centroids(&data, k, 1),
        GreedyKmeansPP => kmeanspp::init_centroids(&data, k, 0)
    };
    println!("running k-means clustering");
    run_k_means(&data, &initial_centroids, MAX_ITERATIONS, EPS)
}

fn run_k_means(data: &Vec<Vec<f64>>, initial_centroids: &Vec<Vec<f64>>, max_iters: usize, eps: f64) -> (Vec<Vec<f64>>, Vec<usize>, f64, usize) {
    let k = initial_centroids.len();
    let mut centroids = initial_centroids.clone();
    let mut clusters = vec![0; data.len()];
    let mut sse = f64::MAX;
    let mut iters = 0;
    for _i in 0..max_iters {
        let (cur_clusters, cur_sse) = compute_clusters(data, &centroids);
        let converged = (sse - cur_sse) / cur_sse < eps;
        sse = cur_sse;
        iters += 1;
        clusters = cur_clusters;
        println!("iteration {}/{}, sse {}", iters, max_iters, sse);
        if converged {
            break;
        }
        centroids = compute_centroids(data, &clusters, k);
    }
    (centroids, clusters, sse, iters)
}

fn compute_clusters(data: &Vec<Vec<f64>>, centroids: &Vec<Vec<f64>>) -> (Vec<usize>, f64) {
    let mut clusters: Vec<usize> = vec![0; data.len()];
    let mut sse = 0f64;
    for point in 0..data.len() {
        let mut min_dist = f64::MAX;
        for centroid in 0..centroids.len() {
            let dist = squared_euclidean_dist(&data[point], &centroids[centroid]);
            if dist < min_dist {
                min_dist = dist;
                clusters[point] = centroid;
            }
        }
        if min_dist < f64::MAX {
            sse += min_dist;
        }
    }
    (clusters, sse)
}

fn squared_euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

fn compute_centroids(data: &Vec<Vec<f64>>, clusters: &Vec<usize>, k: usize) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    let feature_len = data.first().unwrap_or(&vec![]).len();
    for cluster in 0..k {
        let mut centroid = vec![0f64; feature_len];
        let mut cnt = 0;
        for point in 0..data.len() {
            if clusters[point] == cluster {
                for feature in 0..feature_len {
                    centroid[feature] += data[point][feature];
                }
                cnt += 1;
            }
        }
        if cnt > 0 {
            for feature in 0..feature_len {
                centroid[feature] /= cnt as f64;
            }
        }
        result.push(centroid);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_euclidean_dist_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        // Basic test case, distance should be 3^2 + 4^2 = 25
        assert_eq!(squared_euclidean_dist(&a, &b), 25.0);
    }

    #[test]
    fn test_squared_euclidean_dist_with_zeros() {
        let a = vec![0.0, 0.0];
        let b = vec![0.0, 0.0];
        // Both vectors are zeros, distance should be 0
        assert_eq!(squared_euclidean_dist(&a, &b), 0.0);
    }

    #[test]
    fn test_squared_euclidean_dist_negative_numbers() {
        let a = vec![-1.0, -2.0];
        let b = vec![-3.0, -4.0];
        // Test case with negative numbers, distance should be (-1+3)^2 + (-2+4)^2 = 2^2 + 2^2 = 8
        assert_eq!(squared_euclidean_dist(&a, &b), 8.0);
    }

    #[test]
    fn test_squared_euclidean_dist_mixed_values() {
        let a = vec![1.0, -1.0];
        let b = vec![-1.0, 1.0];
        // Mixed values, distance should be (1+1)^2 + (-1-1)^2 = 2^2 + 2^2 = 8
        assert_eq!(squared_euclidean_dist(&a, &b), 8.0);
    }

    #[test]
    fn test_compute_clusters_basic() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![5.0, 5.0],
            vec![6.0, 5.5],
        ];
        let centroids = vec![
            vec![1.0, 2.0],
            vec![5.5, 5.25],
        ];

        let (clusters, sse) = compute_clusters(&data, &centroids);

        assert_eq!(clusters, vec![0, 0, 1, 1]);

        // Manual calculation of SSE
        let expected_sse = squared_euclidean_dist(&data[0], &centroids[0]) +
            squared_euclidean_dist(&data[1], &centroids[0]) +
            squared_euclidean_dist(&data[2], &centroids[1]) +
            squared_euclidean_dist(&data[3], &centroids[1]);
        assert!((sse - expected_sse).abs() < 1e-6); // using a small threshold to avoid floating-point precision issues
    }

    #[test]
    fn test_compute_clusters_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        let centroids = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let (clusters, sse) = compute_clusters(&data, &centroids);

        assert!(clusters.is_empty());
        assert_eq!(sse, 0.0);
    }

    #[test]
    fn test_compute_clusters_no_centroids() {
        let data = vec![vec![1.0, 2.0], vec![1.5, 2.5]];
        let centroids: Vec<Vec<f64>> = vec![];

        let (clusters, sse) = compute_clusters(&data, &centroids);

        // Expect default cluster assignment (index 0) and no error computation if centroids are empty
        assert!(clusters.iter().all(|&x| x == 0));
        assert_eq!(sse, 0.0);
    }

    #[test]
    fn test_compute_clusters_data_equals_centroids() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let centroids = data.clone();

        let (clusters, sse) = compute_clusters(&data, &centroids);

        assert_eq!(clusters, vec![0, 1]);
        assert_eq!(sse, 0.0); // Perfect match between data points and centroids should result in 0 SSE
    }

    #[test]
    fn test_compute_centroids_basic() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let clusters = vec![0, 0, 1, 1];
        let k = 2;

        let centroids = compute_centroids(&data, &clusters, k);

        assert_eq!(centroids.len(), k);
        assert_eq!(centroids[0], vec![1.25, 2.25]);
        assert_eq!(centroids[1], vec![4.0, 5.0]);
    }

    #[test]
    fn test_compute_centroids_single_cluster() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let clusters = vec![0, 0, 0, 0]; // All points in one cluster
        let k = 1;

        let centroids = compute_centroids(&data, &clusters, k);

        assert_eq!(centroids.len(), k);
        assert_eq!(centroids[0], vec![2.625, 3.625]); // Mean of all points
    }

    #[test]
    fn test_compute_centroids_empty_cluster() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let clusters = vec![0, 0, 1, 1]; // Two clusters, third cluster is not used
        let k = 3;

        let centroids = compute_centroids(&data, &clusters, k);

        assert_eq!(centroids.len(), k);
        assert_eq!(centroids[2], vec![0.0, 0.0]); // No points, results in a centroid of zero values
    }

    #[test]
    fn test_compute_centroids_no_data() {
        let data: Vec<Vec<f64>> = Vec::new(); // No data
        let clusters = Vec::new();
        let k = 2;

        let centroids = compute_centroids(&data, &clusters, k);

        assert_eq!(centroids.len(), k);
        assert!(centroids.iter().all(|centroid| centroid.is_empty())); // Centroids are empty because there's no data
    }

    #[test]
    fn test_run_k_means_basic() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let initial_centroids = vec![
            vec![1.0, 2.0],
            vec![5.0, 6.0],
        ];
        let max_iters = 10;
        let eps = 0.0001;

        let (centroids, clusters, sse, iters) = run_k_means(&data, &initial_centroids, max_iters, eps);

        assert_eq!(clusters, vec![0, 0, 0, 1]);
        assert_eq!(centroids.len(), 2);
        assert!(iters <= max_iters);
        assert!(sse.is_finite());
    }

    #[test]
    fn test_run_k_means_convergence() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let initial_centroids = vec![
            vec![1.0, 2.0],
            vec![5.0, 6.0],
        ];
        let max_iters = 100;
        let eps = 0.1;  // Larger epsilon for faster convergence

        let (_, _, _, iters) = run_k_means(&data, &initial_centroids, max_iters, eps);

        assert!(iters < max_iters); // Should converge before reaching max_iters
    }
}

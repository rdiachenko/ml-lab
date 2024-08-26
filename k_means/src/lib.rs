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

pub fn cluster(data: &Vec<Vec<f64>>, k: usize, init_strategy: &CentroidInitStrategy) -> (Vec<Vec<f64>>, Vec<usize>, f64, i32) {
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

fn run_k_means(data: &Vec<Vec<f64>>, initial_centroids: &Vec<Vec<f64>>, max_iters: usize, eps: f64) -> (Vec<Vec<f64>>, Vec<usize>, f64, i32) {
    let k = initial_centroids.len();
    let mut centroids = initial_centroids.clone();
    let mut clusters = vec![0; data.len()];
    let mut sse = f64::MAX;
    let mut iters = 0;
    for _i in 0..max_iters {
        let (cur_clusters, cur_sse) = compute_clusters(data, &centroids);
        let converged = (sse - cur_sse) / cur_sse <= eps;
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
        sse += min_dist;
    }
    (clusters, sse)
}

fn squared_euclidean_dist(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mut result = 0f64;
    for i in 0..a.len() {
        result += (a[i] - b[i]).powf(2.0);
    }
    result
}

fn compute_centroids(data: &Vec<Vec<f64>>, clusters: &Vec<usize>, k: usize) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    let feature_len = data.first().unwrap().len();
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
        for feature in 0..feature_len {
            centroid[feature] /= cnt as f64;
        }
        result.push(centroid);
    }
    result
}

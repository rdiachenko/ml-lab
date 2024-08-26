use crate::squared_euclidean_dist;
use rand::Rng;
use std::collections::HashSet;

pub fn init_centroids(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut centroids = HashSet::new();
    let mut rng = rand::thread_rng();
    centroids.insert(rng.gen_range(0..data.len()));
    while centroids.len() < k {
        let mut max_dist = f64::MIN;
        let mut centroid = 0usize;
        for p in 0..data.len() {
            if !centroids.contains(&p) {
                let mut min_dist = f64::MAX;
                let mut min_dist_point = p;
                for c in &centroids {
                    let dist = squared_euclidean_dist(&data[p], &data[*c]);
                    if dist < min_dist {
                        min_dist = dist;
                        min_dist_point = p;
                    }
                }
                if min_dist > max_dist {
                    max_dist = min_dist;
                    centroid = min_dist_point;
                }
            }
        }
        centroids.insert(centroid);
    }
    centroids.iter().map(|c| data[*c].clone()).collect()
}

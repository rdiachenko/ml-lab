use crate::compute_centroids;
use rand::distributions::{Distribution, Uniform};

pub fn init_centroids(data: &Vec<Vec<f64>>, k: usize) -> Vec<Vec<f64>> {
    let range = Uniform::from(0..k);
    let mut rng = rand::thread_rng();
    let mut clusters: Vec<usize> = vec![0; data.len()];
    for point in 0..data.len() {
        clusters[point] = range.sample(&mut rng);
    }
    compute_centroids(data, &clusters, k)
}

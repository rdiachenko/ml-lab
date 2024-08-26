use crate::compute_centroids;
use rand::distributions::{Distribution, Uniform};

pub fn init_centroids(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let range = Uniform::from(0..k);
    let mut rng = rand::thread_rng();

    let clusters: Vec<usize> = data.iter()
        .map(|_| range.sample(&mut rng))
        .collect();

    compute_centroids(data, &clusters, k)
}

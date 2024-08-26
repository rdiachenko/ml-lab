use rand::prelude::SliceRandom;

pub fn init_centroids(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    data.choose_multiple(&mut rng, k).cloned().collect()
}

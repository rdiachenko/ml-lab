use crate::squared_euclidean_dist;
use rand::distributions::Uniform;
use rand::Rng;

pub fn init_centroids(data: &[Vec<f64>], k: usize, samples_per_iter: usize) -> Vec<Vec<f64>> {
    let local_centers_n = if samples_per_iter > 0 {
        samples_per_iter
    } else {
        (2.0 + (k as f64).log2()).floor() as usize
    };

    let mut rng = rand::thread_rng();
    let initial_centroid = rng.gen_range(0..data.len());
    let mut result = vec![];
    result.push(data[initial_centroid].clone());
    let mut min_dists = vec![f64::MAX; data.len()];
    let mut sse = 0f64;

    for p in 0..data.len() {
        min_dists[p] = squared_euclidean_dist(&data[p], &data[initial_centroid]);
        sse += min_dists[p];
    }

    let range = Uniform::new(0.0, 1.0);
    for _ in 1..k {
        let dists_cumsum: Vec<f64> = min_dists
            .iter()
            .scan(0.0, |acc, &dist| {
                *acc += dist;
                Some(*acc)
            })
            .collect();
        let centroid_candidates: Vec<usize> = (0..local_centers_n)
            .map(|_| {
                let rand_val = rng.sample(range) * sse;
                dists_cumsum
                    .iter()
                    .position(|&dist| dist > rand_val)
                    .unwrap_or(0)
            })
            .collect();

        let mut dist_to_candidates = Vec::new();
        for candidate_index in 0..centroid_candidates.len() {
            dist_to_candidates.push(Vec::new());
            let c = centroid_candidates[candidate_index];
            for p in 0..data.len() {
                let dist = squared_euclidean_dist(&data[p], &data[c]);
                dist_to_candidates[candidate_index].push(dist);
            }
        }

        let mut best_centroid_candidate: usize = 0;
        let mut best_sse = f64::MAX;
        let mut best_min_dists = vec![];

        for candidate_index in 0..centroid_candidates.len() {
            let mut new_min_dists = vec![];
            let mut new_sse = 0.0;
            for (dist_index, min_dist) in min_dists.iter().enumerate() {
                let dist = f64::min(*min_dist, dist_to_candidates[candidate_index][dist_index]);
                new_min_dists.push(dist);
                new_sse += dist;
            }
            if new_sse < best_sse {
                best_sse = new_sse;
                best_min_dists = new_min_dists;
                best_centroid_candidate = centroid_candidates[candidate_index];
            }
        }
        result.push(data[best_centroid_candidate].clone());
        sse = best_sse;
        min_dists = best_min_dists;
    }

    result
}

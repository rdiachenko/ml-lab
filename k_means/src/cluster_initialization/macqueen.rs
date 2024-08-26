use rand::prelude::SliceRandom;

pub fn init_centroids(data: &Vec<Vec<f64>>, k: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut points: Vec<usize> = (0..data.len()).collect();
    points.shuffle(&mut rng);
    let mut result = Vec::new();
    for index in 0..k {
        let centroid = points[index];
        result.push(data[centroid].clone());
    }
    result
}

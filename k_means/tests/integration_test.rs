use k_means::CentroidInitStrategy::*;
use rand::Rng;

#[test]
fn it_clusters_with_forgy() {
    let k = 1;
    let (data, min, max) = data();

    let (centroids, _, sse, iters) = k_means::cluster(&data, k, &Forgy);

    assert(&centroids, min, max, k);
    assert!(sse.is_finite());
    assert!(iters < 10);
}

#[test]
fn it_clusters_with_macqueen() {
    let k = 5;
    let (data, min, max) = data();

    let (centroids, _, sse, iters) = k_means::cluster(&data, k, &MacQueen);

    assert(&centroids, min, max, k);
    assert!(sse.is_finite());
    assert!(iters < 10);
}

#[test]
fn it_clusters_with_maximin() {
    let k = 5;
    let (data, min, max) = data();

    let (centroids, _, sse, iters) = k_means::cluster(&data, k, &Maximin);

    assert(&centroids, min, max, k);
    assert!(sse.is_finite());
    assert!(iters < 10);
}

#[test]
fn it_clusters_with_bradley_fayyad() {
    let k = 1;
    let (data, min, max) = data();

    let (centroids, _, sse, iters) = k_means::cluster(&data, k, &BradleyFayyad);

    assert(&centroids, min, max, k);
    assert!(sse.is_finite());
    assert!(iters < 10);
}

#[test]
fn it_clusters_with_kmeanspp() {
    let k = 10;
    let (data, min, max) = data();

    let (centroids, _, sse, iters) = k_means::cluster(&data, k, &KmeansPP);

    assert(&centroids, min, max, k);
    assert!(sse.is_finite());
    assert!(iters <= 100);
}

#[test]
fn it_clusters_with_greedy_kmeanspp() {
    let k = 10;
    let (data, min, max) = data();

    let (centroids, _, sse, iters) = k_means::cluster(&data, k, &GreedyKmeansPP);

    assert(&centroids, min, max, k);
    assert!(sse.is_finite());
    assert!(iters <= 100);
}

fn assert(centroids: &[Vec<f64>], min: f64, max: f64, k: usize) {
    assert_eq!(centroids.len(), k);
    assert!(
        centroids
            .iter()
            .all(|centroid| centroid.iter().all(|&x| x >= min && x <= max)),
        "Some centroids are out of expected range [{}; {}]: {:?}",
        min,
        max,
        centroids
    );
}

fn data() -> (Vec<Vec<f64>>, f64, f64) {
    let min = -100.0;
    let max = 100.0;
    let rows = 10;
    let columns = 5;
    let mut rng = rand::thread_rng();
    let data = (0..rows)
        .map(|_| (0..columns).map(|_| rng.gen_range(min..=max)).collect())
        .collect();

    (data, min, max)
}

pub mod bradley_fayyad;
pub mod forgy;
pub mod kmeanspp;
pub mod macqueen;
pub mod maximin;

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

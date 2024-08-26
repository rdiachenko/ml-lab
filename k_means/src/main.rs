use image::{DynamicImage, GenericImageView, Rgb};
use k_means::CentroidInitStrategy;
use k_means::CentroidInitStrategy::*;
use std::env;
use std::str::FromStr;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: {} IMAGE K STRATEGY", args[0]);
        eprintln!("Example: {} sky.png 16 m", args[0]);
        std::process::exit(1);
    }

    let image_name = &args[1];
    let k = usize::from_str(&args[2]).expect("error parsing number of clusters");
    let strategy = parse_strategy(&args[3]).expect("error parsing centroid initial strategy");

    let original_img = image::open(image_name).unwrap();
    let img_data = transform(&original_img);

    let (centroids, clusters, sse, iters) = k_means::cluster(&img_data, k, &strategy);
    println!("Converged in {} iterations, sse={}", iters, sse);
    let compressed_img = compress(&original_img, &centroids, &clusters);
    compressed_img
        .save(format!("{:?}-compressed-{}", strategy, image_name))
        .unwrap()
}

fn parse_strategy(s: &str) -> Option<CentroidInitStrategy> {
    match s {
        "F" | "f" => Some(Forgy),
        "M" | "m" => Some(MacQueen),
        "X" | "x" => Some(Maximin),
        "B" | "b" => Some(BradleyFayyad),
        "K" | "k" => Some(KmeansPP),
        "G" | "g" => Some(GreedyKmeansPP),
        _ => None,
    }
}

fn transform(image: &DynamicImage) -> Vec<Vec<f64>> {
    image
        .pixels()
        .map(|pixel| pixel.2 .0)
        .map(|rgba| {
            let mut rgb = Vec::new();
            for val in rgba.iter().take(3) {
                rgb.push(normalize(*val));
            }
            rgb
        })
        .collect()
}

fn normalize(val: u8) -> f64 {
    val as f64 / 255.0
}

fn denormalize(val: f64) -> u8 {
    (val * 255.0) as u8
}

fn compress(
    image: &DynamicImage,
    centroids: &[Vec<f64>],
    clusters: &[usize],
) -> image::ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut result = image::ImageBuffer::new(image.width(), image.height());
    for (j, i, pixel) in result.enumerate_pixels_mut() {
        let point: usize = (i * image.width() + j) as usize;
        let centroid = &centroids[clusters[point]];
        let r = denormalize(centroid[0]);
        let g = denormalize(centroid[1]);
        let b = denormalize(centroid[2]);
        *pixel = Rgb([r, g, b]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba, RgbaImage};
    use std::collections::HashSet;

    #[test]
    fn test_image_compression() {
        let width = 10;
        let height = 10;
        let k = 5; // Number of desired clusters/colors
        let strategy = GreedyKmeansPP;

        // Create an image with more than 'k' distinct colors
        let mut img: RgbaImage = ImageBuffer::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let r = (x * 47 % 255) as u8;
            let g = (y * 47 % 255) as u8;
            let b = (x * y % 255) as u8;
            *pixel = Rgba([r, g, b, 0]);
        }

        let original_img = DynamicImage::ImageRgba8(img.clone());
        let img_data = transform(&original_img);
        let (centroids, clusters, _, _) = k_means::cluster(&img_data, k, &strategy);
        let compressed_img = compress(&original_img, &centroids, &clusters);

        assert_eq!(compressed_img.width(), width);
        assert_eq!(compressed_img.height(), height);

        // Check the number of unique colors in the compressed image
        let unique_colors: HashSet<_> = compressed_img.pixels().map(|p| p.0).collect();
        assert_eq!(unique_colors.len(), k);
    }
}

use ndarray;
use ndarray::{Array2, arr2};
use nalgebra as na;
use na::DMatrix;

fn main() {
    let a: Array2<f64> = arr2(&[[1., 2.], [3., 4.]]);
    let b: Array2<f64> = arr2(&[[5., 6.], [7., 8.]]);
    let c = a.dot(&b);  // Matrix multiplication
    println!("Matrix product:\n{}", c);

    let a = DMatrix::<f64>::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let svd = a.svd(true, true);  // Perform SVD
    println!("Singular values: {:?}", svd.singular_values);
}
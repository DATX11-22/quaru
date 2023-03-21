use ndarray::{array, Array2};
use num::Complex;

/// Given a boolean value, returns the qubit vector representation of
/// that value.
///
/// # Example
/// - `&false` will give the complex version of `[[1],[0]]`
/// - `&true` will give the complex version of `[[0],[1]]`
pub fn to_qbit_vector(bit: &bool) -> Array2<Complex<f64>> {
    match bit {
        true => array![[Complex::new(0.0, 0.0)], [Complex::new(1.0, 0.0)]],
        false => array![[Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0)]],
    }
}

/// Given an 2-dimensional array consisting of floats, returns the 
/// complex version.
pub fn real_to_complex(matrix: Array2<f64>) -> Array2<Complex<f64>> {
    matrix.map(|e| e.into())
}


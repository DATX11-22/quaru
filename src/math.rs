use ndarray::{array, Array2};
use num::Complex;

/// Given a boolean value, returns the qubit vector representation of
/// that value.
///
/// # Example
/// - `&false` will give the complex version of `[[1],[0]]`
/// - `&true` will give the complex version of `[[0],[1]]`
pub fn to_qbit_vector(bit: &bool) -> Array2<c64> {
    match bit {
        true => array![[Complex::new(0.0, 0.0)], [Complex::new(1.0, 0.0)]],
        false => array![[Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0)]],
    }
}

/// Given an 2-dimensional array consisting of floats, returns the
/// complex version.
pub fn real_arr_to_complex(matrix: Array2<f64>) -> Array2<c64> {
    matrix.map(|e| real_to_complex(*e))
}

/// Given a real number `n` returns a complex number with real part `n` and
/// imaginary part 0.
pub fn real_to_complex(n: f64) -> c64 {
    Complex::new(n, 0.0)
}

/// Given a float `re` and `im` returns a complex number with real part `re` and imaginary part
/// `im`.
pub fn new_complex(re: f64, im: f64) -> c64 {
    Complex::new(re, im)
}

/// A 64-bit complex float.
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;
/// A 32-bit complex float.
#[allow(non_camel_case_types)]
pub type c32 = Complex<f64>;

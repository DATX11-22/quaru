use ndarray::{array, Array2};
use num::Complex;

pub fn to_qbit_vector(bit: &bool) -> Array2<Complex<f64>> {
    match bit {
        true => array![[Complex::new(0.0, 0.0)], [Complex::new(1.0, 0.0)]],
        false => array![[Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0)]],
    }
}

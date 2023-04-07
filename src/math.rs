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

/// Given state vectors a and b, check if they are equal, with tolerance.
pub fn equal_qubits(a: Array2<Complex<f64>>, b: Array2<Complex<f64>>) -> bool {
    let mut equal = true;
    for (i, s) in a.iter().enumerate() {
        if (s - b[(i, 0)]).norm() >= 1e-8 {
            equal = false;
        }
    }
    equal
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

/// Given fraction m/n and a positive integer l, returns integers r and s such that
/// r/s is the closest fraction to m/n with denominator bounded by l.
/// Uses the continued fraction algorithm.
/// Adapted from python implementation in https://github.com/python/cpython/issues/95723
pub fn limit_denominator(m: u32, n: u32, l: u32) -> (u32, u32) {
    let (mut a, mut b, mut p, mut q, mut r, mut s, mut v) = (n, m % n, 1, 0, m / n, 1, 1);
    while 0 < b && q + a / b * s <= l {
        (a, b, p, q, r, s, v) = (b, a % b, r, s, p + a / b * r, q + a / b * s, -v);
    }
    let (t, u) = (p + (l - q) / s * r, q + (l - q) / s * s);
    if 2 * b * u <= n {
        (r, s)
    } else {
        (t, u)
    }
}

pub fn modpow(mut base: u32, mut exponent: u32, modulus: u32) -> u32 {
    let mut result = 1;
    while exponent > 0 {
        if exponent % 2 == 1 {
            result = (result * base) % modulus;
        }
        exponent /= 2;
        base = base * base % modulus;
    }
    return result;
}

/// A 64-bit complex float.
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;
/// A 32-bit complex float.
#[allow(non_camel_case_types)]
pub type c32 = Complex<f64>;

#[cfg(test)]
mod tests {
    use num::abs;

    #[test]
    fn limit_denominator_working() {
        let mx = 30;
        for m in 0..mx {
            for n in 1..mx {
                for l in 1..mx {
                    let (r, s) = super::limit_denominator(m, n, l);
                    if m == 0 {
                        assert!(r == 0 && s == 1);
                        continue;
                    }
                    for r2 in 1..mx {
                        for s2 in 1..=l {
                            assert!(
                                abs(r2 as f64 / s2 as f64 - m as f64 / n as f64) + 1e-15
                                    >= abs(r as f64 / s as f64 - m as f64 / n as f64)
                            );
                        }
                    }
                }
            }
        }
    }
}

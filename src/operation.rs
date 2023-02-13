use ndarray::Array2;
use num::Complex;

pub struct Operation<const ARITY: usize> {
}

impl<const ARITY: usize> Operation<ARITY> {
    pub fn matrix(&self) -> Array2<Complex<f64>> {
        unimplemented!()
    }

    pub fn targets(&self) -> [usize; ARITY] {
        unimplemented!()
    }
}

pub fn identity(_target: usize) -> Operation<1> {
    unimplemented!()
}

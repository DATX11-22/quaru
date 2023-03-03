use gcd::Gcd;
use ndarray::array;
use num::Complex;
use quant::{operation, register::Register};

use ndarray::{Array2, ArrayBase, Data, DataMut};

fn main() {
    let reg = Register::new(&[false; 4]);
    let N: u32 = 15;
    let d1 = shores(N);
    let d2 = N / d1;
    println!("The factors of {} are {} and {}", N, d1, d2);

    // let qft = operation::qft(4);
    // println!("{:}", qft.matrix)
}

fn shores(N: u32) -> u32 {
    if N % 2 == 0 {
        // N is even
        return 2;
    }
    loop {
        let mut a: u32 = rand::random();
        a = a % N;
        a = if a <= 1 { 2 } else { a };
        // a = 7;
        let K = gcd::euclid_u32(a, N);
        if K != 1 && K != N {
            return K;
        }

        println!("a = {}", a);
        let r = find_r(N, a);
        if r % 2 == 0 {
            // r is even so we can find a factor
            return gcd::euclid_u32(a.pow(r / 2) + 1, N)
        }
    }
}

//TODO: implement this function
fn find_r(N: u32, a: u32) -> u32 {
    if a == 7 && N == 15 {
        return 4;
    }
    //calculate f(x) as a truth function
    1
}


#[cfg(feature = "blas")]
extern crate blas_src;
// --target x86_64-pc-windows-msvc


use clap::Parser;

use ndarray::{Array2};
use num::Complex;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of times to run multiplication
    #[arg(short, long, default_value_t = 1)]
    times: u32,

    #[arg(short, long, default_value_t = 1)]
    size: usize,


}


fn main(){
    let args = Args::parse();
    let mut tot_time = 0;

    #[cfg(feature = "blas")]
    println!("Using blas");

    for _ in 0..args.times {
        let a = create_matrix(args.size);
        let b = create_matrix(args.size);
        let sw = stopwatch::Stopwatch::start_new();
        let _c = a.dot(&b);
        tot_time += sw.elapsed_ms();
    }
    let avg = tot_time / (args.times as i64);
    print!("Ran {} times, average time: {} ms", args.times, avg);
    // without blas 1024x1024: avg: 717ms
}
fn create_matrix(size: usize) -> Array2<Complex<f64>> {
    let mut matrix = Array2::<Complex<f64>>::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            matrix[[i, j]] = Complex::new(rand::random::<f64>(), rand::random::<f64>());
        }
    }
    return matrix;

}
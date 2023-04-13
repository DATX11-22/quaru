use clap::Parser;

use ndarray::{array, linalg, Array2, ArrayBase, Dim, OwnedRepr};
use num::Zero;
use quaru::math::{c64, equal_qubits, to_qbit_vector};
use quaru::{operation, register::Register};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // We will teleport the qubit [a+bi, c+di].
    #[arg(short, long, default_value_t = 0.8)]
    a: f64,

    #[arg(short, long, default_value_t = 0.0)]
    b: f64,

    #[arg(short, long, default_value_t = 0.0)]
    c: f64,

    #[arg(short, long, default_value_t = -0.6)]
    d: f64,
}

fn main() {
    let args = Args::parse();

    let q0 = array![[c64::new(args.a, args.b)], [c64::new(args.c, args.d)]];
    let (is_equal, result) = test_quantum_teleportation(q0.clone());
    println!("Expected:\n{}\n", q0);
    println!("Got:\n{:}\n", result);
    if is_equal {
        println!("Correct");
    } else {
        println!("WRONG!");
    }
}

fn test_quantum_teleportation(q0: Array2<c64>) -> (bool, Array2<c64>) {
    // State with three qubits: two zeroes and q0.
    let new_state: Array2<c64> = linalg::kron(
        &linalg::kron(&to_qbit_vector(&false), &to_qbit_vector(&false)),
        &q0.clone(),
    );

    // Create register with state new_state
    let mut reg = Register::new(&[false; 3]);
    reg.state = new_state.clone();

    // Run quantum teleportation algorithm
    quantum_teleportation(&mut reg);

    let result = get_state_of_qubit(reg.state.clone(), 2);
    let is_equal = equal_qubits(result.clone(), q0.clone());
    (is_equal, result)
}

/// Quantum teleportation
/// reg is a 3-qubit register where the second and third qubits are zero.
/// The first qubit will be teleported to the third qubit.
fn quantum_teleportation(reg: &mut Register) {
    reg.apply(&operation::hadamard(2));
    reg.apply(&operation::cnot(2, 1));
    reg.apply(&operation::cnot(0, 1));
    reg.apply(&operation::hadamard(0));

    let c_1 = reg.measure(1);
    if c_1 {
        reg.apply(&operation::not(2));
    }
    let c_0 = reg.measure(0);
    if c_0 {
        reg.apply(&operation::pauli_z(2));
    }
}

/// Get the state of the qubit at position n.
/// Doesn't work if n is entangled.
pub fn get_state_of_qubit(
    state: ArrayBase<OwnedRepr<c64>, Dim<[usize; 2]>>,
    n: usize,
) -> Array2<c64> {
    let mut beta: c64 = c64::zero();
    let mut alpha: c64 = c64::zero();
    for (i, s) in state.iter().enumerate() {
        if (i >> n) & 1 == 1 {
            beta += s;
        } else {
            alpha += s;
        }
    }
    let prob = alpha.norm().powi(2) + beta.norm().powi(2);
    alpha /= prob.sqrt();
    beta /= prob.sqrt();
    array![[alpha], [beta]]
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use quaru::math::c64;
    use rand::Rng;

    #[test]
    fn quantum_teleportation() {
        let mut rng = rand::thread_rng();
        for i in 0..20 {
            // Random complex number with norm <= 1
            let a = c64::from_polar(rng.gen_range(0.0 .. 1.0), rng.gen_range(0.0 .. 2.0 * std::f64::consts::PI));
            // Random complex number such that norm(a)^2 + norm(b)^2 = 1
            let b = c64::from_polar((1.0 - a.norm().powi(2)).sqrt(), rng.gen_range(0.0 .. 2.0 * std::f64::consts::PI));
            
            assert!(((a*a).norm() + (b*b).norm() - 1.0).abs() < 1e-10);

            let q0 = array![[a], [b]];
            let (is_equal, _) = super::test_quantum_teleportation(q0);
            assert!(is_equal);
        }
    }
}

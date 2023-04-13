use clap::Parser;

use quaru::{operation, register::Register};

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    #[arg(short, long, default_value = "false", action = clap::ArgAction::Set)]
    a: bool,

    #[arg(short, long, default_value = "true", action = clap::ArgAction::Set)]
    b: bool,
}

fn main() {
    let args = Args::parse();

    let mut register = superdense_preparation();

    println!("Probabilities after preparation:");
    register.print_probabilities();
    println!();

    register = superdense_encode(register, args.a, args.b);

    println!("Probabilities after encoding:");
    register.print_probabilities();
    println!();

    let (a, b) = superdense_decode(register);

    println!("Measured result: {} {}", a, b);

    if a == args.a && b == args.b {
        println!("Correct");
    } else {
        println!("WRONG!");
    }
}

/// Prepare the qubits needed for superdense coding.
/// If Alice wants to send two bits to Bob, the first qubit
/// in register is given to Alice and the second to Bob.
fn superdense_preparation() -> Register {
    let mut register = Register::new(&[false; 2]);

    // Entangle qubit 0 and 1
    register.apply(&operation::hadamard(0));
    register.apply(&operation::cnot(0, 1));

    register
}

/// Alice sends the bits m0 and m1 to Bob by applying this
/// function to her qubit (target=0 in all gates) and then
/// sending her qubit to Bob.
fn superdense_encode(mut register: Register, m0: bool, m1: bool) -> Register {
    let message = [m0, m1];

    if message == [false, false] {
        register.apply(&operation::identity(0));
    } else if message == [false, true] {
        register.apply(&operation::not(0));
    } else if message == [true, false] {
        register.apply(&operation::pauli_z(0));
    } else if message == [true, true] {
        register.apply(&operation::pauli_z(0));
        register.apply(&operation::not(0));
    }

    register
}

/// After receiving Alice's qubit, Bob applies this function
/// to get Alice's message (m0, m1).
fn superdense_decode(mut register: Register) -> (bool, bool) {
    register.apply(&operation::cnot(0, 1));
    register.apply(&operation::hadamard(0));

    (register.measure(0), register.measure(1))
}

#[cfg(test)]
mod tests {
    #[test]
    fn superdense_coding() {
        for a in [false, true] {
            for b in [false, true] {
                let mut register = super::superdense_preparation();
                register = super::superdense_encode(register, a, b);
                let (a2, b2) = super::superdense_decode(register);
                assert_eq!(a, a2);
                assert_eq!(b, b2);
            }
        }
    }
}

use quaru::openqasm;
use std::path::Path;

fn main() {
    let example_dir = Path::new(file!()).parent().unwrap();
    let qasm_path = example_dir.join(Path::new("openqasm_example.qasm"));

    let registers = openqasm::run_openqasm(&qasm_path).ok().unwrap();

    for (name, qreg) in registers.qregs {
        println!("\nQuatnum register: {}", name);
        qreg.print_probabilities();
        println!();
    }
}

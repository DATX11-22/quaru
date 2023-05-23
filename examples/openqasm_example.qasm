OPENQASM 2.0;

gate u2(phi, lambda) q {
    U(pi / 2, phi, lambda) q;
}

gate h a {
    u2(0, pi) a;
}

gate cnot c, t {
    CX c, t;
}


qreg q[2];

h q[0];
cnot q[0], q[1];

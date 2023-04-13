OPENQASM 2.0;

gate u2(phi, lambda) q {
    U(pi / 2, phi, lambda) q;
}

gate h a {
    u2(0, pi) a;
}

gate cx c, t {
    CX c, t;
}

qreg q[3];

h q[0];
cx q[0], q[1];

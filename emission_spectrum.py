#!/usr/bin/python3
import math, sys

def read_emission_table(path_file):
    E, R = [], []
    with open(path_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:  # skip header
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    E.append(float(parts[0]))  # DE/eV
                    R.append(float(parts[2]))  # diff_rate
                except ValueError:
                    pass
    assert E, "No rows parsed from emission-rate.dat"
    return E, R

def gauss_norm(x, mu, delta):
    coeff = 1.0 / (delta * math.sqrt(math.pi/2.0))
    return coeff * math.exp(-2.0 * ((x - mu)/delta)**2)

def emit_spectrum(E, R, delta=0.06, eps=0.002, kappa=3.0):
    Emin = min(E) - kappa*delta
    Emax = max(E) + kappa*delta
    grid, I = [], []
    x = Emin
    n = len(E)
    while x <= Emax + 1e-12:
        s = 0.0
        for Ei, Ri in zip(E, R):
            s += Ri * gauss_norm(x, Ei, delta)
        s /= n
        grid.append(x)
        I.append(s)
        x += eps
    # normalize to 1
    m = max(I) if max(I) > 0 else 1.0
    I = [v/m for v in I]
    return grid, I

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python emission_spectrum.py emission-rate.dat")
        sys.exit(1)
    E, R = read_emission_table(sys.argv[1])
    grid, I = emit_spectrum(E, R, delta=0.06, eps=0.002, kappa=3.0)
    with open("emission_spectrum_eV.dat", "w") as fo:
        for x, y in zip(grid, I):
            fo.write(f"{x:.6f} {y:.8e}\n")
    print("Done: emission_spectrum_eV.dat")


#!/usr/bin/env python3
import numpy as np, argparse, math

def read_emission_table(path):
    E, R = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s: continue
            parts = s.split()
            if i == 0 and ("DE/eV" in s or "lambda" in s):  # 跳过表头
                continue
            if len(parts) < 3: continue
            try:
                E.append(float(parts[0]))   # DE/eV
                R.append(float(parts[2]))   # diff_rate
            except ValueError:
                pass
    if not E:
        raise SystemExit("未解析到数据，请检查 emission-rate.dat 的列顺序/空行。")
    return np.array(E), np.array(R)

def ev_to_nm(E):
    h_evs, c = 4.13566733e-15, 299792458
    with np.errstate(divide='ignore'):
        return (1e9 * h_evs * c) / E

def gauss_matrix(grid, centers, delta):
    coeff = 1.0 / (delta * math.sqrt(math.pi/2.0))  # 与你原脚本一致的归一化
    d = (grid[:, None] - centers[None, :]) / delta
    return coeff * np.exp(-2.0 * (d ** 2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="emission-rate.dat")
    ap.add_argument("--delta", type=float, default=0.06, help="高斯展宽(eV)")
    ap.add_argument("--eps",   type=float, default=0.002, help="能量步长(eV)")
    ap.add_argument("--kappa", type=float, default=3.0, help="边界外扩(×delta)")
    ap.add_argument("--no-smooth", action="store_true", help="不卷积，直接用原网格")
    ap.add_argument("--no-norm",   action="store_true", help="不归一化最大值=1")
    args = ap.parse_args()

    E, R = read_emission_table(args.infile)

    if args.no_smooth or args.delta <= 0:
        grid, I = E.copy(), R.copy()
    else:
        Emin = E.min() - args.kappa * args.delta
        Emax = E.max() + args.kappa * args.delta
        grid = np.arange(Emin, Emax + args.eps/2, args.eps)
        K = gauss_matrix(grid, E, args.delta)         # (Ng, N)
        I = (K @ R) / len(E)                          # 可比性：按样本数归一

    if not args.no_norm:
        m = I.max()
        if m > 0: I = I / m

    np.savetxt("emission_spectrum_eV.dat",
               np.column_stack([grid, I]), fmt="%.6f %.8e")

    lam = ev_to_nm(grid)
    with open("emission_spectrum_full.dat", "w") as fo:
        fo.write("DE/eV    lambda/nm    intensity    +/-error\n")
        for e, l, y in zip(grid, lam, I):
            fo.write(f"{e:8.4f}   {l:10.4E}   {y:12.8E}   0.00000\n")

    print("写出: emission_spectrum_eV.dat, emission_spectrum_full.dat")

if __name__ == "__main__":
    main()


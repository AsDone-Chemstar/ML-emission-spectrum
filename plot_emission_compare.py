#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, numpy as np
import matplotlib
matplotlib.use("Agg")  # 无图形界面时保存到文件
import matplotlib.pyplot as plt

def load_spectrum_eV(path, skip_header=False):
    """读取两列能量域谱: E(eV) I"""
    arr = np.loadtxt(path, skiprows=1 if skip_header else 0)
    E, I = arr[:,0], arr[:,1]
    return E, I

def load_rate_table(path):
    """读取 emission-rate.dat：第1列E、第3列diff_rate"""
    arr = np.loadtxt(path, skiprows=1)
    E, R = arr[:,0], arr[:,2]
    return E, R

def gaussian_broaden(E_centers, weights, delta, Emin=None, Emax=None, eps=0.002, kappa=3.0):
    """与你脚本一致的高斯：coeff=1/(delta*sqrt(pi/2)), exp=-2*((x-mu)/delta)^2"""
    if Emin is None: Emin = E_centers.min() - kappa*delta
    if Emax is None: Emax = E_centers.max() + kappa*delta
    grid = np.arange(Emin, Emax + eps/2.0, eps)
    coeff = 1.0 / (delta * np.sqrt(np.pi/2.0))
    d = (grid[:,None] - E_centers[None,:]) / delta
    I = (coeff * np.exp(-2.0 * d**2) @ weights) / max(len(E_centers), 1)
    return grid, I

def nm_from_e(E):
    with np.errstate(divide='ignore'):
        return 1239.84193 / E

def main():
    ap = argparse.ArgumentParser(description="对比 ML 发射谱与参考谱（能量/波长域）")
    ap.add_argument("--ml-ev", required=True, help="ML 能量域谱，例如 train/.../emission_spectrum_eV.dat")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ref-ev", help="参考 能量域谱（两列 E I），如 emission_spectrum_ref_eV.dat")
    group.add_argument("--ref-rate", help="参考 emission-rate.dat（第1列E、第3列diff_rate）")
    ap.add_argument("--delta", type=float, default=0.06, help="卷积展宽eV（当 --ref-rate 时生效）")
    ap.add_argument("--eps",   type=float, default=0.002, help="能量步长eV（当 --ref-rate 时生效）")
    args = ap.parse_args()

    # 读 ML 能量域谱
    E_ml, I_ml = load_spectrum_eV(args.ml_ev, skip_header=False)
    I_ml = I_ml / (I_ml.max() if I_ml.max()>0 else 1.0)

    # 参考谱：直接读 or 从 emission-rate 卷积
    if args.ref_ev:
        E_ref, I_ref = load_spectrum_eV(args.ref_ev, skip_header=False)
    else:
        E_rate, R_rate = load_rate_table(args.ref_rate)
        E_ref, I_ref = gaussian_broaden(E_rate, R_rate, delta=args.delta, eps=args.eps)
    I_ref = I_ref / (I_ref.max() if I_ref.max()>0 else 1.0)

    # 画能量域
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Normalized intensity")
    plt.plot(E_ml,  I_ml,  label="ML")
    plt.plot(E_ref, I_ref, label="REF")
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_eV.png", dpi=180)

    # 画波长域（各自单独转换）
    lam_ml  = nm_from_e(E_ml)
    lam_ref = nm_from_e(E_ref)
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized intensity")
    plt.plot(lam_ml,  I_ml,  label="ML")
    plt.plot(lam_ref, I_ref, label="REF")
    # 如需红在右、蓝在左：反转x轴
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_nm.png", dpi=180)

    print("Wrote: compare_eV.png, compare_nm.png")

if __name__ == "__main__":
    main()


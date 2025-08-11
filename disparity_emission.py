#!/usr/bin/env python3
import argparse, numpy as np, math, re
from typing import Tuple, Optional

def _try_float(x):
    try:
        return float(x)
    except:
        return None

def load_spectrum(path: str, cols: Optional[Tuple[int,int]]=None, trim_zeros=True):
    """
    读取光谱文件。支持两种常见格式：
      1) 两列:  E   I
      2) 四列:  E   lambda   I(or sigma)   err
    参数:
      cols: (e_col, i_col)  可手动指定列索引；不指定时自动猜测。
      trim_zeros: 去掉首尾强度全为0的区段
    返回:
      E(sorted), I(sorted)
    """
    Es, Is = [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or re.search(r"[A-Za-z]", s):  # 跳过表头/含字母的行
                # 但有些数值格式里带E+02，仍是数字，不会被这行过滤
                # 若你表头没有字母（纯数字标题），可以去掉这一句
                continue
            parts = s.split()
            nums = [_try_float(tok) for tok in parts]
            nums = [x for x in nums if x is not None]
            if len(nums) < 2:
                continue

            if cols is not None:
                e_col, i_col = cols
            else:
                # 自动猜：>=3列 -> 取第0列能量、第2列强度；否则取第0/1列
                if len(nums) >= 3:
                    e_col, i_col = 0, 2
                else:
                    e_col, i_col = 0, 1

            if e_col >= len(nums) or i_col >= len(nums):
                continue

            E = float(nums[e_col]); I = float(nums[i_col])
            Es.append(E); Is.append(I)

    if not Es:
        raise SystemExit(f"[ERROR] 解析失败：{path}")

    E = np.asarray(Es, dtype=float)
    I = np.asarray(Is, dtype=float)

    # 按能量排序
    idx = np.argsort(E)
    E, I = E[idx], I[idx]

    # 去除首尾全0
    if trim_zeros:
        nz = np.nonzero(I)[0]
        if nz.size > 0:
            E = E[nz[0]:nz[-1]+1]
            I = I[nz[0]:nz[-1]+1]

    return E, I

def resample_overlap(E1, I1, E2, I2, eps: Optional[float]=None):
    """
    在重叠能量区间上插值到公共网格。
    eps: 网格步长；None时取两个网格中位间距的较小值。
    返回: grid, I1g, I2g
    """
    left  = max(E1.min(), E2.min())
    right = min(E1.max(), E2.max())
    if right <= left:
        raise SystemExit("[ERROR] 两个光谱没有重叠能量区间。")

    if eps is None:
        def med_dx(E):
            d = np.diff(E)
            d = d[(d > 0) & np.isfinite(d)]
            return np.median(d) if d.size else (E[-1]-E[0])/max(len(E)-1,1)
        eps = min(med_dx(E1), med_dx(E2))
        eps = float(max(eps, 1e-4))  # 给个下限，防止极小

    grid = np.arange(left, right + eps/2, eps)
    # 线性插值；区间外填0
    I1g = np.interp(grid, E1, I1, left=0.0, right=0.0)
    I2g = np.interp(grid, E2, I2, left=0.0, right=0.0)
    return grid, I1g, I2g, eps

def metrics(grid, A, B, tiny=1e-20):
    """
    计算差异指标：
      - L1 面积归一差: ∫|A-B| dE / (Espan)
      - 相对面积差:    ∫|A-B| dE / ∫A dE
      - RMSE:         sqrt(mean( (A-B)^2 ))
      - 余弦相似度:    <A,B> / (||A|| ||B||)
      - 光谱重叠:      ∫min(A,B) dE / ∫max(A,B) dE
      - 峰位偏移:      argmax(A) 与 argmax(B) 的能量差
    """
    dE = np.diff(grid)
    # 用梯形积分
    def trapz(y):
        return np.trapz(y, grid)

    l1_area = trapz(np.abs(A-B))
    span = grid[-1] - grid[0]
    l1_norm = l1_area / max(span, tiny)

    area_A = trapz(A)
    rel_change = l1_area / max(area_A, tiny)

    rmse = math.sqrt(np.mean((A-B)**2))

    dot = float(np.dot(A, B))
    nA = math.sqrt(float(np.dot(A, A)))
    nB = math.sqrt(float(np.dot(B, B)))
    cos_sim = dot / (max(nA, tiny) * max(nB, tiny))

    overlap = trapz(np.minimum(A, B)) / max(trapz(np.maximum(A, B)), tiny)

    peakA = grid[np.argmax(A)]
    peakB = grid[np.argmax(B)]
    peak_shift = peakB - peakA

    return {
        "L1_norm_area": l1_norm,
        "Rel_change": rel_change,
        "RMSE": rmse,
        "Cosine": cos_sim,
        "Overlap": overlap,
        "PeakShift_eV": peak_shift,
        "E_left": grid[0],
        "E_right": grid[-1],
        "Span": span,
    }

def maybe_normalize(A, mode: str):
    if mode == "none":
        return A
    elif mode == "max":
        m = A.max()
        return A / m if m > 0 else A
    elif mode == "area":
        area = np.trapz(A)
        return A / area if area > 0 else A
    else:
        raise SystemExit(f"[ERROR] 未知归一化方式: {mode}")

def main():
    ap = argparse.ArgumentParser(description="对齐两条光谱并评估差异（能量域）")
    ap.add_argument("file_td", help="参考谱（如 TD/NEA 输出），支持 2 列或 4 列格式")
    ap.add_argument("file_ml", help="对比谱（如 ML 预测），支持 2 列或 4 列格式")
    ap.add_argument("--cols1", type=str, default=None,
                    help="参考谱列索引 e_col,i_col 例如 '0,2'；默认自动猜测")
    ap.add_argument("--cols2", type=str, default=None,
                    help="对比谱列索引 e_col,i_col 例如 '0,1'；默认自动猜测")
    ap.add_argument("--eps", type=float, default=None, help="重采样步长(eV)，默认自动")
    ap.add_argument("--norm", choices=["none","max","area"], default="max",
                    help="计算前的归一化方式：none/max/area（默认 max）")
    ap.add_argument("--out", default="disparity_aligned.dat",
                    help="输出对齐后的谱 (E  I_TD  I_ML  |diff|)")
    args = ap.parse_args()

    cols1 = tuple(int(x) for x in args.cols1.split(",")) if args.cols1 else None
    cols2 = tuple(int(x) for x in args.cols2.split(",")) if args.cols2 else None

    E1, I1 = load_spectrum(args.file_td, cols=cols1, trim_zeros=True)
    E2, I2 = load_spectrum(args.file_ml, cols=cols2, trim_zeros=True)

    grid, A, B, eps = resample_overlap(E1, I1, E2, I2, eps=args.eps)

    # 归一化（推荐先 max 归一）
    A = maybe_normalize(A, args.norm)
    B = maybe_normalize(B, args.norm)

    # 指标
    res = metrics(grid, A, B)

    # 输出对齐数据
    diff = np.abs(A-B)
    np.savetxt(args.out, np.column_stack([grid, A, B, diff]), fmt="%.6f %.8e %.8e %.8e")

    # 打印结果
    print(f"Aligned grid: E ∈ [{res['E_left']:.4f}, {res['E_right']:.4f}] eV, step≈{eps:.4f} eV")
    print(f"L1_norm_area     = {res['L1_norm_area']:.6f}   (∫|Δ| dE / span)")
    print(f"Relative_change  = {res['Rel_change']:.6f}   (相对参考谱面积)")
    print(f"RMSE             = {res['RMSE']:.6e}")
    print(f"Cosine similarity= {res['Cosine']:.6f}")
    print(f"Spectrum overlap = {res['Overlap']:.6f}   (∫min / ∫max)")
    print(f"Peak shift (eV)  = {res['PeakShift_eV']:.6f}")
    print(f"Saved aligned spectra -> {args.out}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
import sys, argparse

def load_indices(path, n_use=None):
    idx = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            idx.append(int(s))
    if n_use is not None:
        idx = idx[:int(n_use)]
    return set(idx)  # 1-based 索引

def load_state_file(path, target="E"):
    # 读取: index  E  f   （可有表头行）
    col = 1 if target.lower().startswith("e") else 2
    y = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            parts = s.split()
            if len(parts) < 3: 
                # 可能是表头
                continue
            try:
                i = int(parts[0])
                y[i] = parts[col]  # 用原字符串写出，避免精度格式问题
            except ValueError:
                # 表头包含字母，跳过
                continue
    return y

def main():
    ap = argparse.ArgumentParser(description="混合真值与ML预测，生成全长y文件")
    ap.add_argument("state_file", help="如 state2.index.E.f（包含 index E f）")
    ap.add_argument("pred_file", help="ML 预测文件（每行一个样本，按 1..N 顺序）")
    ap.add_argument("out", help="输出文件名，例如 y.E.em.dat 或 y.f.em.dat")
    ap.add_argument("--itrain", default="itrain.dat", help="训练索引文件（1-based，每行一个）")
    ap.add_argument("--n_train", type=int, required=True, help="训练样本数，使用 itrain 前 n_train 个")
    ap.add_argument("--target", choices=["E","f"], default="E", help="混合哪一列：E 或 f")
    ap.add_argument("--N", type=int, default=None, help="总样本数；默认取预测文件行数")
    args = ap.parse_args()

    train_idx = load_indices(args.itrain, n_use=args.n_train)
    truth_map = load_state_file(args.state_file, target=args.target)

    with open(args.pred_file) as f:
        ml_data = [ln.strip() for ln in f if ln.strip()]

    N = args.N if args.N is not None else len(ml_data)
    if N != len(ml_data):
        print(f"[warn] 预测行数={len(ml_data)} 与 N={N} 不一致，按较小者输出。")
        N = min(N, len(ml_data))

    with open(args.out, "w") as fo:
        for i in range(1, N+1):  # 1-based
            if i in train_idx and i in truth_map:
                fo.write(str(truth_map[i]) + "\n")
            else:
                # 回退到 ML 预测（pred 文件按 1..N 顺序对应）
                fo.write(ml_data[i-1] + "\n")

if __name__ == "__main__":
    main()


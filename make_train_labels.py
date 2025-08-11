#!/usr/bin/env python3
import argparse

def load_indices(path):
    s = set()
    with open(path) as f:
        for line in f:
            t = line.strip()
            if not t: continue
            try: s.add(int(t))
            except: pass
    return s  # 1-based

def load_state_file(path, target="E"):
    # 期望每行: index  E  f  （可含表头；自动跳过非数值）
    col = 1 if target.lower().startswith("e") else 2
    y = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            try:
                i = int(parts[0])
                y[i] = parts[col]  # 保留字符串格式精度
            except:  # 表头或非数值行
                continue
    return y

def main():
    ap = argparse.ArgumentParser(description="生成全长度 y 文件：训练位=真值，其它=NaN/0")
    ap.add_argument("state_file", help="如 state2.index.E.f（含 index E f）")
    ap.add_argument("itrain", help="训练索引文件（1-based）")
    ap.add_argument("out", help="输出文件名，如 y.E.train.dat")
    ap.add_argument("--target", choices=["E","f"], default="E", help="写出哪一列")
    ap.add_argument("--N", type=int, default=None, help="总样本数；默认取 max(索引, 真值索引)")
    ap.add_argument("--fill", choices=["nan","0"], default="nan", help="非训练位填充值")
    args = ap.parse_args()

    train = load_indices(args.itrain)
    truth = load_state_file(args.state_file, target=args.target)

    if not train:
        raise SystemExit("itrain.dat 为空？")

    N = args.N or max(max(train), max(truth.keys() or [0]))
    fill = "nan" if args.fill == "nan" else "0"

    with open(args.out, "w") as fo:
        for i in range(1, N+1):
            if i in train:
                fo.write(str(truth.get(i, "nan")) + "\n")  # 训练位没找到真值也写 NaN
            else:
                fo.write(fill + "\n")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse

def load_indices(path):
    s = set()
    with open(path) as f:
        for line in f:
            t = line.strip()
            if not t: continue
            try: s.add(int(t))
            except: pass
    return s  # 1-based

def load_state_file(path, target="E"):
    # 期望每行: index  E  f  （可含表头；自动跳过非数值）
    col = 1 if target.lower().startswith("e") else 2
    y = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            try:
                i = int(parts[0])
                y[i] = parts[col]  # 保留字符串格式精度
            except:  # 表头或非数值行
                continue
    return y

def main():
    ap = argparse.ArgumentParser(description="生成全长度 y 文件：训练位=真值，其它=NaN/0")
    ap.add_argument("state_file", help="如 state2.index.E.f（含 index E f）")
    ap.add_argument("itrain", help="训练索引文件（1-based）")
    ap.add_argument("out", help="输出文件名，如 y.E.train.dat")
    ap.add_argument("--target", choices=["E","f"], default="E", help="写出哪一列")
    ap.add_argument("--N", type=int, default=None, help="总样本数；默认取 max(索引, 真值索引)")
    ap.add_argument("--fill", choices=["nan","0"], default="nan", help="非训练位填充值")
    args = ap.parse_args()

    train = load_indices(args.itrain)
    truth = load_state_file(args.state_file, target=args.target)

    if not train:
        raise SystemExit("itrain.dat 为空？")

    N = args.N or max(max(train), max(truth.keys() or [0]))
    fill = "nan" if args.fill == "nan" else "0"

    with open(args.out, "w") as fo:
        for i in range(1, N+1):
            if i in train:
                fo.write(str(truth.get(i, "nan")) + "\n")  # 训练位没找到真值也写 NaN
            else:
                fo.write(fill + "\n")

if __name__ == "__main__":
    main()


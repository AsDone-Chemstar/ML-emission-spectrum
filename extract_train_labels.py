#!/usr/bin/env python3
import argparse

def load_indices(path):
    s=[]
    with open(path) as f:
        for ln in f:
            t=ln.strip()
            if t:
                try: s.append(int(t))
                except: pass
    return s  # 保持原顺序

def load_state(path):
    # 期望每行: index  E  f；遇到表头会跳过
    E,f={},{}
    with open(path) as fobj:
        for ln in fobj:
            p=ln.split()
            if len(p) < 3: continue
            try:
                i=int(p[0]); E[i]=p[1]; f[i]=p[2]
            except:  # 表头或非数值
                continue
    return E,f

def main():
    ap=argparse.ArgumentParser(description="按训练索引导出 E/f 训练子集")
    ap.add_argument("state_file", help="如 state2.index.E.f（含 index E f）")
    ap.add_argument("itrain", help="训练索引(1-based)，每行一个")
    ap.add_argument("--prefix", default="", help="输出前缀（可选），如 'S1_'")
    ap.add_argument("--suffix", default="", help="输出后缀（可选），如 '_train'")
    args=ap.parse_args()

    idx = load_indices(args.itrain)
    E,f = load_state(args.state_file)

    with open(f"{args.prefix}E{args.suffix}.dat","w") as fe, \
         open(f"{args.prefix}f{args.suffix}.dat","w") as ff:
        for i in idx:
            if i in E and i in f:
                fe.write(str(E[i])+"\n")
                ff.write(str(f[i])+"\n")

if __name__=="__main__":
    main()


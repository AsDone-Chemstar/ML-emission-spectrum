#!/bin/bash
set -euo pipefail

root="$(pwd)"
script="$root/script"
data="$root/data"
DELTA=0.06
EPS=0.002

# 计算 RIC 的纯 Python（不需要 numpy）：∫|ML-REF|/∫REF，按重叠区间 & 线性插值
ric() {
python - "$1" "$2" <<'PY'
import sys, bisect
def load(path):
    E=[]; I=[]
    for ln in open(path, 'r', encoding='utf-8', errors='ignore'):
        ln=ln.strip()
        if not ln or ln.startswith('#'): continue
        parts=ln.split()
        try:
            e=float(parts[0]); y=float(parts[1])
        except: 
            continue
        E.append(e); I.append(y)
    z=sorted(zip(E,I))
    if not z: print("nan"); sys.exit(0)
    E,I=map(list, zip(*z))
    return E,I

def interp(x, xp, yp):
    # 线性插值；超出范围返回 0
    if x<xp[0] or x>xp[-1]: return 0.0
    j = bisect.bisect_right(xp, x)
    if j==0: return yp[0]
    if j>=len(xp): return yp[-1]
    x0,x1 = xp[j-1], xp[j]
    y0,y1 = yp[j-1], yp[j]
    if x1==x0: return y0
    t = (x-x0)/(x1-x0)
    return y0*(1-t) + y1*t

def trapz(y, x):
    s=0.0
    for k in range(len(x)-1):
        s += 0.5*(y[k]+y[k+1])*(x[k+1]-x[k])
    return s

refE, refI = load(sys.argv[1])
mlE,  mlI  = load(sys.argv[2])

# 以 ML 的网格为主，在 REF 范围内取重叠区间
Egrid = [e for e in mlE if refE[0] <= e <= refE[-1]]
if len(Egrid) < 2:
    # 回退：用 REF 网格在 ML 范围内
    Egrid = [e for e in refE if mlE[0] <= e <= mlE[-1]]
    if len(Egrid) < 2:
        print("nan"); sys.exit(0)

refY=[interp(e, refE, refI) for e in Egrid]
mlY =[interp(e,  mlE,  mlI)  for e in Egrid]

num = trapz([abs(a-b) for a,b in zip(mlY,refY)], Egrid)
den = trapz(refY, Egrid)
print("nan" if den==0 else f"{num/den:.6e}")
PY
}

: > "$root/ML-RIC.result"

while read -r num; do
  workdir="$root/train/$num/spectrum/emission"
  [ -d "$workdir" ] || { echo "[WARN] skip N=$num: $workdir not found"; continue; }
  cd "$workdir"

  # 1) ML eV 谱（若缺则由 ML 速率卷积）
  if [ ! -f emission_spectrum_eV.dat ]; then
    if [ -f emission-rate-ML.dat ]; then
      python "$script/emission_spectrum.py" emission-rate-ML.dat --delta "$DELTA" --eps "$EPS"
    else
      echo "[WARN] N=$num: no emission_spectrum_eV.dat nor emission-rate-ML.dat"; cd "$root"; continue
    fi
  fi

  # 2) 参考 eV 谱（若缺则从 data/emission-rate.dat 卷积）
  if [ ! -f emission_spectrum_ref_eV.dat ]; then
    python "$script/emission_spectrum.py" "$data/emission-rate.dat" --delta "$DELTA" --eps "$EPS"
    mv emission_spectrum_eV.dat emission_spectrum_ref_eV.dat
  fi

  # 3) 计算 RIC（绝对强度对比，不归一化）
  val=$(ric emission_spectrum_ref_eV.dat emission_spectrum_eV.dat)
  echo "$val" >> "$root/ML-RIC.result"
  echo "[OK] N=$num -> RIC=$val"

  cd "$root"
done < "$root/train_nums"


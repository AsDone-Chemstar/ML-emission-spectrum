#!/bin/bash
set -euo pipefail

root="$(pwd)"
script="$root/script"
data="$root/data"
DELTA=0.06
EPS=0.002

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
    if x<xp[0] or x>xp[-1]: return 0.0
    import bisect
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
subE, subI = load(sys.argv[2])

Egrid = [e for e in subE if refE[0] <= e <= refE[-1]]
if len(Egrid) < 2:
    Egrid = [e for e in refE if subE[0] <= e <= subE[-1]]
    if len(Egrid) < 2:
        print("nan"); sys.exit(0)

refY=[interp(e, refE, refI) for e in Egrid]
subY=[interp(e, subE, subI) for e in Egrid]

num = trapz([abs(a-b) for a,b in zip(subY,refY)], Egrid)
den = trapz(refY, Egrid)
print("nan" if den==0 else f"{num/den:.6e}")
PY
}

# 准备一次全参考 eV 谱（根目录）
if [ ! -f "$root/emission_spectrum_ref_eV.dat" ]; then
  echo "[INFO] Building full reference from data/emission-rate.dat ..."
  python "$script/emission_spectrum.py" "$data/emission-rate.dat" --delta "$DELTA" --eps "$EPS"
  mv emission_spectrum_eV.dat "$root/emission_spectrum_ref_eV.dat"
fi

: > "$root/TD-RIC.result"

while read -r num; do
  itrain="$root/train/$num/itrain.dat"
  workdir="$root/train/$num/spectrum/emission"
  [ -f "$itrain" ] || { echo "[WARN] skip N=$num: $itrain not found"; continue; }
  mkdir -p "$workdir"
  cd "$workdir"

  # 抽取 itrain 指定的子集（保留表头）
  awk 'NR==FNR{idx[$1]=1; next}
       FNR==1{print; next}
       idx[FNR-1]{print}' "$itrain" "$data/emission-rate.dat" > "emission-rate-REF-${num}.dat"

  # 卷积子集参考成 eV 两列
  python "$script/emission_spectrum.py" "emission-rate-REF-${num}.dat" --delta "$DELTA" --eps "$EPS"
  mv emission_spectrum_eV.dat "emission_spectrum_ref_${num}_eV.dat"

  # 计算 RIC（全参考 vs 子集参考，绝对强度）
  val=$(ric "$root/emission_spectrum_ref_eV.dat" "emission_spectrum_ref_${num}_eV.dat")
  echo "$val" >> "$root/TD-RIC.result"
  echo "[OK] N=$num -> RIC=$val"

  cd "$root"
done < "$root/train_nums"


#!/bin/bash
# 专用于荧光（S1→S0，state2）训练 + 出谱（修正了 train_nums 路径）

# --- 安全选项（先不开 -u，避免 profile 里引用未定义 PS1 报错）---
set -eo pipefail
export PS1=${PS1:-}
[ -f ~/.bash_profile ] && { set +u; source ~/.bash_profile 2>/dev/null || true; }

# 尝试激活 conda 的 mlatom 环境
if ! python -c "import mlatom" >/dev/null 2>&1; then
  for f in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$f" ] && { set +u; source "$f"; set -u; break; }
  done
  set +u; conda activate mlatom 2>/dev/null || true; set -u
fi

# 现在再开启 nounset
set -u
shopt -s expand_aliases

# ===== 用户可改参数 =====
STATE_NUM=1
EM_STATE=2
DELTA=0.06
EPS=0.002
WEIGHT="E3"
# ======================

main_wd=$(pwd)
script="$main_wd/script"
data="$main_wd/data"
TRAIN_FILE="$main_wd/train_nums"   # <<< 关键修正

# 找 MLatom 启动器
MLCMD=""
if command -v mlatom >/dev/null 2>&1; then
  MLCMD="mlatom"
elif command -v MLatom.py >/dev/null 2>&1; then
  MLCMD="MLatom.py"
else
  SCRIPTS_DIR=$(python - <<'PY'
import sysconfig; print(sysconfig.get_path('scripts') or '')
PY
)
  if [ -n "$SCRIPTS_DIR" ]; then
    [ -x "$SCRIPTS_DIR/mlatom" ] && MLCMD="$SCRIPTS_DIR/mlatom"
    [ -z "$MLCMD" ] && [ -x "$SCRIPTS_DIR/MLatom.py" ] && MLCMD="python $SCRIPTS_DIR/MLatom.py"
  fi
fi
if [ -z "$MLCMD" ]; then
  echo "[ERROR] 未找到 mlatom/MLatom.py。请先激活环境。" >&2
  exit 1
fi

# 检查训练规模列表
if [ ! -s "$TRAIN_FILE" ]; then
  echo "[INFO] 未找到 $TRAIN_FILE，创建默认 40000"
  echo 40000 > "$TRAIN_FILE"
fi

mkdir -p train
cd train
train_wd=$(pwd)
ln -snf "$data/x.dat" x.dat

# 逐个训练规模
while read -r train_set_num; do
  cd "$train_wd"
  mkdir -p "$train_set_num"
  cd "$train_set_num"
  train_num_wd=$(pwd)

  # 切分 train/valid
  cp "$data/itrain.dat" .
  bash "$script/training_set_generator.sh" "$train_set_num"
  train_number=$(wc -l < itrain.dat)
  subtrain_number=$(wc -l < isubtrain.dat)
  validate_number=$(wc -l < ivalidate.dat)
  ln -snf "$data/x.dat" x.dat
  mkdir -p E f all

  ########## 训练 E ##########
  cd "$train_num_wd/E"
  mkdir -p "$EM_STATE"
  cd "$EM_STATE"
  ln -snf ../../itrain.dat itrain.dat
  ln -snf ../../isubtrain.dat isubtrain.dat
  ln -snf ../../ivalidate.dat ivalidate.dat
  cp -d ../../x.dat x.dat
  cp "$data/ml.E.inp" ml.inp
  sed -i "s/E1/E${EM_STATE}/g" ml.inp
  sed -i "s/subtrain_num/${subtrain_number}/g" ml.inp
  sed -i "s/train_num/${train_number}/g" ml.inp
  sed -i "s/validate_num/${validate_number}/g" ml.inp
  python3 "$script/generate_y.E.py" "$data/state${EM_STATE}.index.E.f" "E${EM_STATE}"
  $MLCMD ml.inp > ml.log
  cp "E${EM_STATE}est.dat" ../../all/

  ########## 训练 f ##########
  cd "$train_num_wd/f"
  mkdir -p "$EM_STATE"
  cd "$EM_STATE"
  ln -snf ../../itrain.dat itrain.dat
  ln -snf ../../isubtrain.dat isubtrain.dat
  ln -snf ../../ivalidate.dat ivalidate.dat
  cp -d ../../x.dat x.dat
  cp "$data/ml.f.inp" ml.inp
  sed -i "s/f1/f${EM_STATE}/g" ml.inp
  sed -i "s/subtrain_num/${subtrain_number}/g" ml.inp
  sed -i "s/train_num/${train_number}/g" ml.inp
  sed -i "s/validate_num/${validate_number}/g" ml.inp
  python3 "$script/generate_y.f.py" "$data/state${EM_STATE}.index.E.f" "f${EM_STATE}"
  $MLCMD ml.inp > ml.log
  cp "f${EM_STATE}est.dat" ../../all/

  ########## 混合 真值+预测 ##########
  cd "$train_num_wd"
  mv -f all all.ML.orig.data
  mkdir -p all

  python3 "$script/generate_y.E.mix.py" "$data/state${EM_STATE}.index.E.f" "E${EM_STATE}" "$train_set_num" "all.ML.orig.data/E${EM_STATE}est.dat"
  awk '{print ($0<0?"0.0000":$0)}' "E${EM_STATE}.dat" > "all/E${EM_STATE}est.dat"
  rm -f "E${EM_STATE}.dat"

  python3 "$script/generate_y.f.mix.py" "$data/state${EM_STATE}.index.E.f" "f${EM_STATE}" "$train_set_num" "all.ML.orig.data/f${EM_STATE}est.dat"
  awk '{print ($0<0?"0.0000":$0)}' "f${EM_STATE}.dat" > "all/f${EM_STATE}est.dat"
  rm -f "f${EM_STATE}.dat"

  ########## 由 E,f 生成发射谱 ##########
  mkdir -p spectrum/emission
  cd spectrum/emission

  case "$WEIGHT" in
    f)  AWK_W='r' ;;
    E)  AWK_W='e*r' ;;
    E3) AWK_W='e*e*e*r' ;;
    *)  echo "[WARN] 未知 WEIGHT=${WEIGHT}，使用 E3"; AWK_W='e*e*e*r' ;;
  esac

  {
    echo "DE/eV    lambda/nm    diff_rate        +/-error"
    paste ../../all/E${EM_STATE}est.dat ../../all/f${EM_STATE}est.dat | \
    awk -v w="$AWK_W" '{
      e=$1; r=$2; lam=(e>1e-12)?1239.84193/e:0.0;
      if(w=="r") diff=r; else if(w=="e*r") diff=e*r; else diff=e*e*e*r;
      printf("%8.4f   %10.4E   %12.8E   0.00000\n", e, lam, diff);
    }'
  } > emission-rate-ML.dat

  python3 "$script/emission_spectrum.py" emission-rate-ML.dat --delta "$DELTA" --eps "$EPS"
  echo "[OK] train_set=${train_set_num} -> spectrum/emission/emission_spectrum_eV.dat"

  cd "$train_wd/$train_set_num"
done < "$TRAIN_FILE"


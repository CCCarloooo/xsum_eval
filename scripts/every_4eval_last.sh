#当前文件夹
CURRENT_PATH='/home/mxd/archive/xsum_eval'
cd $CURRENT_PATH

# no_need_to_modify
MODEL_PATH='/home/mxd/archive/hf_mirror/gpt2-xl'
DATA_PATH='data/xsum_test_suitable.json'
METRIC_PATH='rouge'
RANK='1'

# /home/mxd/archive/models/1212_2128_len512_r1_epoch1_seed42_lora_finesave
# /home/mxd/archive/models/1212_2129_len512_r1_epoch1_seed42_periodiclora_finesave
# model_epoch_0_update_0_0.pth
# model_epoch_0_update_0
# seed_42_series/epoch_0_update_0_0

SERIES='/home/mxd/archive/models/1212_2129_len512_r1_epoch1_seed42_periodiclora_finesave/checkpoint/'
change_seed='epoch_0_update_8'
CURRENT="model_${change_seed}"
model_results="seed_42_series_plora/${change_seed}"

for i in {0..3}; do
  CUDA_DEVICES="$(expr $i + 4)"
  if [ $i -eq 0 ]; then
    PTH_PATH="${SERIES}${CURRENT}.pth"
    SAVE_PATH="converted_models/${model_results}"
    OUT_DIR="results/${model_results}"
  else
    PTH_PATH="${SERIES}${CURRENT}_$(expr $i - 1).pth"
    SAVE_PATH="converted_models/${model_results}_$(expr $i - 1)"
    OUT_DIR="results/${model_results}_$(expr $i - 1)"
  fi
  (
    source $CURRENT_PATH/scripts/launch_all.sh
  ) &
done

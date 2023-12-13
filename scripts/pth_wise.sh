CURRENT_PATH='/home/mxd/archive/xsum_eval'
#当前文件夹
cd $CURRENT_PATH

# 检测是否存在results文件夹，不存在则创建
if [ ! -d "results" ]; then
    mkdir results
fi

MODEL_PATH='/home/mxd/archive/hf_mirror/gpt2-xl'
DATA_PATH='data/xsum_test_suitable.json'
METRIC_PATH='rouge'
RANK='1'

CUDA_DEVICES='3'
#the following three lines need to modify with model
PTH_PATH='/home/mxd/archive/models/1209_1750_len512_r1_epoch4_seed3407_lora/whole_model/fullmodel_epoch_3_step_4.pth'
SAVE_PATH='converted_models/3407_seed/epoch_3_step_4_r1'
OUT_DIR='results/3407_seed/epoch_3_step_4_r1'


source $CURRENT_PATH/scripts/launch_all.sh
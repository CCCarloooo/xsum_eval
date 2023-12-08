export CURRENT_PATH='/home/mxd/archive/xsum_eval'
#当前文件夹
cd $CURRENT_PATH

# 检测是否存在results文件夹，不存在则创建
if [ ! -d "results" ]; then
    mkdir results
fi

export MODEL_PATH='/home/mxd/archive/hf_mirror/gpt2-xl'
export DATA_PATH='data/xsum_test_suitable.json'
export METRIC_PATH='rouge'
export RANK='1'

export CUDA_DEVICES='5'
#the following three lines need to modify with model
export PTH_PATH='/home/mxd/archive/models/1208_130_len512_r1_epoch4_seed42_lora/whole_model/fullmodel_epoch_3_step_4.pth'
export SAVE_PATH='converted_models/epoch_3_step_4_r1_seed42'
export OUT_DIR='results/epoch_3_step_4_r1_seed42'


bash $CURRENT_PATH/scripts/launch_all.sh
export CURRENT_PATH='/mnt/data2/mxdi/archive/xsum_eval'
#当前文件夹
cd $CURRENT_PATH

# 检测是否存在results文件夹，不存在则创建
if [ ! -d "results" ]; then
    mkdir results
fi

export MODEL_PATH='/mnt/data2/mxdi/archive/hf-mirror/gpt2-xl'
export DATA_PATH='data/xsum_test_suitable.json'
export METRIC_PATH='rouge'
export CUDA_VISIBLE_DEVICES='5'
export RANK='8'
#the following three lines need to modify with model

export PTH_PATH='/mnt/data2/mxdi/archive/models/1207/1207_0149_len512_r8/whole_model/fullmodel_epoch_1.pth'
export SAVE_PATH='converted_models/ep1_len512-r8'
export OUT_DIR='results/ep1_len512-r8-cut-t0'


bash $CURRENT_PATH/scripts/launch_all.sh
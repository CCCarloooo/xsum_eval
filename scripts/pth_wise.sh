cd /home/mxd/archive/xsum_eval/scripts

# 检测是否存在results文件夹，不存在则创建
if [ ! -d "results" ]; then
    mkdir results
fi

export CURRENT_PATH='/home/mxd/archive/xsum_eval'
export MODEL_PATH='/home/mxd/archive/hf_mirror/gpt2-xl'
export DATA_PATH='/home/mxd/archive/xsum_eval/data/xsum_test_suitable.json'
export METRIC_PATH='/home/mxd/archive/xsum_eval/rouge'
export CUDA_VISIBLE_DEVICES='5'
#the following three lines need to modify with model
export PTH_PATH='/home/mxd/archive/models/1207/ep0_len512-r1.pth'
export SAVE_PATH='/home/mxd/archive/xsum_eval/converted_models/ep0_len512-r1'
export OUT_DIR='/home/mxd/archive/xsum_eval/results/ep0_len512-r1-cut-t0'


bash launch_all.sh
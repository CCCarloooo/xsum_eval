cd /home/mxd/archive/xsum_eval/scripts

export CURRENT_PATH='/home/mxd/archive/xsum_eval'
export MODEL_PATH='/home/mxd/archive/hf_mirror/gpt2-xl'
export PTH_PATH='/home/mxd/archive/models/1207/ep0_len512-r1.pth'
export SAVE_PATH='/home/mxd/archive/xsum_eval/converted_models/ep0_len512-r1'
export METRIC_PATH='/home/mxd/archive/xsum_eval/rouge'
export DATA_PATH='/home/mxd/archive/xsum_eval/data/xsum_test_suitable.json'
export OUT_DIR='/home/mxd/archive/xsum_eval/results/ep0_len512-r1-cut-t0'
export CUDA_VISIBLE_DEVICES='5'

bash launch_all.sh
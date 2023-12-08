cd $CURRENT_PATH

python convert_model.py \
    --model_path $MODEL_PATH \
    --pth_path $PTH_PATH \
    --save_path $SAVE_PATH

echo "模型加载完毕"

CUDA_VISIBLE_DEVICES=5 python generate_vllm.py \
    --metric_path $METRIC_PATH \
    --data_path $DATA_PATH \
    --out_dir $OUT_DIR \
    --model_path $SAVE_PATH \
    --tokenizer_path $SAVE_PATH 

cd $CURRENT_PATH

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python convert_model.py \
    --model_path $MODEL_PATH \
    --pth_path $PTH_PATH \
    --save_path $SAVE_PATH \
    --rank $RANK
    
echo "模型加载完毕"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python generate_vllm.py \
    --metric_path $METRIC_PATH \
    --data_path $DATA_PATH \
    --out_dir $OUT_DIR \
    --model_path $SAVE_PATH \
    --tokenizer_path $SAVE_PATH 

echo "生成完毕"
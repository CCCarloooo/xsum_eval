import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import argparse
from collections import OrderedDict

def main(args):
    pth_path = args.pth_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = torch.load(pth_path, map_location=device)

    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = LoraConfig(
        r=args.rank,
        lora_alpha=2*args.rank,
        target_modules=['c_attn','c_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)

    #新建一个oderdict
    new_state_dict = OrderedDict()
    for k, v in base.items():
        if 'module' in k:
            name = k
            name = name.replace('module.','')
            #添加新的name
            new_state_dict[name] = v
            #删除旧的name
        else:
            new_state_dict[k] = v

    peft_model.load_state_dict(new_state_dict)
    model = peft_model.merge_and_unload()
    save_path = args.save_path
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print('done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None, 
    )  
    parser.add_argument(
        "--pth_path", 
        type=str, 
        default=None, 
    )  
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=None, 
    ) 
    parser.add_argument(
        "--rank", 
        type=int, 
        default=1, 
    ) 
    args = parser.parse_args()
    if not args.model_path:
        raise ValueError("Please specify a --model_path, e.g. --model_path='gpt2-xl'")
    print("启动程序")
    main(args)
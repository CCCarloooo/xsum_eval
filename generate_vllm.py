from vllm import LLM, SamplingParams
import argparse
import os
import json
from evaluate import load

def main(args):
    data_path = args.data_path
    with open(data_path) as f:
        data = json.load(f)

    def generate_cut_or_nocut_prompt(data_point,instruction = "Summarize the following document into a one-sentence summary.", document_max_length=320):
        document = data_point["input"] if len(data_point["input"].split(' ')) <= document_max_length else ' '.join(data_point["input"].split(' ')[:document_max_length])
        full_prompt = f'{instruction}\n\nDocument:\n{document}\n\nSummary:\n'
        return full_prompt
    
    ref = []
    for dic in data:
        ref.append(dic['output'])

    prompts = []
    for i in range(len(data)):
        prompts.append(generate_cut_or_nocut_prompt(data[i]))
    # data

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    llm = LLM(model=model_path, tokenizer=tokenizer_path)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=128,
        stop=["\n\n"],
    )
    generations = llm.generate(prompts, sampling_params)

    prompt_to_output = {
                    g.prompt: g.outputs[0].text for g in generations
                }
    outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
    #generate

    out_dir = args.out_dir
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'prediction.jsonl'), 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')

    metric_path = args.metric_path
    metrics = load(metric_path)
    rouge = metrics.compute(predictions=outputs, references=ref)
    with open(os.path.join(out_dir,"metrics.jsonl"), 'w') as f:
        json.dump(rouge, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric_path",
        type=str, 
        default="/home/mxd/archive/metrics/rouge"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/home/mxd/archive/carlo_periodlr/xsum_test_suitable.json"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="results/debug"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None, 
    )    
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=None, 
    ) 
    args = parser.parse_args()
    # 检测 model_path pth_path
    if not args.model_path:
        raise ValueError("Please specify a --model_path, e.g. --model_path='gpt2-xl'")
    print("启动程序")
    main(args)

import json
import transformers
import torch

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

print("hi")

model_name = 'models/ggml-model-q4_0.bin'
LOAD_8BIT = True
tokenizer = LlamaTokenizer.from_pretrained(model_name)

print("hi")

model = LlamaForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=LOAD_8BIT,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("hi")


with open('inputs.json') as f:
    inputs = json.load(f)
 
def without_emotion(conversation):
    utterence = conversation[-1]

    for u in conversation:
        del u['annotation']

    utterence['emotion'] = ''


    return json.dumps(conversation, indent=4)[:-10]

def get_input(string):
    return f"""Below is a conversation between two people, represented in a JSON format. An emotion is associated with each utterence from the following set: [neutral, joy, sadness, fear, anger, surprise, disgust].
    Non-neutral utterences are utterence whose emotion is ambiguous. {string}"""

def evaluate(
    inp=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = get_input(inp)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output

for conversation in inputs:
    inp = without_emotion(conversation)

    print(evaluate(inp=inp))






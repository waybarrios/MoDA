import argparse
import torch
import pandas as pd
import re
from tqdm import tqdm
from datasets import load_dataset

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import requests
from PIL import Image
from io import BytesIO


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def extract_answer_choice(response_text):
    """
    Extract the answer choice (A), (B), (C), or (D) from the model response.
    """
    # Look for patterns like (A), (B), (C), (D) in the response
    pattern = r'\(([ABCD])\)'
    matches = re.findall(pattern, response_text)
    
    if matches:
        return matches[0]  # Return the first match
    
    # If no parentheses pattern found, look for standalone letters
    pattern = r'\b([ABCD])\b'
    matches = re.findall(pattern, response_text)
    
    if matches:
        return matches[0]
    
    # If still no match, return the first character if it's A, B, C, or D
    first_char = response_text.strip()[0].upper() if response_text.strip() else ""
    if first_char in ['A', 'B', 'C', 'D']:
        return first_char
    
    return None  # No valid answer found


def create_prompt_with_choices(question, choices):
    """
    Create a prompt that encourages the model to respond with (A), (B), (C), or (D).
    """
    prompt = f"{question}\n\n"
    for i, choice in enumerate(choices):
        letter = chr(65 + i)  # A, B, C, D
        prompt += f"({letter}) {choice}\n"
    
    prompt += "\nPlease answer with only the letter choice in parentheses, such as (A), (B), (C), or (D)."
    return prompt


def eval_model_on_cvbench(args):
    # Model loading
    disable_torch_init()
    
    model_name = get_model_name_from_path(args.model_path)
    model_name = 'llava'
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    
    # Load CV-Bench dataset
    print("Loading CV-Bench dataset...")
    cv_bench = load_dataset("nyu-visionx/CV-Bench")
    
    # Use the test split if available, otherwise use the default split
    if 'test' in cv_bench:
        dataset = cv_bench['test']
    else:
        dataset = cv_bench['train'] if 'train' in cv_bench else cv_bench[list(cv_bench.keys())[0]]
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Set conversation mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    
    if args.conv_mode is not None:
        conv_mode = args.conv_mode
    
    print(f"Using conversation mode: {conv_mode}")
    
    # Results storage
    results = []
    
    # Process each sample
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            # Get image
            image = sample['image']
            if image is None:
                print(f"Skipping sample {idx}: No image available")
                continue
                
            # Create prompt with choices
            question = sample['question']
            choices = sample['choices']
            qs = create_prompt_with_choices(question, choices)
            
            # Add image tokens
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            
            # Setup conversation
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Process image
            images_tensor = process_images(
                [image],
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
            
            # Tokenize
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            
            # Generate response
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            
            # Decode response
            response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Extract answer choice
            predicted_choice = extract_answer_choice(response)
            correct_answer = sample['answer']
            
            # Check if prediction is correct
            is_correct = 1 if predicted_choice == correct_answer else 0
            
            # Store result
            result = {
                'idx': sample['idx'],
                'type': sample['type'],
                'task': sample['task'],
                'source': sample['source'],
                'source_dataset': sample['source_dataset'],
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_choice,
                'full_response': response,
                'result': is_correct,
                'filename': sample.get('filename', ''),
                'target_class': sample.get('target_class', ''),
                'target_size': sample.get('target_size', ''),
                'bbox': sample.get('bbox', '')
            }
            
            results.append(result)
            
            if idx % 100 == 0 and idx > 0:
                print(f"Processed {idx}/{len(dataset)} samples")
                print(f"Valid predictions so far: {len([r for r in results if r['predicted_answer'] is not None])}/{len(results)}")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Save results to CSV
    df = pd.DataFrame(results)
    output_file = args.output_file
    df.to_csv(output_file, index=False)
    print(f"\nInference completed!")
    print(f"Results saved to {output_file}")
    print(f"Total samples processed: {len(results)}")
    print(f"Samples with valid predictions: {len([r for r in results if r['predicted_answer'] is not None])}")
    print(f"\nUse cvbench_evaluation.py to calculate accuracy metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, 
                       default="/gpudata3/Wayner/LLaVA-MORE-l1-linear-xattn/checkpoints/masking_l1_xtann_layer1_llava")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default='llama_3_1')
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)  # Reduced since we only need short answers
    parser.add_argument("--output-file", type=str, default="cv_bench_results_llava_more_openai_layer1.csv")
    
    args = parser.parse_args()
    
    print(f"Conversation mode: {args.conv_mode}")
    print(f"Model path: {args.model_path}")
    print(f"Output file: {args.output_file}")
    
    eval_model_on_cvbench(args)

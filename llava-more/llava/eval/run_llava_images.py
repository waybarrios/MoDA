from tqdm import tqdm
import argparse
import os
import glob
import re
import torch
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

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
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


def load_image(image_file):
    """
    Carga una imagen desde una URL o desde disco.
    """
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    # Deshabilitamos la inicialización por defecto de Torch para evitar warnings.
    disable_torch_init()

    # Cargamos el modelo y sus componentes
    model_name = get_model_name_from_path(args.model_path)
    # En este ejemplo se sobreescribe el nombre a 'llava'
    model_name = 'llava'
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Preparamos el query de conversación
    qs = args.query
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

    # Se infiere el modo de conversación en base al nombre del modelo
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

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f"[WARNING] El modo de conversación inferido es {conv_mode}, mientras que '--conv-mode' es {args.conv_mode}. Se usará {args.conv_mode}.")
    else:
        args.conv_mode = conv_mode

    # Se configura la conversación a partir del template
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    conv.tokenizer = tokenizer

    # Preparamos los input_ids a partir del prompt (se reusa para cada imagen)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # Obtenemos la lista de imágenes (en este ejemplo, se buscan archivos .jpg en la carpeta)
    image_files = glob.glob(os.path.join(args.image_folder, "*.jpg"))
    if not image_files:
        print(f"No se encontraron imágenes en {args.image_folder}")
        return

    results = []  # Lista para almacenar los resultados (nombre de imagen y caption)

    # Iteramos sobre cada imagen
    for image_file in tqdm(image_files):
        print(f"Procesando {image_file} ...")
        image = load_image(image_file)
        # Se obtiene el tamaño de la imagen (dentro de una lista, ya que se procesa en batch de 1)
        image_size = [image.size]
        # Se procesa la imagen y se convierte a tensor
        images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

        # Se corre la inferencia para generar la descripción
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_size,
                image_name = image_file,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        results.append({"image": os.path.basename(image_file), "caption_50": outputs})
        print(f"Caption generada: {outputs}\n")

    # Se guardan los resultados en un archivo CSV usando pandas
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Resultados guardados en {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning",
        help="Ruta o identificador del modelo preentrenado",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-folder",
        type=str,
        default="./images_demo",
        help="Carpeta donde se encuentran las imágenes (formato jpg)",
    )
    parser.add_argument("--query", type=str, default="Describe this image.")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_csv", type=str, default="captions_50.csv", help="Archivo CSV de salida con las captions")
    args = parser.parse_args()

    print(f"Modo de conversación: {args.conv_mode}")
    print(f"Ruta del modelo: {args.model_path}")
    print(f"Carpeta de imágenes: {args.image_folder}")

    main(args)


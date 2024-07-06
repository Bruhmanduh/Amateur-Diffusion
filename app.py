import streamlit as st
import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import model_converter
import time

def generate_image(text: str):
    DEVICE = "cpu"

    ALLOW_CUDA = True
    ALLOW_MPS = False

    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
        DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    tokenizer = CLIPTokenizer("C:\\Users\\samch\\Stable_Diffusion\\data\\tokenizer_vocab.json", merges_file="C:\\Users\\samch\\Stable_Diffusion\\data\\tokenizer_merges.txt")
    model_file = "C:\\Users\\samch\\Stable_Diffusion\\data\\v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

    ## TEXT TO IMAGE

    prompt = text
    uncond_prompt = ""  # Also known as negative prompt
    do_cfg = True
    cfg_scale = 13  # min: 1, max: 14
    strength = 0.9

    ## SAMPLER

    sampler = "ddpm"
    num_inference_steps = 50
    seed = 42

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=None,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    # Combine the input image and the output image into a single image.
    image = Image.fromarray(output_image)

    return image


st.title('Amateur Diffusion')
with st.expander("About this app"):
    st.write("This is a goofy app by a rando highschooler playin around with ai")

input_text = st.text_input('Enter your prompt cuh')
if input_text is not None:
    if st.button('Generate Image'):
        st.info(input_text)

        image = generate_image(input_text)
        st.image(image, caption='this image is generated using Amateur Diffusion', use_column_width=True)




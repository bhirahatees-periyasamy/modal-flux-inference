
import random
import time
from io import BytesIO
from pathlib import Path

import modal
import secrets
import string

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).entrypoint([])



diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]==0.33.1",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.0",
        "peft==0.7.1",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": "/cache"})
)



flux_image = flux_image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "HF_TOKEN" : "<hf_token>"
    }
)


app = modal.App("lora-flux", image=flux_image)

with flux_image.imports():
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import login
    import os 

# ## Defining a parameterized `Model` inference class

MINUTES = 60  # seconds
VARIANT = "dev"  # or "dev", but note [dev] requires you to accept terms and conditions on HF
NUM_INFERENCE_STEPS = 50  # use ~50 for [dev], smaller for [schnell]


@app.cls(
    gpu="A100",  # fastest GPU on Modal
    scaledown_window=20 * MINUTES,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name("triton-cache", create_if_missing=True),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True
        ),
    },
)
class Model:
    compile: bool = (  # see section on torch.compile below for details
        modal.parameter(default=False)
    )

    @modal.enter()
    def enter(self):
        if os.getenv("HF_TOKEN"):
            login(token=os.getenv("HF_TOKEN"))
        pipe = DiffusionPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{VARIANT}", torch_dtype=torch.bfloat16,
        ).to("cuda")  # move model to GPU
        pipe.load_lora_weights("<hf_repository_name>")
        self.pipe = optimize(pipe, compile=self.compile)

    @modal.method()
    def inference(self, prompt: str) -> list[bytes]:
        print("🎨 generating image...")
        out = self.pipe(
            prompt,
            output_type="pil",
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_images_per_prompt=2
        ).images
        result = []
        for i in out:
            byte_stream = BytesIO()
            i.save(byte_stream, format="JPEG")
            result.append(byte_stream.getvalue())
        return result


@app.local_entrypoint()
def main(
    prompt: str = """beautiful watercolor style blue, business portrait photo Caucasian man with glasses""",
    twice: bool = False,
    compile: bool = False,
):
    t0 = time.time()
    image_bytes = Model(compile=compile).inference.remote(prompt)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    if twice:
        t0 = time.time()
        image_bytes = Model(compile=compile).inference.remote(prompt)
        print(f"🎨 second inference latency: {time.time() - t0:.2f} seconds")
    for image in image_bytes:
        characters = string.ascii_letters + string.digits
        name = ''.join(secrets.choice(characters) for _ in range(15))
        output_path = Path("/tmp") / "flux" / f"{name}.jpg" # Output directory
        output_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"🎨 saving output to {output_path}")
        output_path.write_bytes(image)



def optimize(pipe, compile=True):
    # Try to fuse QKV projections in Transformer and VAE
    try:
        pipe.transformer.fuse_qkv_projections()
    except (AttributeError, RuntimeError) as e:
        print(f"⚠️ Skipping QKV fusion in transformer: {e}")

    try:
        pipe.vae.fuse_qkv_projections()
    except (AttributeError, RuntimeError) as e:
        print(f"⚠️ Skipping QKV fusion in VAE: {e}")

    # Switch to channels_last memory format for performance
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # Set torch.compile optimizations
    config = torch._inductor.config
    config.disable_progress = False
    config.conv_1x1_as_mm = True
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False

    # Compile Transformer and VAE decoder
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    # Trigger compilation by running a dummy inference
    print("🔦 running torch compilation (may take up to 20 minutes)...")
    pipe(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=NUM_INFERENCE_STEPS,
    ).images[0]
    print("✅ finished torch compilation")

    return pipe

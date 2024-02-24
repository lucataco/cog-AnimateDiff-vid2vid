# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import imageio
import subprocess
from PIL import Image
from io import BytesIO
from diffusers.utils import export_to_gif
from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/SG161222/Realistic_Vision_V5.1_noVAE.tar"
ADAPTER_NAME = "guoyww/animatediff-motion-adapter-v1-5-2"
ADAPTER_CACHE = "adapter-cache"
ADAPTER_URL = "https://weights.replicate.delivery/default/guoyww/animatediff-motion-adapter-v1-5-2.tar"

# helper function to load videos
def load_video(file_path: str):
    images = []
    vid = imageio.get_reader(file_path)
    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)

    return images

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading model")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        print("Loading adapter")
        if not os.path.exists(ADAPTER_CACHE):
            download_weights(ADAPTER_URL, ADAPTER_CACHE)
        # Load the motion adapter
        self.adapter = MotionAdapter.from_pretrained(
            ADAPTER_NAME,
            torch_dtype=torch.float16,
            cache_dir=ADAPTER_CACHE
        )
        self.pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
            BASE_MODEL,
            motion_adapter=self.adapter,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE
        ).to("cuda")
        # enable memory savings
        self.pipe.enable_vae_slicing()
        self.pipe.enable_model_cpu_offload()

    @torch.inference_mode()
    def predict(
        self,
        video: Path = Input(description="Input video"),
        prompt: str = Input(description="Prompt for the model", default="panda playing a guitar, on a boat, in the ocean, high quality"),
        negative_prompt: str = Input(description="Negative prompt for the model", default="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"),
        guidance_scale: float = Input(description="Guidance scale for the model", default=7.5),
        num_inference_steps: int = Input(description="Number of inference steps", default=25),
        strength: float = Input(description="Strength of the initial image", default=0.6),
        seed: int = Input(description="Random seed, leave blank to randomize the seed", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        scheduler = DDIMScheduler.from_pretrained(
            BASE_MODEL,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        self.pipe.scheduler = scheduler

        video = load_video(str(video))
        output = self.pipe(
            video = video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
        )
        frames = output.frames[0]
        output_path = "/tmp/output.gif"
        export_to_gif(frames, output_path)

        return Path(output_path)
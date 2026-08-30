"""
Modal deployment for EchoMimicV3-Flash talking head generation.

Deploy:
    modal deploy docker/modal-echomimic3/app.py

Candidate replacement for docker/modal-sadtalker/app.py. Generates a talking
head video from a portrait image + audio file, using Ant Group's EchoMimicV3
(Apache 2.0, 1.3B params, built on Wan2.1-Fun-V1.1-1.3B-InP).

Why the Flash variant: 5-8 denoise steps instead of 25, and it is the one that
fits the toolkit's existing 24GB endpoint tier. `echomimicv3-flash-pro` weights
override the base Wan transformer.

Long audio:
    Upstream `infer_flash.py` generates ONE segment and silently truncates the
    audio to `video_length` frames (81 = 3.24s at 25fps), and the Flash pipeline
    (`pipeline_wan_fun_inpaint_audio_2512`) does not accept the long-video kwargs
    that the preview pipeline does. So the segment loop lives here instead:
    re-anchor each segment on the last `overlap` frames of the previous one and
    cross-fade the seam. This mirrors the loop in upstream `infer_preview.py`.

Note this is NOT a drop-in for SadTalker's aspect behaviour by accident -- it is
better on purpose. Output aspect ratio follows the input image (see _fit_size),
so a 16:9 portrait yields a 16:9 clip with no `--preprocess full` workaround.

Input format (POST JSON to web endpoint):
{
    "image_url" | "image_base64": str,
    "audio_url" | "audio_base64": str,
    "prompt": str,                  # default: "A person is speaking."
    "steps": int,                   # default: 8 (5 is enough for talking head)
    "video_length": int,            # frames per segment, default 81
    "overlap": int,                 # blended frames between segments, default 8
    "anchor_retreat": int,          # max frames to back off a bad seam, default 6
    "sample_size": [int, int],      # area target, default [768, 768]
    "guidance_scale": float,        # default 6.0
    "audio_guidance_scale": float,  # default 3.0  (1.8-2.0 per upstream README)
    "audio_scale": float,           # default 1.0
    "shift": float,                 # default 5.0
    "seed": int,                    # default 43
    "fps": int,                     # default 25
    "teacache_threshold": float,    # default 0.1, 0 disables
    "negative_prompt": str,
    "wav2vec": "chinese" | "english",
    "r2": dict                      # optional R2 upload config
}
"""

import os

import modal

REPO_URL = "https://github.com/antgroup/echomimic_v3.git"
# Pinned so a rebuild cannot silently pick up a new upstream main whose src/
# module layout no longer matches the loader below. Bump deliberately, then
# smoke-test the endpoint (same lesson as flux2/diffusers in #71, #74).
REPO_REF = "7e89489ca51c0d008fc1963ec6c03fc5bd0b9397"

# Weights are pinned by revision for the same reason the repo is. It matters more
# here than in the baked apps: with the weights in a Volume they are no longer
# part of the image, so nothing but these SHAs stops image and weights drifting
# apart (#76). Bump deliberately, then re-run populate_weights.
BASE_MODEL = "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP"
BASE_MODEL_REV = "fc913c34361f4ec879e2f9c78b4f11ae50a937d1"
ECHO_MODEL = "BadToBest/EchoMimicV3"
ECHO_MODEL_REV = "311e176905a8c4c24b240b530488fe636ce4d249"
WAV2VEC_CN = "TencentGameMate/chinese-wav2vec2-base"   # what run_flash.sh uses
WAV2VEC_CN_REV = "3991242c806928916fff4a8c0e4f76acf661b743"
WAV2VEC_EN = "facebook/wav2vec2-base-960h"             # preview default
WAV2VEC_EN_REV = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"

APP_DIR = "/app/echomimic_v3"
MODELS_DIR = "/models"

# Where weights come from. This app keeps them in a Modal Volume, unlike the
# other six modal-* apps which bake them into the image. That split is deliberate
# and measured (#76): rebuild after a dependency change is 1.8-8.2s on a volume
# against 79-385s baked, while cold start and generation speed are the same
# either way. Getting this model working took four dependency changes, so the
# rebuild cost is the one that bites. The settled apps gain nothing by moving.
#
# Set ECHOMIMIC_WEIGHTS=image at DEPLOY time for the baked variant, which
# deploys as a separate app -- self-contained, but re-downloads 26GB whenever a
# layer above the weights is invalidated.
WEIGHTS_SOURCE = os.environ.get("ECHOMIMIC_WEIGHTS", "volume")
if WEIGHTS_SOURCE not in ("image", "volume"):
    raise ValueError(f"ECHOMIMIC_WEIGHTS must be 'image' or 'volume', got {WEIGHTS_SOURCE!r}")

USE_VOLUME = WEIGHTS_SOURCE == "volume"

app = modal.App(
    "video-toolkit-echomimic3" if USE_VOLUME else "video-toolkit-echomimic3-baked"
)

# Created in both modes: unused (and empty, so free) in image mode, but having
# the handle unconditionally keeps the module importable either way.
volume = modal.Volume.from_name("echomimic3-weights", create_if_missing=True)

_WEIGHT_FETCH = [
    (BASE_MODEL, BASE_MODEL_REV, f"{MODELS_DIR}/Wan2.1-Fun-V1.1-1.3B-InP", None),
    # Only the Flash transformer -- skips the larger preview checkpoint.
    (ECHO_MODEL, ECHO_MODEL_REV, f"{MODELS_DIR}/EchoMimicV3", ["echomimicv3-flash-pro/*"]),
    (WAV2VEC_CN, WAV2VEC_CN_REV, f"{MODELS_DIR}/chinese-wav2vec2-base", None),
    (WAV2VEC_EN, WAV2VEC_EN_REV, f"{MODELS_DIR}/wav2vec2-base-960h", None),
]

# chinese-wav2vec2-base ships pytorch_model.bin, and current transformers refuses
# torch.load outright below torch 2.6 (CVE-2025-32434) -- it fails the load rather
# than warning. Converting to safetensors is the cheap fix: bumping to torch 2.6
# would flip torch.load's weights_only default and break EchoMimic's own .pth
# loads for the VAE, text encoder and CLIP, trading one breakage for three.
def _fetch_weights():
    """Download weights into MODELS_DIR. Runs at image build OR into the volume."""
    import os

    import safetensors.torch
    import torch
    from huggingface_hub import snapshot_download

    for repo, revision, dest, patterns in _WEIGHT_FETCH:
        snapshot_download(repo, revision=revision, local_dir=dest, allow_patterns=patterns)

    for d in (f"{MODELS_DIR}/chinese-wav2vec2-base", f"{MODELS_DIR}/wav2vec2-base-960h"):
        b, sf = os.path.join(d, "pytorch_model.bin"), os.path.join(d, "model.safetensors")
        if os.path.exists(b) and not os.path.exists(sf):
            sd = torch.load(b, map_location="cpu", weights_only=True)
            safetensors.torch.save_file({k: v.contiguous() for k, v in sd.items()}, sf)
            os.remove(b)
            print("converted", d)
        else:
            print("already safetensors", d)

    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(MODELS_DIR) for f in fs
    )
    print(f"weights ready: {total / 1e9:.1f} GB")


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1")
    .pip_install(
        # Pinned, not floored. EchoMimicV3's requirements.txt says
        # diffusers>=0.30.1 / transformers>=4.46.2, but those floors resolve to
        # current releases that the repo's vendored code does not survive:
        #   - transformers 5.x drives output_hidden_states from config rather than
        #     the kwarg EchoMimic's Wav2Vec2Model subclass passes, so the encoder
        #     returns hidden_states=None and there are no audio embeddings at all
        #     (setting config.output_hidden_states does not rescue it).
        #   - transformers >=4.51.3 also refuses torch.load below torch 2.6.
        #   - current diffusers moved load_model_dict_into_meta, which silently
        #     disables low_cpu_mem_usage on the transformer load.
        # These two are contemporary with the repo and mutually compatible.
        "diffusers==0.32.2",
        "transformers==4.49.0",
        "accelerate>=0.25.0",
        "safetensors",
        "omegaconf",
        "einops",
        "timm",
        "tomesd",
        "torchdiffeq",
        "torchsde",
        "imageio[ffmpeg]",
        "imageio[pyav]",
        "opencv-python-headless",
        "scikit-image",
        "librosa",
        "pyloudnorm",
        "SentencePiece",
        "ftfy",
        "func_timeout",
        "Pillow",
        "numpy<2",
        "boto3",
        "requests",
        "fastapi[standard]",
        "huggingface_hub>=0.25.0",
    )
    # Upstream requirements.txt also pins tensorflow==2.15.0 + retina-face +
    # gradio + decord + moviepy. None of those are reachable from the Flash
    # path: tensorflow/retina-face are only for src.face_detect (the preview
    # ip_mask), and gradio/moviepy only for the demo UIs. Left out to keep the
    # image small -- add them back if the preview variant is ever wired up.
    # Fetch-by-SHA rather than `clone --branch`, which only accepts a branch or
    # tag name. Still a single-commit download.
    .run_commands(
        f"git init {APP_DIR}",
        f"git -C {APP_DIR} remote add origin {REPO_URL}",
        f"git -C {APP_DIR} fetch --depth 1 origin {REPO_REF}",
        f"git -C {APP_DIR} checkout FETCH_HEAD",
    )
    .env({
        "PYTHONPATH": APP_DIR,
        "TOKENIZERS_PARALLELISM": "false",
        # Modal re-imports this module inside the container, where the local
        # shell env does NOT exist. Without baking the mode in, USE_VOLUME reads
        # False in-container regardless of how it was deployed -- which silently
        # drops the volume mount and hides populate_weights.
        "ECHOMIMIC_WEIGHTS": WEIGHTS_SOURCE,
    })
)

if not USE_VOLUME:
    # Baked variant: weights become image layers. Every dependency change above
    # this point invalidates them and re-downloads ~26GB.
    image = image.run_function(_fetch_weights)

# Volume variant carries code only, so a dependency change rebuilds in seconds
# and leaves the weights untouched. Downloading needs no GPU.
download_image = (
    modal.Image.debian_slim(python_version="3.10")
    # numpy is not optional here: torch.load pulls it in during the
    # .bin -> safetensors conversion below.
    .pip_install("huggingface_hub>=0.25.0", "safetensors", "torch==2.5.1", "numpy<2")
    .env({"ECHOMIMIC_WEIGHTS": WEIGHTS_SOURCE})
)


@app.function(image=download_image, volumes={MODELS_DIR: volume}, timeout=3600)
def populate_weights():
    """One-off, idempotent: fill the volume before first use.

    ECHOMIMIC_WEIGHTS=volume modal run docker/modal-echomimic3/app.py::populate_weights

    Defined in both modes on purpose. Gating it behind USE_VOLUME made it vanish
    inside the container, where the module is re-imported without the deploy-time
    shell env.
    """
    _fetch_weights()
    volume.commit()


@app.cls(
    image=image,
    # A10G (24GB, 22.06 usable) matches the tier the toolkit's other endpoints
    # already run on. It does NOT fit with everything resident -- see
    # enable_model_cpu_offload in load_pipeline. With offload on it does fit,
    # at the cost of paging modules over PCIe on every pipeline call.
    gpu="A10G",
    volumes=({MODELS_DIR: volume} if USE_VOLUME else {}),
    timeout=7200,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=1)
class EchoMimicV3:
    @modal.enter()
    def load_pipeline(self):
        import os
        import time
        import torch

        _load_started = time.time()
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

        from src.wan_vae import AutoencoderKLWan
        from src.wan_image_encoder import CLIPModel
        from src.wan_text_encoder import WanT5EncoderModel
        from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel
        from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
        from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from src.utils import filter_kwargs
        from src.wav2vec2 import Wav2Vec2Model

        print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.model_name = f"{MODELS_DIR}/Wan2.1-Fun-V1.1-1.3B-InP"

        cfg = OmegaConf.load(f"{APP_DIR}/config/config.yaml")
        self.cfg = cfg

        # config.yaml uses transformer_subpath "./" -- the transformer config
        # lives at the root of the Wan2.1-Fun repo, and the Flash safetensors
        # below replace its weights.
        transformer = WanTransformerAudioMask3DModel.from_pretrained(
            os.path.join(self.model_name, cfg["transformer_additional_kwargs"].get("transformer_subpath", "./")),
            transformer_additional_kwargs=OmegaConf.to_container(cfg["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
        )

        flash_ckpt = f"{MODELS_DIR}/EchoMimicV3/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors"
        from safetensors.torch import load_file

        state_dict = load_file(flash_ckpt)
        state_dict = state_dict.get("state_dict", state_dict)
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        print(f"Flash checkpoint: {len(missing)} missing keys, {len(unexpected)} unexpected")
        # A large `missing` count means the Flash weights did not line up with
        # the base transformer -- output would be garbage rather than an error.
        if len(missing) > 50:
            raise RuntimeError(
                f"Flash checkpoint mismatch: {len(missing)} missing keys. "
                "Check that BASE_MODEL and the flash weights are the matching pair."
            )

        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(self.model_name, cfg["vae_kwargs"].get("vae_subpath", "Wan2.1_VAE.pth")),
            additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"]),
        ).to(self.dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.model_name, cfg["text_encoder_kwargs"].get("tokenizer_subpath", "google/umt5-xxl")),
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(self.model_name, cfg["text_encoder_kwargs"].get("text_encoder_subpath")),
            additional_kwargs=OmegaConf.to_container(cfg["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
        ).eval()
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(self.model_name, cfg["image_encoder_kwargs"].get("image_encoder_subpath")),
        ).to(self.dtype).eval()

        # Flow_Unipc is what run_flash.sh uses; it wants shift folded into the
        # pipeline call rather than the scheduler config.
        scheduler_cfg = OmegaConf.to_container(cfg["scheduler_kwargs"])
        scheduler_cfg["shift"] = 1
        scheduler = FlowUniPCMultistepScheduler(
            **filter_kwargs(FlowUniPCMultistepScheduler, scheduler_cfg)
        )

        self.pipeline = WanFunInpaintAudioPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        # Model CPU offload, NOT pipeline.to("cuda"). Measured on an A10 (22.06GB
        # usable): keeping the umt5-xxl text encoder, CLIP-huge image encoder,
        # transformer and VAE all resident OOMs during load -- 21.98GB consumed
        # before inference allocates anything. The pipeline declares
        # model_cpu_offload_seq, so diffusers can keep only the executing module
        # on GPU and page the rest back to host RAM.
        # Do not add a .to(device) call alongside this; diffusers rejects both.
        self.pipeline.enable_model_cpu_offload(device=self.device)
        self.vae_ratio = vae.config.temporal_compression_ratio

        # Audio encoders stay on CPU (upstream does the same) -- they are small
        # and this keeps VRAM for the transformer.
        self.audio_encoders = {}
        for key, path in (("chinese", f"{MODELS_DIR}/chinese-wav2vec2-base"),
                          ("english", f"{MODELS_DIR}/wav2vec2-base-960h")):
            enc = Wav2Vec2Model.from_pretrained(path, local_files_only=True).to("cpu")
            # EchoMimic's Wav2Vec2Model subclass passes output_hidden_states as a
            # kwarg to the inner encoder, but current transformers drives it from
            # config instead (the same release that deprecates use_return_dict).
            # Without this the encoder returns hidden_states=None and the audio
            # embedding stack fails -- and the audio embeddings ARE the lip sync,
            # so this has to be right, not merely non-crashing.
            enc.config.output_hidden_states = True
            enc.feature_extractor._freeze_parameters()
            self.audio_encoders[key] = (
                enc,
                Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True),
            )

        if torch.cuda.is_available():
            print(f"VRAM resident after load: {torch.cuda.memory_allocated() / 1e9:.1f}GB "
                  f"(offloaded; modules page in per call)")
        self.load_seconds = time.time() - _load_started
        print(f"Pipeline ready in {self.load_seconds:.1f}s")

    # -- helpers ------------------------------------------------------------

    def _round_frames(self, n: int, up: bool = False) -> int:
        """Snap a frame count to the VAE's temporal compression grid (4k+1).

        Segment lengths round UP so the final segment still covers the tail of
        the audio: rounding down leaves a remainder that the loop can only chew
        through a frame at a time. The total is rounded DOWN, and the render is
        trimmed to it, so the extra frames never reach the output.
        """
        if n <= 1:
            return 1
        r = self.vae_ratio
        steps = -(-(n - 1) // r) if up else (n - 1) // r
        return int(steps * r) + 1

    @staticmethod
    def _fit_size(img, target):
        """Pick an output size that keeps the image's aspect ratio.

        Scales to roughly `target` pixel area, rounded to /16. This is why a
        16:9 input gives a 16:9 output -- no square crop, unlike SadTalker.
        """
        import math

        w, h = img.size
        ori_a, tgt_a = w * h, target[0] * target[1]
        if tgt_a < ori_a:
            ratio = math.sqrt(ori_a / tgt_a)
            w, h = w / ratio // 16 * 16, h / ratio // 16 * 16
        else:
            w, h = w // 16 * 16, h // 16 * 16
        return int(h), int(w)

    @staticmethod
    def _build_inputs(start_images, video_length, height, width):
        """Build (input_video, input_video_mask, clip_image) for one segment.

        Reimplemented rather than calling src.utils.get_image_to_video_latent2:
        that helper calls .resize() on its argument before checking whether it
        is a list, so the multi-frame re-anchoring this loop needs would raise
        AttributeError there.
        """
        import numpy as np
        import torch
        from PIL import Image

        if not isinstance(start_images, list):
            start_images = [start_images]
        imgs = [
            im.convert("RGB").resize((width, height), resample=Image.Resampling.LANCZOS)
            for im in start_images
        ]
        clip_image = imgs[0]

        start = torch.cat(
            [torch.from_numpy(np.array(im)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for im in imgs],
            dim=2,
        )  # [1, 3, n, H, W]
        n = min(start.shape[2], video_length)

        video = torch.tile(start[:, :, :1], [1, 1, video_length, 1, 1]).clone()
        video[:, :, :n] = start[:, :, :n]
        video = video / 255

        mask = torch.zeros_like(video[:, :1])
        mask[:, :, n:] = 255
        return video, mask, clip_image

    @staticmethod
    def _pick_anchor_retreat(accumulated, overlap, max_retreat):
        """How many trailing frames to drop before re-anchoring the next segment.

        Each continuation segment starts from the last `overlap` frames of the
        previous one, so whatever pose those frames hold becomes the next
        segment's opening pose. When they land mid-blink the model starts
        closed-eyed and *holds* it -- observed as a prolonged closure across a
        segment boundary, and the reason this exists.

        Rather than detect eyes (which needs the face-landmark stack this image
        deliberately omits), score each candidate window by how much motion it
        contains in the upper half of the frame -- where blinks live and mouth
        movement does not -- and anchor on the calmest one. A blink is the
        largest short transient up there, so it scores worst and gets skipped.
        Anchoring on a settled pose is the better default regardless, since a
        continuation has to extrapolate from whatever it is handed.

        Returns frames to discard: 0 keeps the current behaviour, which is also
        what a clip with no transient near the seam gets, because dropping
        frames means regenerating them.
        """
        import torch

        total = accumulated.shape[2]
        # Never retreat past having `overlap` frames left to anchor on.
        budget = max(0, min(int(max_retreat), total - overlap - 1))
        if budget <= 0:
            return 0

        # Upper half only, and greyscale: blink transients are small, and mouth
        # motion in the lower half would otherwise dominate every score.
        tail = accumulated[0, :, -(overlap + budget + 1):]        # [C, n, H, W]
        upper = tail[:, :, : max(1, tail.shape[2] // 2)].mean(dim=0)  # [n, H/2, W]
        motion = (upper[1:] - upper[:-1]).abs().mean(dim=(1, 2))  # [n-1] per-frame

        # Window r covers the `overlap` frames ending `r` from the end. Score it
        # by its worst frame, not its mean: one blink frame in the window is
        # enough to poison the anchor.
        n = len(motion)
        scores = [
            torch.max(motion[n - overlap - r: n - r]).item()
            for r in range(budget + 1)
        ]

        best = min(range(len(scores)), key=lambda r: scores[r])
        # Only pay for a retreat when it is a clear improvement. Without this,
        # sensor-level noise picks an arbitrary r on every segment and quietly
        # adds a regenerated frame budget to clips that never needed one.
        if best == 0 or scores[best] >= 0.8 * scores[0]:
            return 0
        return best

    def _audio_embed(self, wav, start_frame, seg_frames, fps, which, sr=16000):
        """Wav2Vec embeddings for one segment, windowed the way Flash expects.

        Upstream builds a +/-2 frame window per output frame -> [F, 5, 12, 768].
        Slicing the waveform per segment (rather than slicing a whole-clip
        embedding) is what infer_flash.py does, and it keeps the encoder's
        seq_len argument consistent with the segment length.
        """
        import numpy as np
        import torch
        from einops import rearrange

        encoder, extractor = self.audio_encoders[which]

        lo = int(start_frame / fps * sr)
        hi = int((start_frame + seg_frames) / fps * sr)
        chunk = wav[lo:hi]

        feature = np.squeeze(extractor(chunk, sampling_rate=sr).input_values)
        feature = torch.from_numpy(feature).float().unsqueeze(0)
        with torch.no_grad():
            out = encoder(feature, seq_len=int(seg_frames), output_hidden_states=True,
                          return_dict=True)

        if getattr(out, "hidden_states", None) is None:
            raise RuntimeError(
                "wav2vec returned no hidden_states -- transformers is not honouring "
                "output_hidden_states for this encoder. Check config.output_hidden_states "
                "in load_pipeline, or pin transformers to a release contemporary with "
                "EchoMimicV3 (>=4.46.2, before use_return_dict was deprecated)."
            )

        emb = torch.stack(out.hidden_states[1:], dim=1).squeeze(0)
        emb = rearrange(emb, "b s d -> s b d").cpu().detach()

        indices = torch.arange(5) - 2
        centers = torch.arange(0, seg_frames).unsqueeze(1) + indices.unsqueeze(0)
        centers = torch.clamp(centers, min=0, max=emb.shape[0] - 1)
        return emb[centers].unsqueeze(0)

    # -- endpoint -----------------------------------------------------------

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Return as soon as @modal.enter() has finished.

        Timing a GET against a cold container measures schedule + weight fetch +
        model load, without spending GPU minutes on a generation that would tell
        us nothing new (inference speed cannot differ between image and volume).
        """
        return {
            "ok": True,
            "weights_source": WEIGHTS_SOURCE,
            "load_seconds": round(getattr(self, "load_seconds", -1), 1),
        }

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        import base64
        import shutil
        import subprocess
        import tempfile
        import time
        import uuid
        from pathlib import Path

        import librosa
        import numpy as np
        import pyloudnorm as pyln
        import requests as req
        import torch
        from PIL import Image

        from src.utils import save_videos_grid
        from src.cache_utils import get_teacache_coefficients

        start_time = time.time()

        image_url = request.get("image_url")
        image_base64 = request.get("image_base64")
        audio_url = request.get("audio_url")
        audio_base64 = request.get("audio_base64")

        if not image_url and not image_base64:
            return {"error": "Missing image_url or image_base64"}
        if not audio_url and not audio_base64:
            return {"error": "Missing audio_url or audio_base64"}

        prompt = request.get("prompt") or "A person is speaking."
        steps = int(request.get("steps", 8))
        seg_length = int(request.get("video_length", 81))
        overlap = int(request.get("overlap", 8))
        anchor_retreat = max(0, int(request.get("anchor_retreat", 6)))
        sample_size = request.get("sample_size") or [768, 768]
        guidance_scale = float(request.get("guidance_scale", 6.0))
        audio_guidance_scale = float(request.get("audio_guidance_scale", 3.0))
        audio_scale = float(request.get("audio_scale", 1.0))
        shift = float(request.get("shift", 5.0))
        seed = int(request.get("seed", 43))
        fps = int(request.get("fps", 25))
        teacache_threshold = float(request.get("teacache_threshold", 0.1))
        which_wav2vec = request.get("wav2vec", "chinese")
        negative_prompt = request.get("negative_prompt") or (
            "Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. "
            "Bad fingers. Unclear and blurry hands."
        )
        r2_config = request.get("r2")

        if which_wav2vec not in self.audio_encoders:
            return {"error": f"wav2vec must be one of {list(self.audio_encoders)}"}

        work_dir = Path(tempfile.mkdtemp(prefix="modal_echomimic3_"))

        try:
            image_path = work_dir / "input_image.png"
            if image_url:
                resp = req.get(image_url, stream=True, timeout=300)
                resp.raise_for_status()
                with open(image_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
            else:
                data = image_base64.split(",", 1)[-1]
                image_path.write_bytes(base64.b64decode(data))

            audio_path = work_dir / "input_audio.wav"
            if audio_url:
                resp = req.get(audio_url, stream=True, timeout=300)
                resp.raise_for_status()
                with open(audio_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
            else:
                data = audio_base64.split(",", 1)[-1]
                audio_path.write_bytes(base64.b64decode(data))

            ref_image = Image.open(image_path).convert("RGB")
            height, width = self._fit_size(ref_image, sample_size)

            wav, sr = librosa.load(str(audio_path), sr=16000)
            total_duration = len(wav) / sr
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            if abs(loudness) <= 100:
                wav = pyln.normalize.loudness(wav, loudness, -23)

            total_frames = self._round_frames(int(total_duration * fps))
            seg_length = self._round_frames(seg_length)

            # Checked after rounding, which can pull seg_length down onto overlap.
            # overlap must be >= 1: continuation segments are anchored on those
            # frames, and `tensor[:, :, -0:]` selects the whole tensor rather than
            # nothing, so a zero would silently blend over the entire clip.
            if not 1 <= overlap < seg_length:
                return {"error": f"overlap ({overlap}) must be >= 1 and < video_length ({seg_length})"}

            # A rounded-up final segment asks for audio past the end of the clip.
            # Wav2Vec resamples whatever it is given to seq_len, so a short tail
            # would be stretched and drift out of sync -- pad with silence instead.
            padded_frames = total_frames + seg_length
            wanted_samples = int(padded_frames / fps * sr)
            if len(wav) < wanted_samples:
                wav = np.pad(wav, (0, wanted_samples - len(wav)))

            print(
                f"Audio {total_duration:.1f}s -> {total_frames} frames @ {fps}fps, "
                f"output {width}x{height}, {steps} steps, segment={seg_length}, overlap={overlap}"
            )

            if teacache_threshold > 0:
                coefficients = get_teacache_coefficients(self.model_name)
                if coefficients is not None:
                    self.pipeline.transformer.enable_teacache(
                        coefficients, steps, teacache_threshold, num_skip_start_steps=5, offload=False
                    )

            generator = torch.Generator(device=self.device).manual_seed(seed)
            mix_ratio = torch.linspace(0, 1, steps=overlap).view(1, 1, -1, 1, 1)

            accumulated = None
            starts = ref_image
            produced = 0
            segments = 0

            with torch.no_grad():
                while produced < total_frames:
                    # A continuation segment re-generates the `overlap` frames it
                    # is anchored on, so it has to be that much longer to still
                    # advance by the frames actually remaining.
                    remaining = total_frames - produced
                    want = remaining if accumulated is None else remaining + overlap
                    seg_frames = min(seg_length, self._round_frames(want, up=True))
                    if seg_frames - (0 if accumulated is None else overlap) < 1:
                        break

                    seg_start = produced if accumulated is None else produced - overlap
                    audio_embeds = self._audio_embed(
                        wav, seg_start, seg_frames, fps, which_wav2vec
                    ).to(device=self.device, dtype=self.dtype)

                    input_video, input_video_mask, clip_image = self._build_inputs(
                        starts, seg_frames, height, width
                    )

                    seg_started = time.time()
                    sample = self.pipeline(
                        prompt,
                        num_frames=seg_frames,
                        negative_prompt=negative_prompt,
                        audio_embeds=audio_embeds,
                        audio_scale=audio_scale,
                        ip_mask=None,
                        use_un_ip_mask=False,
                        height=height,
                        width=width,
                        generator=generator,
                        neg_scale=1.0,
                        neg_steps=0,
                        use_dynamic_cfg=False,
                        use_dynamic_acfg=False,
                        guidance_scale=guidance_scale,
                        audio_guidance_scale=audio_guidance_scale,
                        num_inference_steps=steps,
                        video=input_video,
                        mask_video=input_video_mask,
                        clip_image=clip_image,
                        cfg_skip_ratio=0.0,
                        shift=shift,
                    ).videos

                    segments += 1
                    print(
                        f"  segment {segments}: frames {seg_start}-{seg_start + seg_frames} "
                        f"in {time.time() - seg_started:.0f}s"
                    )

                    if accumulated is None:
                        accumulated = sample
                    else:
                        # Cross-fade the re-generated overlap so the seam between
                        # segments does not pop.
                        accumulated[:, :, -overlap:] = (
                            accumulated[:, :, -overlap:] * (1 - mix_ratio)
                            + sample[:, :, :overlap] * mix_ratio
                        )
                        accumulated = torch.cat([accumulated, sample[:, :, overlap:]], dim=2)

                    produced = accumulated.shape[2]
                    if produced >= total_frames:
                        break

                    # Drop a few trailing frames if they hold a blink or another
                    # transient, so the next segment does not start from it and
                    # latch it. Costs the dropped frames, which get regenerated.
                    retreat = self._pick_anchor_retreat(
                        accumulated, overlap, anchor_retreat
                    )
                    if retreat:
                        accumulated = accumulated[:, :, : produced - retreat]
                        produced = accumulated.shape[2]
                        print(f"    re-anchored {retreat} frames back (transient at seam)")

                    starts = [
                        Image.fromarray(
                            (accumulated[0, :, i].permute(1, 2, 0) * 255)
                            .clamp(0, 255)
                            .numpy()
                            .astype(np.uint8)
                        )
                        for i in range(-overlap, 0)
                    ]

            if accumulated is None:
                return {"error": "No frames generated (audio too short?)"}

            silent = work_dir / "silent.mp4"
            save_videos_grid(accumulated[:, :, :total_frames], str(silent), fps=fps)

            final_video = work_dir / "final.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(silent), "-i", str(audio_path),
                 "-c:v", "copy", "-c:a", "aac", "-shortest", str(final_video)],
                capture_output=True, timeout=300, check=True,
            )

            elapsed = time.time() - start_time
            realtime_factor = elapsed / total_duration if total_duration else 0
            print(f"Done: {elapsed:.1f}s for {total_duration:.1f}s of video "
                  f"({realtime_factor:.1f}x realtime), {segments} segments")

            result = {
                "success": True,
                "duration_seconds": round(total_duration, 2),
                "segments": segments,
                "width": width,
                "height": height,
                "steps": steps,
                "processing_time_seconds": round(elapsed, 2),
                "realtime_factor": round(realtime_factor, 1),
            }

            if r2_config:
                import boto3
                from botocore.config import Config

                client = boto3.client(
                    "s3",
                    endpoint_url=r2_config["endpoint_url"],
                    aws_access_key_id=r2_config["access_key_id"],
                    aws_secret_access_key=r2_config["secret_access_key"],
                    config=Config(signature_version="s3v4"),
                )
                object_key = f"echomimic3/results/{uuid.uuid4().hex[:12]}.mp4"
                client.upload_file(
                    str(final_video), r2_config["bucket_name"], object_key,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
                result["video_url"] = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
                    ExpiresIn=7200,
                )
                result["r2_key"] = object_key
            else:
                result["video_base64"] = base64.b64encode(final_video.read_bytes()).decode("utf-8")
                print("Warning: Returning video as base64 (use R2 for large files)")

            return result

        except torch.cuda.OutOfMemoryError:
            return {
                "error": "CUDA OOM. Lower sample_size (e.g. [576, 576]) or "
                         "video_length (e.g. 65), or redeploy the app on L40S."
            }
        except subprocess.CalledProcessError as e:
            return {"error": f"ffmpeg mux failed: {e.stderr[-300:] if e.stderr else e}"}
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            return {"error": f"Internal error: {e}"}
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

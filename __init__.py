import fiftyone as fo
from datetime import datetime
import fiftyone.operators as foo
from fiftyone.operators import types
import uuid
import time
import requests
import glob
from pprint import pprint
import os
import fiftyone.core.utils as fou
from importlib.util import find_spec

SD_MODEL_URL = "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"

SD_SIZING_STRATEGIES = (
                        "maintain_aspect_ratio",
                        "crop_to_16_9",
                        "use_image_dimensions"
)

SD_VIDEO_LENGTH = (
                    "14_frames_with_svd",
                    "25_frames_with_svd_xt"
)

replicate = fou.lazy_import("replicate")


def allows_replicate_models():
    """Returns whether the current environment allows replicate models."""
    return (
        find_spec("replicate") is not None
        and "REPLICATE_API_TOKEN" in os.environ
    )

class Image2Video:
    """Wrapper for a Image2Video model."""

    def __init__(self):
        self.name = None
        self.model_name = None

    def generate_video(self, ctx):
        pass

class StableDiffusion(Image2Video):
    """Wrapper for a StableDiffusion model."""

    def __init__(self):
        super().__init__()
        self.name = "stable-diffusion"
        self.model_name = SD_MODEL_URL

    def generate_video(self, ctx, dataset):
        target = ctx.params.get("target", None)
        target_view = _get_target_view(ctx, target)
        for sample in target_view:
            image = sample.filepath

            video_length = ctx.params.get("length_choices", "None provided")
            sizing_strategy = ctx.params.get("width_choices", "None provided")
            frames_per_second = ctx.params.get("frames_per_second", "None provided")
            motion_bucket_id = ctx.params.get("motion_bucket_id", "None provided")
            cond_aug = ctx.params.get("cond_aug", "None provided")
            decoding_t = ctx.params.get("decoding_t", "None provided")
            seed = ctx.params.get("seed", "None provided")

            response = replicate.run(
                self.model_name,
                input={
                    "input_image": open(image, "rb"),
                    "video_length": video_length,
                    "sizing_strategy": sizing_strategy,
                    "frames_per_second": frames_per_second,
                    "motion_bucket_id": motion_bucket_id,
                    "cond_aug": cond_aug,
                    "decoding_t": decoding_t,
                    "seed": seed,
                },
            )
            if type(response) == list:
                response = response[0]

            image_url = response

            filepath = generate_filepath(ctx)
            download_image(image_url, filepath)

            sample = fo.Sample(
                filepath=filepath,
                tags=["generated"],
                model=self.name,
                original_image_filepath=image,
                date_created=datetime.now(),
            )


            dataset.add_sample(sample, dynamic=True)
        return 
    

def get_model(model_name):
    mapping = {
        "sd": StableDiffusion,
    }
    return mapping[model_name]()

def download_image(image_url, filename):
    img_data = requests.get(image_url).content
    with open(filename, "wb") as handler:
        handler.write(img_data)

def generate_filepath(ctx):
    download_dir = ctx.params.get("download_dir", {})
    if type(download_dir) == dict:
        download_dir = download_dir.get("absolute_path", "/tmp")

    filename = str(uuid.uuid4())[:13].replace("-", "") + ".mp4"
    return os.path.join(download_dir, filename)


#### MODEL CHOICES ####
def _add_replicate_choices(model_choices):
    model_choices.add_choice("sd", label="Stable Diffusion")


#### STABLE DIFFUSION INPUTS ####
def _handle_stable_diffusion_input(ctx, inputs):
    size_choices = SD_SIZING_STRATEGIES
    width_choices = types.Dropdown(label="Sizing Strategy")
    for size in size_choices:
        width_choices.add_choice(size, label=size)

    inputs.enum(
        "width_choices",
        width_choices.values(),
        default="maintain_aspect_ratio",
        view=width_choices,
    )

    len_choices = SD_VIDEO_LENGTH
    length_choices = types.Dropdown(label="Length")
    for len in len_choices:
        length_choices.add_choice(len, label=len)

    inputs.enum(
        "length_choices",
        length_choices.values(),
        default="14_frames_with_svd",
        view=length_choices,
    )

    fps_slider = types.SliderView(
        label="Frames Per Second",
        componentsProps={"slider": {"min": 1, "max": 120, "step": 1}},
    )
    inputs.int("frames_per_second", default=6, view=fps_slider)

    motion_slider = types.SliderView(
        label="Increase overall motion in the generated video",
        componentsProps={"slider": {"min": 1, "max": 500, "step": 1}},
    )
    inputs.int("motion_bucket_id", default=127, view=motion_slider)

    noise_slider = types.SliderView(
        label="Amount of noise to add to input image",
        componentsProps={"slider": {"min": 0, "max": 1, "step": 0.01}},
    )
    inputs.float("cond_aug", default=0.02, view=noise_slider)

    decoding_slider = types.SliderView(
        label="Number of frames to decode at a time",
        componentsProps={"slider": {"min": 0, "max": 120, "step": 1}},
    )
    inputs.int("decoding_t", default=14, view=decoding_slider)

    seed_slider = types.SliderView(
        label="Choose seed to randomize from",
        componentsProps={"slider": {"min": 1, "max": 1000, "step": 1}},
    )
    inputs.int("seed", default=51, view=seed_slider)


INPUT_MAPPER = {
    "sd": _handle_stable_diffusion_input,
}

def _resolve_download_dir(ctx, inputs):
    if len(ctx.dataset) == 0:
        file_explorer = types.FileExplorerView(
            choose_dir=True,
            button_label="Choose a directory...",
        )
        inputs.file(
            "download_dir",
            required=True,
            description="Choose a location to store downloaded images",
            view=file_explorer,
        )
    else:
        base_dir = os.path.dirname(ctx.dataset.first().filepath).split("/")[
            :-1
        ]
        ctx.params["download_dir"] = "/".join(base_dir)


def _handle_input(ctx, inputs):
    model_name = ctx.params.get("model_choices", "sd")
    model_input_handler = INPUT_MAPPER[model_name]
    model_input_handler(ctx, inputs)


class Image2Video(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="img2video",
            label="Create videos with GenAI from image",
            dynamic=True,
        )
        _config.icon = "/assets/video.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        _resolve_download_dir(ctx, inputs)

        replicate_flag = allows_replicate_models()
        if not replicate_flag:
            inputs.message(
                "message",
                label="No models available. Please set up your environment variables.",
            )
            return types.Property(inputs)

        model_choices = types.Dropdown()
        if replicate_flag:
            _add_replicate_choices(model_choices)
        inputs.enum(
            "model_choices",
            model_choices.values(),
            default=model_choices.choices[0].value,
            label="Model",
            description="Choose a model to generate images",
            view=model_choices,
        )
        target_view = get_target_view(ctx, inputs)

        _handle_input(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        target = ctx.params.get("target", None)
        target_view = _get_target_view(ctx, target)
        if "image2video"  in fo.list_datasets():
            dataset = fo.load_dataset("image2video")
        else:
            dataset = fo.Dataset("img2video")
            dataset.persistent = True
        
        model_name = ctx.params.get("model_choices", "None provided")
        model = get_model(model_name)
        model.generate_video(ctx,dataset)

        
    

        if dataset.get_dynamic_field_schema() is not None:
            dataset.add_dynamic_sample_fields()
            ctx.trigger("reload_dataset")
        else:
            ctx.trigger("reload_samples")


def register(plugin):
    plugin.register(Image2Video)

def get_target_view(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = None

    if has_view or has_selected:
        target_choices = types.RadioGroup(orientation="horizontal")
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Process the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Process the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Process only the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            required=True,
            label="Target view",
            view=target_choices,
        )

    target = ctx.params.get("target", default_target)

    return _get_target_view(ctx, target)

def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view
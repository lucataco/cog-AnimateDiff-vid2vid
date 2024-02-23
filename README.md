# Cog AnimateDiff-vid2vid Model

This is an implementation of [AnimateDiff-vid2vid](https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff#animatediffvideotovideopipeline) as a [Cog](https://github.com/replicate/cog) model to produce a stereo video.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

Run a prediction:

    cog predict -i video=@demo.gif

## Output

![output](output.gif)
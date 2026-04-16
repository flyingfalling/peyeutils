## REV: computes saliency of gaze target and random controls.
##      random controls is simply mean density of gaze of subject during any video (easier
##      than any-but-this-video), i.e. looking prior.

## Downsample to frame-rate of video (mean?). Exclude saccs/blinks.
## Do some 3d tensor of frame over time, and sample from X/Y pos inside it. Convolution with something perhaps.
## Just use null (prior) distribution, and multiply by saliency to get value.



## Also: chunk/export individual trials as CSVs (easier). Or e.g. npy or etc.
## Also: do full pairwise correlation, and plot individual trials all together (as single-row).


## Decoding/Encoding based no only on saliency, but based on Deep CNN (e.g. alexnet), on same inputs.

def main():
    return 0;


### Some notes for myself

At first we are going to test our pipeline
in an image to make sure we are correctly
Identifying the lane lines.

The first step is to retrieve relevant data from
the image using some kind of feature extraction.

I've tried visualizing the images in several color schemas
And the ones where the lane lines are most visible are
in RGB - R and G channels and HSL - S and L channels
Although, I'm not sure wether I can rely safely on the L channel
for feature extraction, because the lightness changes very much
in adverse scenarios and I don't have a lot of images to test on.

It would be preferable if I had some night images as well, just to
be sure of wether or not I should make use of the L channel for feature
extraction.

S channel has been the most reliable this far, it seems to make
most of the lane lines in the image visible, but not all of them
I need to find some other way of getting more information
Applying Sobel operator on the S channel has brought me some good
results.

Maybe trying to filter out non-yellow and non-white sections of the
image can bring me good results as well.

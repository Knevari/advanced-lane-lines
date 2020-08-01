# Advanced Lanes Detection

The challenge in this project is to precisely detect lane lines in a video stream and extract subtle information about the car's current state, there are several steps we need to walk through in order to achieve the desired results, and we're gonna explore some of them in this article.

### Detecting lane lines

It's a trivial task for humans to recognize lane lines whilst driving. But for an autonomous car it is not an easy task, there are a lot of details we need to pay attention while implementing an algorithm to detect lane lines and several things we need to ask ourselves while programming it, for instance, what would be the best way of abstracting away all the unnecessary details of an image and keep just what we're looking for? Is there some deceptive noise or too much information? It's hard to keep track of these things when our data has a high variance and the context of what we need to track is broader, for example, what if the algorithm works well on bright images, but performs poorly during the night? This is a scenario we can't face while working with self-driving cars, because lives can be at stake.

In the first project, we utilized [Canny Edges Detection](https://en.wikipedia.org/wiki/Canny_edge_detector#:~:text=The%20Canny%20edge%20detector%20is,explaining%20why%20the%20technique%20works.) and [Probabilistic Hough Transform](https://en.wikipedia.org/wiki/Hough_transform#:~:text=The%20Hough%20transform%20is%20a,shapes%20by%20a%20voting%20procedure.) to detect lines in parameter space and draw them on road images, it works well in some situations, but it is not a reliable algorithm because it can't detect steep curves very well, in this project we are going to make use of [Sobel Operator](https://en.wikipedia.org/wiki/Sobel_operator) to extract only relevant edges and build our way up to a better feature extraction pipeline. One of the reasons we are not using Canny here is because it is "way too good" for the purpose of the program, to summarize, canny gets more detail than we actually want to retrieve from the image, with Sobel, we have more control of what edges we want to find, since most of the lane lines are almost vertical, it makes sense to search for horizontal edges and not otherwise.

##### Sobel Operator

Sobel is an algorithm to emphasize edges over a direction getting an approximation to the derivative on an image, it runs a odd sized kernel over the image pixels and return the summation of the element-wise product between the kernel and the pixels surrounding the pixel.

<p align="center" style="display: flex; justify-content: center; align-items: center">
    <img src="/github_examples/straight_lines1.jpg" width=250 height=200 alt="Original Image"
    	title="Original Image" />
    <img src="/github_examples/sobel_x.jpg" width=250 height=200 alt="Sobel X"
    	title="Sobel applied in X direction" />
    <img src="/github_examples/sobel_x.jpg" width=250 height=200 alt="Sobel Y"
    	title="Sobel applied in Y direction" />
</p>

Here is an [excellent video](https://www.youtube.com/watch?v=sRFM5IEqR2w) explaining how it works

### Pipeline

One of the most important steps in our algorithm is the pipeline, the part of the program responsible for removing unnecessary details and extracting useful information for us to work with. there are several ways of building a good pipeline and in this project I tried a **lot** of different ways to achieve the best result. It is important to notice that our desired state is the one where we abstract away noise and unnecessary information from the image, it is not essential lefting out only the lane lines, because the algorithm can deal relatively well with noisy data and we are working only with a small section of the image where lane lines are expected to be, but we should do our best to tune the parameters and find the best combination of steps to build a reliable pipeline

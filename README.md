# Advanced Lanes Detection

The challenge in this project is to precisely detect lane lines in a video stream and extract subtle information about the car's current state, there are several steps we need to walk through in order to achieve the desired results, and we're gonna explore some of them in this article.

### Detecting lane lines

It's a trivial task for humans to recognize lane lines whilst driving. But for an autonomous car it is not an easy task, there are a lot of details we need to pay attention while implementing an algorithm to detect lane lines and several things we need to ask ourselves while programming it, for instance, what would be the best way of abstracting away all the unnecessary details of an image and keep just what we're looking for? Is there some deceptive noise or too much information? It's hard to keep track of these things when our data has a high variance and the context of what we need to track is broader, for example, what if the algorithm works well on bright images, but performs poorly during the night? This is a scenario we can't face while working with self-driving cars, because lives can be at stake.

In the first project, we utilized [Canny Edges Detection](https://en.wikipedia.org/wiki/Canny_edge_detector#:~:text=The%20Canny%20edge%20detector%20is,explaining%20why%20the%20technique%20works.) and [Probabilistic Hough Transform](https://en.wikipedia.org/wiki/Hough_transform#:~:text=The%20Hough%20transform%20is%20a,shapes%20by%20a%20voting%20procedure.) to detect lines in parameter space and draw them on road images, it works well in some situations, but it is not a reliable algorithm because it can't detect steep curves very well, in this project we are going to make use of [Sobel Operator](https://en.wikipedia.org/wiki/Sobel_operator) to extract only relevant edges and build our way up to a better feature extraction pipeline. One of the reasons we are not using Canny here is because it is "way too good" for the purpose of the program, to summarize, canny gets more detail than we actually want to retrieve from the image, with Sobel, we have more control of what edges we want to find, since most of the lane lines are almost vertical, it makes sense to search for horizontal edges and not otherwise.

##### Sobel Operator

Sobel is an algorithm to emphasize edges over a direction getting an approximation to the derivative on an image, it runs a odd sized kernel over the image pixels and return the summation of the element-wise product between the kernel and the pixels surrounding the pixel.

<table>
  <tr>
    <td>Original Image</td>
    <td>Sobel applied in X direction</td>
    <td>Sobel applied in Y direction</td>
  </tr>
  <tr>
    <td valign="top"><img title="Original Image" alt="Original Image" src="/github_examples/straight_lines1.jpg" width=300 height=250></td>
    <td valign="top"><img title="Sobel applied in X direction" alt="Sobel X" src="/github_examples/sobel_x.jpg" width=300 height=250></td>
    <td valign="top"><img title="Sobel applied in Y direction" alt="Sobel Y" src="/github_examples/sobel_y.jpg" width=300 height=250></td>
  </tr>
 </table>

Here is an [excellent video](https://www.youtube.com/watch?v=sRFM5IEqR2w) explaining how it works

### Pipeline

One of the most important steps in our algorithm is the pipeline, the part of the program responsible for removing unnecessary details and extracting useful information for us to work with. there are several ways of building a good pipeline, in this project I tried a lot of different ways to achieve the best result. It is important to notice that our desired state is the one where we abstract away noise and unnecessary information from the image, it is not essential lefting out only the lane lines, because the algorithm can deal relatively well with noisy data and we are working only with a small section of the image where lane lines are expected to be, but we should do our best to tune the parameters and find the best combination of steps to build a reliable pipeline.

We are only interested in edges of a particular orientation, because lane lines are mostly close to vertical, so we can filter out edges by their direction, that is simply the inverse tangent of the y gradient divided by the x gradient:

![Grad Dir](https://latex.codecogs.com/svg.latex?arctan(Sy%20/%20Sx))

And can be implemented with numpy's arctan2 function. Another useful way of retrieving useful data is finding the magnitude of the gradient, which is given by:

![Mag Dir](https://latex.codecogs.com/svg.latex?\sqrt(Sx%20^2%20+%20Sy%20^2)) 

And can help us filter some noise by removing the weaker edges from our output. In the next example, we are going to implement those methods and see the combined result of both.

```python
import cv2
import numpy as np

def sobel(image, orient="x", ksize=3):
    if orient == "x":
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    return np.absolute(sobel)

def gradient_direction(gray):
    # Calculate Sobel for X and Y axis
    sobel_x = sobel(gray, orient="x")
    sobel_y = sobel(gray, orient="y")
    
    # Calculate the gradient direction
    grad_dir = np.arctan2(sobel_y, sobel_x)
    
    # Filter edges by direction
    dir_binary = np.zeros_like(gray)
    dir_binary[(grad_dir >= np.pi/6) & (grad_dir < np.pi/2)] = 1
    return dir_binary

def gradient_magnitude(gray):
    # Calculate Sobel for X and Y axis
    sobel_x = sobel(gray, orient="x")
    sobel_y = sobel(gray, orient="y")
    
    # Calculate the gradient magnitude
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    # Filter edges by strength
    mag_binary = np.zeros_like(gray)
    mag_binary[(grad_mag >= 20) & (grad_mag < 100)] = 1
    return mag_binary

image = cv2.imread("some_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

direction = gradient_direction(gray)
magnitude = gradient_magnitude(gray)

combined = np.zeros_like(gray)
combined[(direction == 1) & (magnitude == 1)] = 1
combined *= 255

cv2.imshow("Combined Thresholds", combined)
```

![Combined Mag Dir](/github_examples/combined_mag_dir.jpg "Combined Magnitude and Direction")

As you can see, there is a lot of noise, but as I said before, it is not essential to boil out all the unnecessary details to have a reliable prediction, the problem with this approach is that it was taking too much time to calculate the direction of the gradient and just calling np.arctan2 was making the algorithm take a lot more time to process, so I had to try another approach that didn't make use of the direction of the edges. I tried applying color and lighting thresholding, but as expected, it didn't work very well with varying inputs, my final approach was making use of HLS - L and S channels, applying a gaussian blur of kernel size equals 3 in the beginning of the pipeline and getting the threshold of both L and S channels which gave me a pretty decent output and was a lot faster to process.
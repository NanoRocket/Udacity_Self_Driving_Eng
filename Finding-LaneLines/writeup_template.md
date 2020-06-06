# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of several steps done in turn. All depended heavily on the provided helper functions.

* First, I converted the images to grayscale, then I applied a gaussian blur. This filtered noise and sharp gradients, and provided the cv2 functions with the required format.
* I then applied the Canny Edge detection to find edges in the image. The tuning for these parameters was done using the GUI linked in the lesson.
* This was followed by a masking that only highlighted edges near where the lane lines should be in the image. For speed I really feel like this should have been one of the first things I did (why perform the above operations if we can just throw out part of the image), but I did not change it.
* This was followed by using the Hough transform to identify the straight lines in the image. This also utilized a gui to try and optimize the parameters above. I do not think it was particulary effective however, as finding a good spot between robustness and accuracy was difficult. Moreover, there parameter space is multidimensional, which made tuning by eye unsystematic.
* I finished out by using the weighted function to meld the lane lines and an unchanged image together.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by using 4 lists, that would give point coordinates in the left lane and the right lane. These lists would be appended with the appropriate coordinates based on the slope of the line.

Once the lists in the image were sorted, I then got the equations for a linear fit line using the Numpy polyfit function. I then placed these line in the image based on their x intercepts, the bottom of the image, and a hardcoded 60 pixels below the middle of the image. I toyed with the idea of using the max y value detected in each line (so that only interpolation occured instead of extrapolation), but this seemed to constantly limit the annotation. I do believe such a feature could be integral in maintaining line stability, such as using a previous measurement if the current measurements has too small an interpolated line length.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is often the lines would demonstrate large instability. A single measurement could throw the line from the lane to nearly horizontal, simply because my pipeline has no weighting or averaging of past results. I expected such a failing, but currently do not have the time to correct it.

Another shortcoming of my pipeline was this weird tendency for red and green to show up in the image very well, but for the blue to be almost completely invisible. This is likely due to some weird interplay between the pixel coloring of the images, using CV2 to import them, and outputing them. I spent a number of hours trying to fix this problem to no avail.

On another aspect of color, I feel like I should be using the fact that we are looking for yellow and white markings to our advantage in some way. I did not use this in my pipeline.

Finally, the absolute largest shortcoming is the inability to handle curves. Again, to be expected, given that I am only fitting lines. Further, I doubt this detector could handle different lighting conditions, or objects lying on the road. Essentially, there is no helping 'internal logic' that tells the pipeline "this is not a lane line". All it sees are all the lines in the Canny image.


### 3. Suggest possible improvements to your pipeline

I would improve the pipeline by:

* Integrating some sort of stabilization for the fitted lines themselves. Either through throwing out lines that do not closely match what come before (filtering), or, adding a continuous averaging mechanism so that outliers could occur, but only slightly change the trajectory. I would likely incorporate weighting into that based on the gradient of change.

* We could integrate logic into the Hough helper function. Throw out hough lines that don't seem physical.

* Make the fitting nonlinear, or more likley try a nonlinear fit when needed using some logic.

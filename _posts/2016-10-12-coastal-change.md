---
layout: post
title: Monitoring and measuring coastal change
---

Coastal change is an accelerating global phenomenon that is mainly attributed to climate change, extreme weather and powerful sea currents. It has the potential to [threaten communities](http://www.nytimes.com/2016/09/04/science/flooding-of-coast-caused-by-global-warming-has-already-begun.html?_r=0)
and [local economies](https://www.bostonglobe.com/lifestyle/real-estate/2014/09/13/climate-change-concerns-weigh-cape-home-buying-decisions/SnTafe2lwWiOsLt5AtezHK/story.html)
of coastal towns and cities.
Accurate monitoring of coastal change could offer valuable insight into climate change and provide input to climate models, as well as
help in [flooding disaster preparedness and mitigation](https://msc.fema.gov/portal).   

We developed an end-to-end GBDX workflow for coastal change detection and measurement at the **native resolution** of our 8-band multispectral imagery. All you need to run the workflow is a GBDX account and Python. The workflow can measure coastal changes **at the meter level** and **at unprecedented scale**.

The example we cover here focuses on Cape Cod, an area which is well known
[for extreme changes in the coastal landscape](http://earthobservatory.nasa.gov/Features/WorldOfChange/cape_cod.php). The workflow takes two roughly collocated images of Cape Cod, captured in 2010 and 2016 by WorldView-2 and WorldView-3, respectively, and computes coastal change on the entire images, roughly an area of 1500km2 in **less than 30 minutes**.

![header3.png]({{ site.baseurl }}/images/coastal-change/header3.png)


# Contents
1. [Imagery](#imagery)
2. [Workflow](#workflow)
  - [Image pair alignment](#image-pair-alignment)
  - [Water mask](#water-mask)
  - [Exclusion mask](#exclusion-mask)
  - [Binary change detection](#binary-change-detection)
3. [Measuring change](#measuring-change)
4. [Running the workflow](#running-the-workflow)
5. [Discussion](#discussion)


# Imagery

We've picked two images of Cape Cod (MA); the details are shown in Table 1.
Click on each catalog id to view the imagery.

| **field** | **image 1** | **image 2** |
| :-------- |:-------- | :-------- |
| catID | [1030010004C45A00]({{ site.baseurl }}/pages/coastal-change/pre.html) | [104001001E9EF700]({{ site.baseurl }}/pages/coastal-change/post.html)
date | 4/15/2010 | 6/27/2016 |
satellite | [WorldView-2](https://www.digitalglobe.com/about/our-constellation)  | [WorldView-3](https://www.digitalglobe.com/about/our-constellation) |
off nadir angle | 12 deg | 16 deg
cloud cover | 0.0% | 0.0%

*Table 1. Image attributes.*

Note that this workflow requires imagery from either WV2 or WV3; it is only these two satellites that provide high-resolution (< 2m) 8-band multispectral imagery necessary for selected tasks in this exercise. The image acquisition dates differ by 6 years and 2 months. They were both acquired from similar angles during spring/early summer, and contain no clouds. Coming from different sensors they are of different spatial resolution and they overlap for most of their extent. Figure 1 shows the image bounding boxes projected on Google Earth; the red frame marks the intersection.

![CapCopAOIs.png]({{ site.baseurl }}/images/coastal-change/CapCopAOIs.png)
*Figure 1. Image bounding boxes and intersection marked in red.*

We ordered the two images, ran them through the [Advanced Image Preprocessor](http://gbdxdocs.digitalglobe.com/docs/advanced-image-preprocessor) and stored them in S3 - see Figure 2.

![AOP.png]({{ site.baseurl }}/images/coastal-change/AOP.png)
*Figure 2. Image preprocessing.*

What preprocessing involves in this case is ortho-rectification, projection to UTM coordinates (good for accurate distance calculations but not necessary for simple visual inspection of the results) and atmospheric compensation. If you want to order and preprocess the images yourself, you can run the following script in python. Otherwise, skip this and go directly to the next section.

```python
import gbdxtools
from os.path import join
import random
import string

# create a gbdx interface
gbdx = gbdxtools.Interface()

# specify catalog ids
catalog_ids = ['1030010004C45A00', '104001001E9EF700']

# create order task
# it the images are not on GBDX, this task will order them from the DG factory
order = gbdx.Task('Auto_Ordering')
# pass both catalog ids as inputs - this will launch a batch workflow
order.inputs.cat_id = catalog_ids
# for this particular task, we need to set this flag to true
order.impersonation_allowed = True

# create preprocessing task and set input parameters
aop = gbdx.Task('AOP_Strip_Processor')
aop.inputs.data = order.outputs.s3_location.value
aop.inputs.bands = 'MS'
aop.inputs.enable_dra = False
aop.inputs.enable_pansharpen = False
aop.inputs.enable_acomp = True
aop.inputs.enable_tiling = False
aop.inputs.ortho_epsg = 'UTM'     # this setting is optional

# create two-task preprocessing workflow
preprocess_wf = gbdx.Workflow([order, aop])

# set output location to platform-stories/trial-runs/random_str within your bucket/prefix
random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_location = join('platform-stories/trial-runs', random_str)

# the two processed images will be stored under output_location
preprocess_wf.savedata(aop.outputs.data, output_location)

# execute
preprocess_wf.execute()

# monitor status
preprocess_wf.status
```

AOP_Strip_Processor usually takes about an hour to process an entire strip.
If the images have not been previously ordered, the preprocessing workflow will take
longer to complete.


# Workflow

Our workflow is shown in Figure 3. Green boxes correspond to GBDX tasks and blue boxes to images or information layers. image1 and image2 correspond to the post- and pre- image. However, keep in mind that the workflow is symmetric with respect to the chronological order and image1 and image2 can be interchanged.

![cd.png]({{ site.baseurl }}/images/coastal-change/cd.png)
*Figure 3. The coastline change detection workflow.*

The workflow starts with the Image Pair Alignment (IPA) task which takes as input two AOP-ready images, computes their geographic intersection and returns two new images, each containing the subset of the original that lies within the intersection.
After successful conclusion of the IPA task a water mask (WM) is computed from each IPA-ready image. A water mask is a binary image in which white or 'foreground' pixels correspond to automatically detected water in the original, and black or 'background' pixels to everything else. The two images are also used to compute a single exclusion mask (EM). The EM is a the sum of all image regions that might generate a change with respect to the target class (water in this case) that is **irrelevant** to the workflow, such as cloud cover and no-data zones. The point of the EM is to remove these regions from the final change detection map.

The three masks are fed to three different Binary Change Detection (BCD) tasks, GBCD, LBCD and TBCD.
GBCD and LBCD produce binary images that show the water gain and loss, respectively. TBCD delivers a tristate image that shows water gain in green, water loss in read and no change regions in white (and black corresponding to regions masked out by the EM).

In many cases, we need to produce semantic layers, e.g., contours of coastline change, or statistics such as the change mean or variance. For this purpose, we have created two additional tasks (see Figure 4) which use the water gain and loss maps
to produce **change detection distance maps**. See [this section](#change-detection-distance-map) for details.   

![ddt.png]({{ site.baseurl }}/images/coastal-change/ddt.png)
*Figure 4. The distance transform tasks of the change maps.*  

The tasks used in the workflow are listed in Table 2.

| **Name** | **Acronym** | **GBDX Task Registry Name** | **Description** |
| :---------- | :----------| :----------| :---------
|Image Pair Alignment | IPA | [protogenV2CD_READY](https://github.com/TDG-Platform/docs/blob/master/protogenV2CD_READY.md) | Image preparation for change detection |
| Water Mask | WM | [protogenV2RAW](https://gbdxdocs.digitalglobe.com/docs/8-band-water-mask)     | Water layer extraction |
| Exclusion Mask| EM |protogenV2CD_LULC | Computes an exclusion mask based on selected land use land cover classes |
| Binary Change Detection; gain | GBCD | protogenV2CD_BIN_GAIN | Computes the change map between two binary input layers and returns the gain with respect to the most recent one |
| Binary Change Detection; loss | LBCD | protogenV2CD_BIN_LOSS | Computes the change map between two binary input layers and returns the loss with respect to the most recent one |
| Binary Change Detection; tristate | TBCD | protogenV2CD_BIN_TRI | Computes the change map between two binary input layers and returns the tristate output: gain in green, loss in red, white for no change, black for irrelevant |
| Gain map Discrete Distance Transform | GDDT | protogenV2CD_GDDT | Computes the distance map of the gain map |
| Loss map Discrete Distance Transform | LDDT | protogenV2CD_LDDT | Computes the distance map of the loss map |

*Table 2. Workflow tasks.*

If you want to find out more about each individual task, read on.

## Image pair alignment

The IPA processor takes as input two AOP-ready images. It verifies that they are in the same coordinate system, it computes their bounding boxes (black dotted lines in Figure 5) and checks if these overlap geographically. If they do not overlap an error is returned and the workflow terminates. Otherwise, the IPA processor computes their geographic intersection that is a bounding box in itself (orange dotted line in Figure 5) and it stores it as a vector.

![step1.png]({{ site.baseurl }}/images/coastal-change/step1.png)
*Figure 5. IPA processor, step 1/4: Computation of the intersection.*

The IPA processor uses the intersection vector to extract the AOI (i.e., the contents) of each image that lies within it, and save it in a new image. If the two AOIs differ in spatial resolution, the high resolution AOI is downsampled using bilinear interpolation to the lower resolution. These steps are shown in Figure 6.

![step2.png]({{ site.baseurl }}/images/coastal-change/step2.png)
*Figure 6. IPA processor, steps 2-4: AOI extraction from each image using the intersection bounding box, and resampling, if necessary, for resolution normalization.*

Here's this first part of the workflow in python:

```python
import gbdxtools
from os.path import join
gbdx = gbdxtools.Interface()

# specify imagery location
input_location = 's3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform-stories/coastal-change/images'

# create ipa task
ipa = gbdx.Task('protogenV2CD_READY')
ipa.inputs.raster = join(input_location, 'post')
ipa.inputs.slave  = join(input_location, 'pre')
```

## Water mask

A WM of an image is a binary image where white (foreground) pixels correspond to water bodies and black (background) pixels
to everything else. The GBDX task which generates WMs is [protogenV2RAW](https://gbdxdocs.digitalglobe.com/docs/8-band-water-mask).
protogenV2RAW is completely **unsupervised**; it uses the image spectral information to locate water bodies.
tively.

```python
# create water mask tasks
# enable raid domain as this task is compute-intensive
water_pre = gbdx.Task('protogenV2RAW')
water_post = gbdx.Task('protogenV2RAW')

# chain rawpre input to cdready slave output
water_pre.inputs.raster = ipa.outputs.slave.value

# chain rawpost input to cdready main output
water_post.inputs.raster = ipa.outputs.data.value
```

Screenshots of the pre and post WMs are shown in Figure 7.

![slide_RAW.png]({{ site.baseurl }}/images/coastal-change/slide_RAW.png)
*Figure 7. Pre and post WMs. White is water and black is land.*


## Exclusion mask

The exclusion mask (EM) is the sum of all image regions from both IPA-ready images, that may generate a change with respect to the foreground class (water) that is irrelevant to the objective. For example, consider a cloud-covered region in one of the two images that is found within a water body. If the second image does not contain the same cloud in the same area, a change will be registered. This change is irrelevant to coastal change and needs to be marked as such.

The EM is a binary image computed from the Land Use Land Cover (LULC) layers of the input images.
In this workflow, the EM consists of the sum (logical OR) of all cloud and no-data zone pixels in both input images. The EM is generated using the protogenV2CD_LULC task which we created for this purpose.

![slide_LULC.png]({{ site.baseurl }}/images/coastal-change/slide_LULC.png)
*Figure 8. The EM task takes as input two IPA-ready images, computes the respective LULC layers internally, identifies the foreground class and constructs the output image using the preselected class blending operator; in this case, logical OR.*

Figure 8 shows the intermediate and output layers of the exclusion mask creation task. Note that on the output layer the two foreground regions at the center of the image are LULC layer errors (classes misinterpreted as clouds). The frame around the image is made of no-data zones accumulated from both input images.

```python
# create exclusion mask task
# enable raid domain as this task is compute-intensive
exclusion_mask = gbdx.Task('protogenV2CD_LULC')
exclusion_mask.domain = 'raid'

# chain inputs to cdready outputs
exclusion_mask.inputs.raster = ipa.outputs.data.value
exclusion_mask.inputs.slave = ipa.outputs.slave.value
```

## Binary change detection

Change between the WMs is detected on a pixel by pixel basis. If the pixels have different values the result is 1, else it is 0 (an XOR operation). The workflow deploys three different BCD tasks; protogenV2CD_BIN_GAIN for water gain detection, protogenV2CD_BIN_LOSS for water loss detection and protogenV2CD_BIN_TRI for water gain/loss/no-change detection. Screenshots of the three outputs are shown in Figure 9. In the tristate image, green, red and white designate water gain, loss, and no-change respectively, while black highlights the regions marked by the EM as irrelevant to water change.

![slide_DG.png]({{ site.baseurl }}/images/coastal-change/slide_CD.png)  
*Figure 9. BCD inputs and outputs.*

```python
# create BCD tasks and chain to water and exclusion masks
bcd_gain = gbdx.Task('protogenV2CD_BIN_GAIN')
bcd_gain.inputs.raster = water_post.outputs.data.value
bcd_gain.inputs.slave  = water_pre.outputs.data.value
bcd_gain.inputs.mask   = exclusion_mask.outputs.data.value

bcd_loss = gbdx.Task('protogenV2CD_BIN_LOSS')
bcd_loss.inputs.raster = water_post.outputs.data.value
bcd_loss.inputs.slave  = water_pre.outputs.data.value
bcd_loss.inputs.mask   = exclusion_mask.outputs.data.value

bcd_tri = gbdx.Task('protogenV2CD_BIN_TRI')
bcd_tri.inputs.raster = water_post.outputs.data.value
bcd_tri.inputs.slave  = water_pre.outputs.data.value
bcd_tri.inputs.mask   = exclusion_mask.outputs.data.value
```

# Measuring change

The water gain and loss change maps permit the visualization of change but do not contain information about its magnitude.
We implemented two GBDX tasks, protogenV2CD_GDDT and protogenV2CD_LDDT, that apply a Discrete Distance Transform (DDT) on the gain and loss change maps. The DDT encodes in the intensity of each pixel in regions that have undergone change, its closest distance from the coastline in the pre image. In other words, the pixel intensity **reflects the degree of change**.

## Measuring water loss

The next figure shows an area of extensive water retreat.

![post_loss_1_ori_pre.png]({{ site.baseurl }}/images/coastal-change/post_loss_1_ori_pre.png)  
![post_loss_1_ori_post.png]({{ site.baseurl }}/images/coastal-change/post_loss_1_ori_post.png)

*Figure 10. Pre (top) and post (bottom); water retreat.*

Figure 11, left, shows the DDT of the loss map. White pixels indicate water which is present in the pre and post. Grayscale pixels
indicate the region of water loss; the brighter the pixel, the larger the distance from the pre-coastline. Black pixels correspond to everything else. For this visualization, we converted the 32-bit output of LDDT to 8-bit which results in intensity saturation. As a result, a part of the outer contour of the DDT appears as white. We experimented with pseudocolor which resulted in the visual shown on the right and shows the true extent of the water retreat.

The maximum pixel intensity is 487; since the resolution is about 2m/pixel, an approximate calculation reveals that the maximum water retreat in this AOI is 974m! Note that the DDT is sensitive to pixel [isotropy](https://en.wikipedia.org/wiki/Isotropy). Consequently, in order for these calculations to be accurate, input images given in lat,long should be converted to UTM.

![post_loss_1_ddt_gray.png]({{ site.baseurl }}/images/coastal-change/post_loss_1_ddt_gray.png)
![post_loss_1_ddt_rgb2b.png]({{ site.baseurl }}/images/coastal-change/post_loss_1_ddt_rgb2b.png)

*Figure 11. The DDT of the loss map (top) and a visual impression in pseudocolor (bottom).*

## Measuring water gain

Figure 12 shows an AOI where the landscape has changed dramatically between pre and post. Water has appeared in place of land and vice versa.

![post_gain_1_ori_pre.png]({{ site.baseurl }}/images/coastal-change/post_gain_1_ori_pre.png)
![post_gain_1_ori_post.png]({{ site.baseurl }}/images/coastal-change/post_gain_1_ori_post.png)

*Figure 12. Pre (top) and post (bottom); water gain and loss.*

Figure 13, left, shows the DDT of the gain map. The two grayscale ribbons correspond to areas where water has appeared in place of land; the pixel intensity captures the degree to which the water has advanced to reach that very point. A visual impression in pseudocolor is shown on the right. The dark green line indicates that the maximum expansion has occurred in the center of the ribbon; water moved in  from both directions to cover that stretch of land! In this case, the maximum pixel intensity is 188 which implies that the extent of the water gain in this AOI is 376m.

![post_gain_1_ddt_gray.png]({{ site.baseurl }}/images/coastal-change/post_gain_1_ddt_gray.png)
![post_gain_1_ddt_rgb3b.png]({{ site.baseurl }}/images/coastal-change/post_gain_1_ddt_rgb3b.png)

*Figure 13. The DDT of the gain map (top) and a visual impression in pseudocolor (bottom).*

And here is the python code for this last chunk of the workflow.

```python
# create DDT tasks and chain to water masks
ddt_gain = gbdx.Task('protogenV2CD_GDDT')
ddt_gain.inputs.raster = water_post.outputs.data.value
ddt_gain.inputs.slave = bcd_gain.outputs.data.value

ddt_loss = gbdx.Task('protogenV2CD_LDDT')
ddt_loss.inputs.raster = water_post.outputs.data.value
ddt_loss.inputs.slave  = bcd_loss.outputs.data.value
```

# Running the workflow

Having created all the task objects and specified the relations between them according to Figure 3, we can run the workflow
and save the output as follows:

```python
import random
import string

# create workflow from constituent tasks
wf = gbdx.Workflow([ipa, water_pre, water_post, exclusion_mask, bcd_tri, bcd_loss, bcd_gain, ddt_gain, ddt_loss])

# set output location to platform-stories/trial-runs/random_str within your bucket/prefix
random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_location = join('platform-stories/trial-runs', random_str)

# set which task outputs are to be stored and where
wf.savedata(water_pre.outputs.data, output_location + '/water_pre')
wf.savedata(water_post.outputs.data, output_location + '/water_post')
wf.savedata(exclusion_mask.outputs.data, output_location + '/exclusion_mask')
wf.savedata(bcd_tri.outputs.data, output_location + '/bcd_tristate')
wf.savedata(bcd_gain.outputs.data, output_location + '/bcd_gain')
wf.savedata(bcd_loss.outputs.data, output_location + '/bcd_loss')
wf.savedata(ddt_gain.outputs.data, output_location + '/ddt_gain')
wf.savedata(ddt_loss.outputs.data, output_location + '/ddt_loss')

# execute workflow
wf.execute()
```

The outputs will be saved under platform-stories/trial-runs/random_str within your bucket/prefix,
where random_str is random string identifier. Here is the [entire workflow](https://github.com/PlatformStories/coastal-change/blob/master/workflow.py).

# Discussion

You can explore the tristate image overlaid on the pre image [here]({{ site.baseurl }}/pages/coastal-change/tristate.html). To create this slippy map, we've ingested the pre AOP-ready image as well as the tristate map into [IDAHO](http://gbdxdocs.digitalglobe.com/docs/idaho-course) format, and used [Leaflet](http://leafletjs.com/) to call the IDAHO TMS service.

We ran the workflow for Cape Cod; you can easily imagine running the exact same workflow for the East Coast of the United States. What you need are the pairs of pre and post imagery; you can use our [catalog](gbdx.geobigdata.io) to find these. You can also experiment with this workflow to detect continental water changes, along rivers and lakes.

Attempts to estimate coastal change have been made by other parties, including [USGS](https://coastalmap.marine.usgs.gov/FlexWeb/national/ShoreLC/). The key advantages of the proposed workflow are precision and scaleability; it runs on high-resolution multi-spectral imagery and it runs on GBDX.

The results of this workflow can be used by interested parties to study the causes of coastal change; tide, sea currents, climate change or any combination thereof could be potential causes. We are currently working on a more thorough quality assessment of our results.

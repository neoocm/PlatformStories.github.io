---
layout: post
title: Detecting population centers in Nigeria
---

There are large regions of the planet which, although inhabited, remain unmapped to this day.
In the past, DigitalGlobe has launched crowdsourcing campaigns to detect remote population centers
in [Ethiopia](http://blog.tomnod.com/mapping-ethiopia), [Sudan](http://blog.tomnod.com/assessing-food-security-in-war-torn-south-sudan)
and [Swaziland](http://blog.tomnod.com/Mapping_Swaziland) in support of NGO vaccination and aid distribution initiatives.
Beyond DigitalGlobe, there are [other](http://www.missingmaps.org/) [initiatives](http://mappingafrica.princeton.edu/) under way to
fill in the gaps in the global map, aiding first responders in their effort to provide relief to vulnerable, yet inaccessible, people.

Crowdsourcing the detection of villages is accurate but slow. Human eyes can easily detect buildings but it takes them a while to cover large
swaths of land. In the past, we have combined crowdsourcing with deep learning on GBDX [to detect and classify objects](http://gbdxstories.digitalglobe.com/swimming-pools/) at scale. This is the approach: collect training samples from the crowd, train a neural network to identify the object of interest, then deploy the trained model on large areas.
The cherry on the cake is to use the crowd to [weed out the errors of the machine](http://blog.tomnod.com/crowd-and-machine-combo), in order to obtain the best of both worlds.

In the context of a recent large-scale population mapping campaign, we were faced with the usual question. Find buildings with the crowd, or train a machine to do it?
This led to another question: can the [convolutional neural network (CNN)](https://github.com/DigitalGlobe/mltools/blob/master/examples/polygon_classify_cnn/README.md) that we trained to find swimming pools in Adelaide be trained to detect buildings in Nigeria?

![villages.png]({{ site.baseurl }}/images/building-detection/villages.png)
*Population centers in Nigeria. Intensity of green corresponds to confidence in the presence of buildings.*

## How

The area of interest consists of 9 WorldView-2 and 2 GeoEye-1 image strips collected between January 2015 and May 2016 over northeastern Nigeria, close to as well as on the border with Niger and Cameroon. We picked 4 WorldView-2 strips, divided them in square chips of side 115m (250 pixels at sensor resolution) and asked our crowd to label them as 'Buildings' or 'No Buildings'. The output of the crowdsourcing campaign is the file train.geojson which contains the labeled chip geometries (a small sample [here](https://github.com/PlatformStories/building-detection/blob/master/train.geojson)).

As shown in the following diagram, train.geojson and the image GeoTiff files are given as input to [train-cnn-classifier](https://github.com/PlatformStories/swimming-pools/blob/master/docs/train-cnn-classifier.md) which produces a trained Keras model. The images are orthorectified, atmospherically compensated and pansharpened using our [image preprocessor](https://gbdxdocs.digitalglobe.com/docs/advanced-image-preprocessor).

![train_wf.png]({{ site.baseurl }}/images/building-detection/train_wf.png)  
*Training on a subset of the strips.*

With a trained model at hand, we can detect buildings in the remaining 7 images. This involves dividing each image
in chips of the same size as those that we trained on to create target.geojson (small sample [here](https://github.com/PlatformStories/building-detection/blob/master/target.geojson))
and passing target.geojson and the image to [deploy-cnn-classifier](https://github.com/PlatformStories/swimming-pools/blob/master/docs/deploy-cnn-classifier.md).

![deploy_wf.png]({{ site.baseurl }}/images/building-detection/deploy_wf.png)  
*Deploying on the remainder of the strips.*

The output of deploy-cnn-classifier is classified.geojson (small sample [here](https://github.com/PlatformStories/building-detection/blob/master/classified.geojson)),
which contains all the chips in target.geojson classified as 'Buildings' or 'No Buildings'
and a confidence score on each classification.

Here is the entire workflow in Python. Note that the deploy tasks are launched in parallel in a simple 'for' loop.

```python
from gbdxtools import Interface
from os.path import join
import uuid

gbdx = Interface()

# specify location of input files
input_location = 's3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform-stories/building-detection'

# train task
train_task = gbdx.Task('train-cnn-classifier')
train_task.inputs.images = join(input_location, 'images')
train_task.inputs.geojson = join(input_location, 'train-geojson')
train_task.inputs.classes = 'No Buildings, Buildings'     # classes in train.geojson

# set hyperparameters
train_task.inputs.nb_epoch = '75'              # validation loss plateaus at around 60 - 70 epochs
train_task.inputs.train_size = '5000'          # enough to get a reliable model, too much higher will make training too slow
train_task.inputs.learning_rate = '0.001'
train_task.inputs.max_side_dim = '245'         # chip side at sensor resolution
train_task.inputs.resize_dim = '(3, 150, 150)' # down sample chips due to memory constraints
train_task.inputs.two_rounds = 'False'         # second round results in low recall
train_task.inputs.test_size = '1500'
train_task.inputs.bit_depth = '8'      
train_task.inputs.batch_size='32'              # low enough to fit into memory

# deploy task
deploy_tasks = {}
deploy_ids = ['103001003D8CC700',       # WV-2
              '1030010041B6F800',       # WV-2   
              '1030010051A75500',       # WV-2  
              '1030010054A8BD00',       # WV-2
              '1030010055AF2D00',       # WV-2
              '10504100120ADF00',       # GE-1
              '1050410012CDC100']       # GE-1

for catid in deploy_ids:
    deploy_task = gbdx.Task('deploy-cnn-classifier')
    deploy_task.inputs.model = train_task.outputs.trained_model.value     # Trained model from train_task
    deploy_task.inputs.images = join(input_location, 'deploy-images', catid)
    deploy_task.inputs.geojson = join(input_location, 'target-geojsons', catid)
    deploy_task.inputs.classes = 'No Buildings, Buildings'
    deploy_task.inputs.bit_depth = '8'
    deploy_task.inputs.min_side_dim = '0'    
    deploy_task.inputs.max_side_dim = '245'
    deploy_tasks[catid] = deploy_task

# define workflow
workflow = gbdx.Workflow([train_task] + deploy_tasks.values())

# set output location to platform-stories/trial-runs/random_str within your bucket/prefix
random_str = str(uuid.uuid4())
output_location = join('platform-stories/trial-runs', random_str)

# save workflow outputs
workflow.savedata(train_task.outputs.trained_model, join(output_location, 'trained_model'))

# save output from each deploy_task
for catid, task in deploy_tasks.iteritems():
    workflow.savedata(task.outputs.classified_geojson, join(output_location, catid, 'classified_geojson'))

# execute workflow
workflow.execute()
```

Once the workflow completes, you can download the outputs as follows.
(Keep in mind that you need to create the local directories before downloading.)

```python
# download model architecture and weights
gbdx.s3.download(join(output_location, 'trained_model', 'model_architecture.json'), 'trained_model')
gbdx.s3.download(join(output_location, 'trained_model', 'model_weights.h5'), 'trained_model')

# download classified chips for one of the strips
catid = '103001003D8CC700'
gbdx.s3.download(join(output_location, catid, 'classified_geojson'), 'classified_geojson')
```

A few words about training the model. We used a training set of 5000 chips, 2500 from each class, which we randomly sampled from the 4 training strips. We down-sampled each chip from 245x245 to 150x150 in order for the network to fit into memory.
After trial and error, we found that the optimum batch size was 32 chips; a smaller size caused validation loss to bounce around during training, while a larger size resulted in memory issues. Finally, we settled on a learning rate of 0.001 because it was the fastest rate that resulted in convergence to the minimum loss.


## Results

We used [gdal_rasterize](http://www.gdal.org/gdal_rasterize.html) in order to create a heat map
of the model confidence in the presence of buildings across each strip that the model was deployed on
(More on how to do this in a future post!) You can examine a subset of the results [on this map]({{ site.baseurl }}/pages/building-detection/diffa.html), where we have overlaid the building heat map on one of the strips near Diffa.

It is quite apparent that the CNN is more confident in the presence of buildings when more of them are
present in the chip! As the building density decreases, the confidence decreases as well. Here are some screenshots
that demonstrate this point.

![result_screenshot.png]({{ site.baseurl }}/images/building-detection/heatmap-samples.png)  
*Sample classifications. CNN confidence increases with building density.*

What is the CNN actually learning? Below are examples of hidden layer outputs produced during classification of a chip that contains buildings. Note that as the chip is processed by successive layers, the locations of buildings become more and more illuminated, leading to a high confidence decision that the chip contains buildings.

![building-layers.png]({{ site.baseurl }}/images/building-detection/building-layers.png)
*Various hidden layers for a chip containing buildings.*

In contrast, a chip which does not contain buildings becomes progressively darker as it travels through the layers. This is because the CNN is not picking up on any of the learned abstract qualities of buildings.

![no-building-layers.png]({{ site.baseurl }}/images/building-detection/no-building-layers.png)
*Various hidden layers for a chip containing no buildings.*


## Discussion

It is fascinating that the same CNN architecture can be used successfully on WorldView-3 imagery to
detect swimming pools in a suburban environment in Australia, and on WorldView-2 and GeoEye-1 imagery to detect buildings in the Nigerian desert.

It takes about 10 hours to train the model for 75 epochs on 4 strips, on a
g2.2xlarge AWS instance. The trained model can classify approximately 200000 chips, i.e., a little over 3000 km2, per hour on the same instance type. For the purpose of this demo, we deployed the trained model on 7 strips. We ran another experiment with 72 strips which is roughly 3 million chips and a total area of around 40000 km2. It took a little less than 2 hours to complete this experiment. Due to the inherent parallelization offered by GBDX, this time is dictated by the size of the largest strip. Going from 7 to 72 strips is simply a matter of adding more catalog ids to the 'for' loop in our Python script.

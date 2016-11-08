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
This led to another question: can the neural network that we trained to [find swimming pools in Adelaide](http://gbdxstories.digitalglobe.com/swimming-pools/) be trained to detect buildings in Nigeria?

![villages.png]({{ site.baseurl }}/images/building-detection/villages.png)
*Population centers in Nigeria.*

## How

The GBDX workflow is shown here. The area of interest consists of N image strips in the region of [Kostas: ...]

![wf.png]({{ site.baseurl }}/images/building-detection/wf.png)  
*GBDX workflow for building detection*

[train.geojson](https://github.com/PlatformStories/building-detection/blob/master/train.geojson) contains a collection of 150x150 m2 chips from N different strips that have been labeled by the crowd as 'Building' and 'No building'. [Kostas: How many of each class? What is N?]. train.geojson and the corresponding imagery are fed to [train-cnn-classifier](https://github.com/PlatformStories/swimming-pools/blob/master/docs/train-cnn-classifier.md) which produces a trained Keras model. This model is then deployed, **in parallel** on a the remainder of the strips by deploy-cnn-classifier. In detail, a strip is divided into a collection of chips which is stored in [target.geojson](https://github.com/PlatformStories/building-detection/blob/master/train.geojson). [deploy-cnn-classifier](https://github.com/PlatformStories/swimming-pools/blob/master/docs/deploy-cnn-classifier.md) accepts target.geojson, the corresponding strip and the trained model as inputs, and produces classified.geojson where each chip in target.geojson is enriched with a label 'Building' or 'No Building', and a confidence score on this classification.

Here is the entire workflow in Python.

```python

from gbdxtools import Interface
from os.path import join
import random
import string

gbdx = Interface()

# specify location of input files
input_location = 's3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform-stories/swimming-pools'

# train task
train_task = gbdx.Task('train-cnn-classifier')
train_task.inputs.images = join(input_location, 'images')
train_task.inputs.geojson = join(input_location, 'train_geojson')
train_task.inputs.classes = 'No swimming pool, Swimming pool'     # Classes exactly as they appear in train.geojson

# set hyperparameters
train_task.inputs.nb_epoch = '30'
train_task.inputs.nb_epoch_2 = '5'
train_task.inputs.train_size = '4500'
train_task.inputs.train_size_2 = '2500'
train_task.inputs.test_size = '1000'
train_task.inputs.bit_depth = '8'         

# deploy task
deploy_task = gbdx.Task('deploy-cnn-classifier')
deploy_task.inputs.model = train_task.outputs.trained_model.value     # Trained model from train_task
deploy_task.inputs.images = join(input_location, 'images')
deploy_task.inputs.geojson = join(input_location, 'target_geojson')
deploy_task.inputs.classes = 'No swimming pool, Swimming pool'
deploy_task.inputs.bit_depth = '8'
deploy_task.inputs.min_side_dim = '10'    

# define workflow
workflow = gbdx.Workflow([train_task, deploy_task])

# set output location to platform-stories/trial-runs/random_str within your bucket/prefix
random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_location = join('platform-stories/trial-runs', random_str)

# save workflow outputs
workflow.savedata(train_task.outputs.trained_model, join(output_location, 'trained_model'))
workflow.savedata(deploy_task.outputs.classified_geojson, join(output_location, 'classified_geojson'))

# execute workflow
workflow.execute()
```

Once the workflow completes, you can download the outputs as follows.
(The exclamation marks allow you to execute bash commands within ipython.)

```python
! mkdir trained_model

# train-cnn-classifier sample output: final model
gbdx.s3.download(join(output_location, 'trained_model/model_architecture.json'), 'trained_model/')
gbdx.s3.download(join(output_location, 'trained_model/model_weights.h5'), 'trained_model/')
gbdx.s3.download(join(output_location, 'trained_model/test_report.txt'), 'trained_model/')

! mkdir classified_geojson

# deploy-cnn-classifier sample output
gbdx.s3.download(join(output_location, 'classified_geojson'), 'classified_geojson')
```

## Results

You can visualize the results here.

![result_screenshot.png]({{ site.baseurl }}/images/swimming-pools/result_screenshot.png)  
*Sample results.*




## Discussion


It takes [Kostas: ?] hours to train the model. To deploy...

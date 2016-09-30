---
layout: post
title: Finding swimming pools in Australia
---

Information about built environments is extremely valuable
to [insurance companies](https://www.allstate.com/tools-and-resources/home-insurance/swimming-pool-insurance.aspx), [tax accessors](http://www.spiegel.de/international/europe/finding-swimming-pools-with-google-earth-greek-government-hauls-in-billions-in-back-taxes-a-709703.html), and public agencies, as it empowers a wide range of decision-making including
urban and regional planning and management, risk estimation, and emergency response.
Extracting this information using human analysts to scour satellite imagery
is prohibitively expensive and time consuming. Feature extraction and machine
learning algorithms are the only viable way to perform this type of attribution
at scale.  This is why the Australian company PSMA teamed up with DG to develop the product
[Geoscape](https://www.psma.com.au/geoscape): a diverse set of building attributes
including height, rooftop material, solar panel installation and presence of a swimming pool
in the property across the **entire Australian continent**.

We'll demonstrate how we used deep learning on GBDX to identify swimming pools in thousands of properties across
[Adelaide](https://en.wikipedia.org/wiki/Adelaide), a major city on the southern coast of Australia with a population of approximately one million. The result: **31071 of the 670784 properties** we classified contain pools; approximately 4.6%. This is more or less consistent with the Australian Bureau of Statistics data for [households with a pool in South Australia](http://www.abs.gov.au/ausstats/abs@.nsf/Products/F4641220D19FD971CA2573A80011AD41?opendocument). Compare this with the corresponding figure for New South Wales which is upwards of 10%. Given the similar arid or semi-arid climate, could we identify other reasons for the discrepancy? A little digging into the regions' economic health might provide some clues:
New South Wales has [a significantly higher average annual income](http://www.abs.gov.au/ausstats/abs@.nsf/mf/5673.0.55.003) compared to South Australia. Given the installation and maintenance costs of a swimming pool, their number could be a potential indicator of economic health!

![properties.png]({{ site.baseurl }}/images/swimming-pools/properties.png)
*A sample of properties. Green/red indicate presence/absence of pool.*

### How

Our GBDX workflow is shown in the following figure.

![workflow_diag.png]({{ site.baseurl }}/images/swimming-pools/workflow_diag.png)  
*Platform workflow for property classification.*

#### Preprocessing

The workflow begins with the file [*properties.geojson*](https://github.com/PlatformStories/swimming-pools/blob/master/cnn_classifier_tasks/train_cnn_classifier/properties.geojson). This file contains a collection of polygons in (longitude, latitude) coordinates, each representing a property. Each polygon has two attributes: an image_id, which determines the DG catalog id of the satellite image corresponding to that polygon, and a feature_id, which is simply a number that uniquely identifies that property. This particular file only contains properties from the image 1040010014800C00, which is a cloudless WV03 image over Adelaide, Australia. You can see this image [here]({{ site.baseurl}}/pages/swimming-pools/my_map.html) by entering your GBDX token when prompted. The token can be found in the *.gbdx-config* file in your root directory. Alternatively, you can open a Python terminal and use gbxtools to get the token as follows:

```python    
from gbdxtools import Interface
gbdx = Interface()
gbdx.gbdx_connection.access_token
```

You can also view the image thumbnail by searching for 1040010014800C00 [here](https://discover.digitalglobe.com/) and navigating to Adelaide using the map.

The main idea is to label a small percentage of the property parcels using [crowdsourcing](http://www.tomnod.com/) in order to create a training set [*train.geojson*](https://github.com/PlatformStories/swimming-pools/blob/master/cnn_classifier_tasks/train_cnn_classifier/train.geojson). We then use *train.geojson* to train a CNN-based classifier to identify the presence of a swimming pool in each of the remaining unlabeled properties of *properties.geojson* (referred to as [*target.geojson*](https://github.com/PlatformStories/swimming-pools/blob/master/cnn_classifier_tasks/deploy_cnn_classifier/target.geojson)). For object classification at a continental or global scale this procedure is a must; it would be virtually impossible to label millions of properties manually in a reasonable amount of time.

Before executing the workflow, the raw image has to be ordered from the factory and processed into a format that is usable by a machine learning algorithm. The last part is trickier than it sounds. We lovingly refer to it as UGHLi: Undifferentiated Geospatial Heavy Lifting. In this example, it involves orthorectification, atmospheric compensation, pansharpening and dynamic range adjustment; all this can be achieved with a single GBDX task [AOP_Strip_Processor](http://gbdxdocs.digitalglobe.com/docs/advanced-ortho-product-aop). For the purpose of this example, the UGHLi'ed image already sits in an S3 bucket.

#### Workflow Inputs

The workflow requires three input directories. These can be found on S3 under the platform_stories/swimming_pools bucket.

<b>train_geojson</b>: The name of the directory where [*train.geojson*](https://github.com/PlatformStories/swimming-pools/blob/master/cnn_classifier_tasks/train_cnn_classifier/train.geojson) is stored. This directory name is required as input by the training task train_cnn_classifier.

![train_geojson.png]({{ site.baseurl }}/images/swimming-pools/train_geojson.png)  

<b>target_geojson</b>: The name of the directory where [*target.geojson*](https://github.com/PlatformStories/swimming-pools/blob/master/cnn_classifier_tasks/deploy_cnn_classifier/target.geojson) is stored. This directory name is required as input by the deploy task deploy_cnn_classifier.

![deploy_data.png]({{ site.baseurl }}/images/swimming-pools/deploy_data.png)  

<b>images</b>: The name of the directory where image files required by train_cnn_classifier and deploy_cnn_classifier are stored. The images should be in [GeoTiff](https://en.wikipedia.org/wiki/GeoTIFF) format. The required format is catalog_id.tif. In this example, we have only one image, *1040010014800C00.tif*.

![strip.png]({{ site.baseurl }}/images/swimming-pools/strip.png)  
*Imagery strip containing pixel data for each polygon in train.geojson and target.geojson.*

Both tasks accept string inputs in addition to the directory inputs listed above. These inputs include class names, training hyper-parameters, and deploy restrictions. See the [docs](https://github.com/PlatformStories/swimming-pools/tree/master/cnn_classifier_tasks/docs) for a full list of inputs.

#### Tasks

The following two GBDX tasks comprise the workflow. They are linked by the trained model, which can be fed directly as input to deploy_cnn_classifier from the output of train_cnn_classifier.

<b>[train_cnn_classifier](https://github.com/PlatformStories/swimming-pools/blob/master/cnn_classifier_tasks/docs/Train_CNN_Classifier.md)</b>: A task to train a CNN classifier on the polygons in *train.geojson*. Required inputs are *train.geojson*, associated image strips, and class names as a string argument. This task returns the architecture and weights of the trained model, which can be saved to an s3 location as well as fed to deploy_cnn_classifier.

![train_cnn_classifier.png]({{ site.baseurl }}/images/swimming-pools/train_cnn_classifier.png)  
 *train_cnn_classifier takes train.geojson and imagery, and produces a trained CNN classifier.*

[deploy_cnn_classifier](https://github.com/PlatformStories/swimming-pools/blob/master/cnn_classifier_tasks/docs/Deploy_CNN_Classifier.md)</b>: A task to deploy a trained CNN model on *target.geojson*. The task requires a trained model, *target.geojson*, and associated image strips, and returns a classified version of *target.geojson*. deploy_cnn_classifier can classify approximately 250,000 polygons per hour.

![deploy_cnn_classifier.png]({{ site.baseurl }}/images/swimming-pools/deploy_cnn_classifier.png)  
*deploy_cnn_classifier takes a model, target.geojson and imagery, and produces classified.geojson.*


#### Workflow Outputs

<b>trained_model</b>: This is the output of the train_cnn_classifier task. It is a directory containing the weights and architecture of the trained model, model weights after each epoch, and a test report. The structure of the output directory is detailed below.

    /trained_model
            ├── model_arch.json
            ├── model_weights.h5
            ├── test_report.txt
            └── model_weights
                ├── round_1
                │   └── weights after each epoch
                └── round_2
                    └── weights after each epoch  

The classifier is a [trained keras model](https://keras.io/models/about-keras-models/). You can find information on the model [here](https://github.com/DigitalGlobe/mltools/tree/master/examples/polygon_classify_cnn)

<b>classified.geojson</b>: This is the output of the deploy_cnn_classifier task. It is simply the results of deploying the model on *target.geojson*. This output directory contains the file *classified.geojson*, which includes all the properties in *target.geojson* classified into 'Swimming pool' and 'No swimming pool'. Neat!

![classified_shapefile.png]({{ site.baseurl }}/images/swimming-pools/classified_shapefile.png)  
*A sample of properties in classified.geojson.*

### Executing the Workflow

We'll now put everything together in gbdxtools.

1. Begin by starting up an iPython terminal, creating a GBDX interface, and getting the input location information:

2. Create a train_task object and set the required inputs:

```python
train_task = gbdx.Task('train_cnn_classifier')
train_task.inputs.images = join(story_prefix, 'images')
train_task.inputs.geojson = join(story_prefix, 'train_geojson')
train_task.inputs.classes = 'No swimming pool, Swimming pool'     # Classes exactly as they appear in train.geojson
```

3. In training our model, we can set optional hyper-parameter. See the [docs](https://github.com/PlatformStories/swimming-pools/blob/master/cnn_classifier_tasks/docs/Train_CNN_Classifier.md) for detailed information. Training should take around 3 hours to complete.

```python
train_task.inputs.nb_epoch = '30'
train_task.inputs.nb_epoch_2 = '5'
train_task.inputs.train_size = '4500'
train_task.inputs.train_size_2 = '2500'
train_task.inputs.test_size = '1000'
train_task.inputs.bit_depth = '8'         # Provided imagery is dra'd
```  

4. Create a deploy_task object with the required inputs, and set the *model* input as the output of train_task. 

```python
deploy_task = gbdx.Task('deploy_cnn_classifier')
deploy_task.inputs.model = train_task.outputs.trained_model.value     # Trained model from train_task
deploy_task.inputs.images = join(story_prefix, 'images')
deploy_task.inputs.geojson = join(story_prefix, 'target_geojson')
```

5. We can also restrict the size of polygons that we deploy on and set the appropriate bit depth for the input imagery:

```python
deploy_task.inputs.bit_depth = '8'
deploy_task.inputs.min_side_dim = '10'    # Minimum acceptable side dimension for a polygon
deploy_task.inputs.classes = 'No swimming pool, Swimming pool'
```

6. String the two tasks together in a workflow and save the output data to S3:

```python
workflow = gbdx.Workflow([train_task, deploy_task])
workflow.savedata(train_task.outputs.trained_model, 'your-bucket-name/train_output')
workflow.savedata(deploy_task.outputs.classified_shapefile, 'your-bucket-name/deploy_output')
```

7. Execute the workflow:

```python
workflow.execute()
```

8. Depending on the hyperparameters set on the model, training sizes, and size of the deploy file, this workflow can take several hours to run. You may check on the status periodically with the following commands:

```python
workflow.status
workflow.events # a more in-depth summary of the workflow status
```

9. You can explore the workflow outputs as follows:

```python
# train_cnn_classifier sample output: final model
mkdir train_output
gbdx.s3.download('platform_stories/swimming_pools/train_output/model_architecture.json', 'train_output/')
gbdx.s3.download('platform_stories/swimming_pools/train_output/model_weights.h5', 'train_output/')
gbdx.s3.download('platform_stories/swimming_pools/train_output/test_report.txt', 'train_output/')

# train_cnn_classifier sample output: weights after each epoch
mkdir train_output/round_1
mkdir train_output/round_2
gbdx.s3.download('platform_stories/swimming_pools/train_output/model_weights/round_1/', 'train_output/round_1/')
gbdx.s3.download('platform_stories/swimming_pools/train_output/model_weights/round_2/', 'train_output/round_2/')

#deploy_cnn_classifier sample output
gbdx.s3.download('platform_stories/swimming_pools/deploy_output/')
```

### Visualizing the Results

You can visualize the classification results [here]({{ site.baseurl }}/pages/swimming-pools/my_map_with_vectors.html) once you have entered your GBDX token in the prompt. Green/red polygons indicate presence/absence of pool. Clicking on each polygon shows the corresponding feature id and assigned classification.  

![vis_example.png]({{ site.baseurl }}/images/swimming-pools/vis_example.png)  
*Sample results of our workflow. Properties containing pools are outlined in green.*

For this visualization, we used the [geojson-vt](https://github.com/mapbox/geojson-vt) library developed by Mapbox to slice the GeoJSON data into vector tiles on the fly. Pretty cool!


### Discussion

This is a small example of what is possible on GBDX. Once a trained model is
obtained, it can be deployed on properties over hundreds of different images
**in parallel**. Continental scale classification becomes a matter of hours.

![scale.png]({{ site.baseurl }}/images/swimming-pools/scale.png)
*GBDX allows large scale parallelization.*  

In order to exploit the power of GBDX, an algorithm must be packaged into a GBDX task. 
The procedure is described [here](https://platformstories.github.io/create-task) in detail.
You can find more information on the algorithm used in this example
[here](https://developer.digitalglobe.com/gbdx-poolnet-identifying-pools-satellite-imagery/) and
[here](https://github.com/DigitalGlobe/mltools/tree/master/examples/polygon_classify_cnn).

---
layout: post
title: Creating a GBDX task
---

Tasks are the bread and butter of the GBDX Platform.
This walkthrough will take you through the steps of creating a task that can
run your Python code on GBDX. We will start with a very simple task and then cover the more advanced
case of creating a machine learning task. We will also demonstrate how you can
setup your machine learning task to run on a GPU.

![header.png]({{ site.baseurl }}/images/create-task/header.png)

## Contents

1. [Background](#background)
2. [Hello GBDX](#hello-gbdx)
3. [Dockerizing](#dockerizing)
    - [About Docker](#about-docker)
    - [Creating a Docker Image](#creating-a-docker-image)
    - [Testing a Docker Image](#testing-a-docker-image)
4. [Registering a Task on GBDX](#registering-a-task-on-gbdx)
5. [Machine Learning on GBDX](#machine-learning-on-gbdx)
    - [Random Forest Classifier](#random-forest-classifier)
    - [Using the GPU](#using-the-gpu)
    - [Convolutional Neural Network](#convolutional-neural-network)


## Background

A GBDX [**task**](http://gbdxdocs.digitalglobe.com/docs/task-and-workflow-course) is a process that performs a specific action on its inputs and generates a set of outputs. In the vast majority of cases, inputs and outputs consist of satellite image files (usually in tif format), vector files (shapefile, geojson), text files and various metadata files (XML, IMD and other).

Tasks can be chained together in a **workflow** where one task's outputs can be the inputs to one or more different tasks. In this manner, more complicated processes can be executed than what is possible within a single task. For example, you can imagine a ship detection workflow consisting of the following tasks: (a) pansharpen the raw satellite image (b) create a sea mask (c) look for boats in the sea.   

When a workflow is executed, tasks are scheduled appropriately by a **scheduler** and the system generates status indicators that are available via the GBDX API. Using the task **definition** on the **task registry**, each task is executed by a **worker node** within a **docker container** that contains the task code and its dependencies in an encapsulated environment. You can find additional information on GBDX in the [GBDX documentation](http://gbdxdocs.digitalglobe.com/).


## Hello GBDX

In this section, we will write a Python script for our Hello GBDX task: 'hello-gbdx'.
The script [*hello_gbdx.py*](https://github.com/kostasthebarbarian/platform_stories/blob/master/create_task/hello-gbdx/code/hello_gbdx.py) (not to be confused with the task name 'hello-gbdx')
does the following: it obtains a list of the task input files and prints this list
in the file *out.txt*, along with a user defined message.
This script is executed by the Python interpreter within the task's Docker container (more on this [a bit later](#about-docker)).

```python
import os
from gbdx_task_interface import GbdxTaskInterface

class HelloGbdxTask(GbdxTaskInterface):

    def invoke(self):

        # Get inputs
        input_dir = self.get_input_data_port('data_in')
        message = self.get_input_string_port('message', default='No message!')

        # Get output
        output_dir = self.get_output_data_port('data_out')
        os.makedirs(output_dir)

        # Write message to file
        with open(os.path.join(output_dir, 'out.txt'), 'w') as f:
            input_contents = ','.join(os.listdir(input_dir))
            f.write(input_contents + '\n')
            f.write(message)


if __name__ == "__main__":
    with HelloGbdxTask() as task:
        task.invoke()
```

What exactly is going on in this script?

The **HelloGbdxTask** class that inherits from the [**GbdxTaskInterface**](https://github.com/TDG-Platform/gbdx-task-interface) class is defined:

```python
import os
from gbdx_task_interface import GbdxTaskInterface

class HelloGbdxTask(GbdxTaskInterface):

    def invoke():
```

You can think of **GbdxTaskInterface** as a GBDX task Python template. It contains prebuilt functions to read the names and values of the task ports and clean-down code to record the result of the task execution ('success' or 'fail'). The ```invoke()``` function implements the task functionality, which is described in the following.

The name of the input directory is obtained using the ```get_input_data_port``` function (inherited from **GbdxTaskInterface**).  

```python
# Get inputs
input_dir = self.get_input_data_port('data_in')
```
data_in is the task **directory input port**. What ```get_input_data_port('data_in')``` does behind the scenes is return the string 'mnt/work/input/data_in'. When the task is executed by the GBDX worker, the contents of the location on S3 specified by data_in are copied onto the Docker container under mnt/work/input/data_in.

The value of the input string port message is obtained using the ```get_input_string_port``` function (also inherited from **GbdxTaskInterface**).  

```python
message = self.get_input_string_port('message', default='No message!')
```

message is one of possibly many task **string input ports**. What ```get_input_string_port('message', default='No message!')``` does behind the scenes is read the value of message from the file *ports.json* which is found under mnt/work/input/. (Keep in mind that you don't have to worry about these inner workings if you don't want to!) If the value is not specified, it returns a default value.

The name of the output directory is obtained using the ```get_output_data_port``` function (inherited from **GbdxTaskInterface**) and the output directory is created.  

```python
# Get output
output_dir = self.get_output_data_port('data_out')
os.makedirs(output_dir)
```

data_out is the task **directory output port**. What ```get_output_data_port('data_out')``` does behind the scenes is return the string 'mnt/output/data_out'. When the task is executed by the GBDX worker, the contents of this directory are copied to the S3 location specified by data_out. Note that this location can be **the same as the location specified by the input port of another task**; this is how tasks as chained together. Moreover, note that **it is the responsibility of the script** to create the output directory.

*out.txt* is created and saved in the output directory:

```python
# Write message to file
with open(os.path.join(output_dir, 'out.txt'), 'w') as f:
    input_contents = ','.join(os.listdir(input_dir))
    f.write(input_contents + '\n')
    f.write(message)
```

Finally, the script executes the ```invoke()``` function when it is run.  

```python
if __name__ == "__main__":
    with HelloGbdxTask() as task:
        task.invoke()
```

In the next section we go through the steps of creating a Docker image for hello-gbdx.

## Dockerizing

Dockerizing is a crucial step to get your code to run as a task on GBDX.
In this section, we will provide a high level overview of [Docker](https://www.docker.com/) and review its terminology. Next, we'll show you how to build your own hello-gbdx Docker image and run a container locally. At the end of this section, you will have an image that can be used to execute hello-gbdx, and have all of the materials necessary to register hello-gbdx on GBDX.

### About Docker

Docker is a software containerization platform that allows developers to package up an application with its dependencies, and deliver it to a user in a single, self-sufficient package (referred to as a container).

![docker_workflow.png]({{ site.baseurl }}/images/create-task/docker_workflow.png)
*Figure 1: Docker allows you to deliver various libraries and scripts in a lightweight package. Docker is required to create a task on GBDX.*

#### Docker and GBDX

When a task is run on GBDX, a Docker container containing the task code, all its required dependencies, and the operating system is run by a worker node. Docker provides an efficient method for delivering the task code and its dependencies to the worker in an encapsulated environment.

#### Docker Lingo

Because Docker can be confusing if you are not used to it, we will define some terms you will encounter in this tutorial. You can consult the [Docker glossary](https://docs.docker.com/engine/reference/glossary/#base-image) for more information.

* <b>Image</b>: Docker images are the basis of containers. An image is a static, immutable file that describes the container environment. Running an image produces a container.

* <b>Container</b>: A container is a runtime instance of a Docker image (similar to an object being an instantiation of a class in Python).

* <b>DockerFile</b>: A Dockerfile is a text document that contains all the commands you would normally execute manually in order to build a Docker image. Docker can build images automatically by reading the instructions from a Dockerfile.
*It is only necessary to create a DockerFile if you elect to create your image from scratch*.

* <b>DockerHub</b>: A repository of images. You can pull and edit these images for your own use, analogous to cloning or forking a GitHub repo.

### Creating a Docker Image

Before creating your image [sign up](https://hub.docker.com/) for a DockerHub account and follow [these](https://docs.docker.com/engine/getstarted/step_one/) instructions to install Docker toolbox on your machine. This will allow you to create, edit, and push your Docker images locally from the command line. In this section, we review two methods for creating a Docker image:

- [Pulling an existing image from DockerHub](#pulling-an-image-from-dockerhub) and [editing it](#adding-your-code-to-the-image) to include additional packages and your code.

- [Building an image from scratch](#building-an-image-with-a-dockerfile) using a DockerFile. If you elect to go this route you can skip the next two sections. (You may wish to read through them anyway as it is useful to know how to edit a Docker image without use of a DockerFile.)

#### Pulling an Image from DockerHub

Before choosing an image to work with, you should choose an operating system and have a list of the libraries your task requires. hello-gbdx requires Python to run, so we will look for a simple base image with Python packages installed. Note that unnecessary libraries and packages will only slow down our container and the task will consequently take longer to run. There are many publicly available images that you can [search](https://docs.docker.com/engine/tutorials/dockerrepos/#searching-for-images) through on DockerHub, so chances are you'll be able to find one that suits your needs. Note that you may install additional dependencies on the image once you've pulled it from DockerHub, so it is not necessary to find one with all of the the required libraries.   

Below is a list of images that may be useful to you:

- **[ubuntu](https://hub.docker.com/r/library/ubuntu/)**: A basic image with an Ubuntu OS and a good starting point for very simple tasks. This can be configured further once you have your own version tagged, which we will review below.

- **[naldeborgh/gdal_base](https://hub.docker.com/r/naldeborgh/gdal_base/)**: An image with Ubuntu 14.04 and GDAL v.1.10.1. (See the repo for a complete list of packages installed.)

- **[naldeborgh/python_vim](https://hub.docker.com/r/naldeborgh/python_vim/)**: An image with Ubuntu 14.04, Python libraries and vim.

Since hello-gbdx only requires Python we will use naldeborgh/python_vim as our base image. In the following steps you will pull this image from its DockerHub repository, tag your own version for editing, and push the new image up to your own DockerHub account:

Login to Docker with your username and password.  

```bash
docker login -u <your_username> -p <your_password>
```

Download the python_vim image to your machine using the following command. Pulling images can take a few minutes depending on their size.

```bash
docker pull naldeborgh/python_vim
```

Once complete, you can confirm that the image has been downloaded by executing the following command:

```bash
docker images
```

You should see something that looks like this:

```bash
REPOSITORY                             TAG                 IMAGE ID            CREATED             SIZE
naldeborgh/python_vim                  latest              ddd4c238e314        About an hour ago   461.1 MB
```

Finally, you can tag the image under your username. This enables you to edit the image and push it to your personal DockerHub repository. Name the image hello-gbdx-docker-image, as we will be moving our hello-gbdx task code into it shortly.

```bash
docker tag naldeborgh/python_vim <your_username>/hello-gbdx-docker-image
docker push <your_username>/hello-gbdx-docker-image
```

This step is optional. We are going to run the image interactively (using the '-it' flag) to produce a container. Recall that a container is a running instance of the image, but the image itself remains static. This means that any changes you make to the container will not affect the image and will be lost once the container is stopped. Familiarize yourself with the structure of the container using ```cd``` and ```ls```:

```bash
# Run the image
docker run -it <your_username>/hello-gbdx-docker-image

# Now we are in the container. Explore some as you would in the command line
root@ff567ca72fa0:/ ls
>>> boot  etc   lib    media  opt   root  sbin  sys  usr  bin  dev   home  lib64  mnt    proc  run   srv   tmp  var
```

Congratulations! You now have your own docker image to which you can add your code.  

#### Adding Your Code to the Image

![docker_commit_wf.png]({{ site.baseurl }}/images/create-task/docker_commit_wf.png)
*Figure 2: Adding code to a docker image.*

The following steps (shown in Figure 2) walk you through adding [*hello_gbdx.py*](https://github.com/kostasthebarbarian/platform_stories/blob/master/create_task/hello-gbdx/code/hello_gbdx.py) and [*gbdx_task_interface.py*](https://github.com/kostasthebarbarian/platform_stories/blob/master/create_task/hello-gbdx/code/gbdx_task_interface.py) to hello-gbdx-docker-image. Before getting started ensure that both scripts are saved to your current working directory.

First we run the image using the ```docker run``` command in detached mode (using the -d flag). This tells the container to run in the background so we can access the files on our local machine.

```bash
docker run -itd <your_username>/hello-gbdx-docker-image
>>> ff567ca72fa0ed6cdfbe0a5c02ea3e04f88ec49239344f217ce1049651d01344
```

The value returned is the *container id*. Make note of this because we will need it.

We now use the ```docker cp``` command to copy our scripts into the container. The format of this command is as follows: ```docker cp <filename> <container_id>:<container_dest_path>```.

```bash
# Copy hello_gbdx.py to the root directory of the container
docker cp hello_gbdx.py <container_id>:/
docker cp gbdx_task_interface.py <container_id>:/
```

Our scripts are now in the container. You may confirm this by attaching to the container (bringing it back to the foreground) as follows:

```bash
docker attach <container_id>

# Notice that the scripts now live in the root directory of the container
root@ff567ca72fa0:/ ls
gbdx_task_interface.py   hello_gbdx.py   boot  etc   lib    media  opt   root  sbin  sys  usr  bin  dev   home  lib64  mnt    proc  run   srv   tmp  var
```

You may detach from the container (sending it back to background) without stopping it using the following escape sequence: <kbd>Ctrl</kbd>-<kbd>p</kbd> + <kbd>Ctrl</kbd>-<kbd>q</kbd>  

Up until now we have been working in a container and not the image. If we were to stop the container now, all of our changes would be lost and hello-gbdx-docker-image image would remain unchanged. To permanently update the image we must commit our changes to it. This command will save the changes you made to your container to hello-gbdx-docker-image.

```bash
# Commit the changes from the container to the image
docker commit -m 'add scripts to root' <container_id> <your_username>/hello-gbdx-docker-image

# Push the changes up to DockerHub
docker push <your_username>/hello-gbdx-docker-image
```

Now when you run hello-gbdx-docker-image, *hello_gbdx.py* and *gbdx_task_interface.py* will be in the root directory.

**Extra credit**: Although hello-gbdx does not require any additional libraries to run, often times you will need to install a package that is not provided in the image that you pulled. Let's say our task requires numpy to run; the process of adding it to the image is similar:

```bash
# Start up a container in attached mode
docker run -it <your_username>/hello-gbdx-docker-image

# Import numpy (can take several minutes)
root@ff567ca72fa0:/ pip install numpy

# Exit the container
root@ff567ca72fa0:/ exit

# Commit your changes to the image, push to DockerHub
docker commit -m 'Add numpy' <container_id> <your_username>/hello-gbdx-docker-image
docker push <your_username>/hello-gbdx-docker-image
```

Our image is ready! If you would like to learn how to build an image from scratch continue on the [next section](#building-an-image-with-a-dockerfile). Otherwise, move on to [testing the image locally](#testing-a-docker-image).

#### Building an Image with a DockerFile

Feeling a little ambitious? You can build your own image from scratch with a DockerFile.
For more information on DockerFiles see [here](https://docs.docker.com/engine/reference/builder/) and [here](https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/).

![dockerfile.png]({{ site.baseurl }}/images/create-task/dockerfile.png)
*Figure 3: Building a docker image with a DockerFile.*

In this example, we will build hello-gbdx-docker-image.

We begin by making the directory [hello-gbdx-build](https://github.com/kostasthebarbarian/platform_stories/tree/master/create_task/hello-gbdx/hello-gbdx-build), which will contain our DockerFile, and the subdirectory bin in which we copy *hello_gbdx.py* and *gbdx_task_interface.py*.

```bash
# Make build an bin directories
mkdir hello-gbdx-build/
cd hello-gbdx-build/
mkdir bin/

# Copy both scripts for hello-gbdx into the bin/ directory
cp path/to/hello_gbdx.py bin/
cp path/to/gbdx_task_interface.py bin/
```

Now we can create our DockerFile. From within the hello-gbdx-build directory, type ```vim Dockerfile``` (or whatever code editor you prefer) to create a blank document. The first line of *DockerFile* is the OS that your image uses. We will be working with Ubuntu. Type the following in the first line of the file:

```bash
FROM ubuntu:14.04
```

Next we add the commands that install the required dependencies preceded by the keyword RUN.

```bash
# install Python packages
RUN apt-get update && apt-get -y install\
    python \
    vim\
    build-essential\
    python-software-properties\
    software-properties-common\
    python-pip\
    python-dev
```

We instruct Docker to place the contents of [bin](https://github.com/kostasthebarbarian/platform_stories/tree/master/create_task/hello-gbdx/hello-gbdx-build/bin) into the image root directory. Add the following line to the end of *DockerFile*:

```bash
# Add all scripts in bin to the image root directory
ADD ./bin /
```

Our DockerFile is now [complete]((https://github.com/kostasthebarbarian/platform_stories/blob/master/create_task/hello-gbdx/hello-gbdx-build/Dockerfile)). Exit vim with the ```:wq``` command.

We are now ready to build the image. Make sure that you are still in the hello-gbdx-build directory and execute the following command:

```bash
docker build -t <your_username>/hello-gbdx-docker-image .
```

The build will take several minutes the first time through. If you change your script or the DockerFile you will need to rebuild the image. Subsequent builds will be faster since Docker builds images in layers.

Finally, push the image to DockerHub.

```bash
docker push <your username>/hello-gbdx-docker-image
```

Our Docker image is now ready to be tested with sample inputs.

### Testing a Docker Image

At this point you should have hello-gbdx-docker-image which includes *hello_gbdx.py*.
In this section, we will run this image with actual input data. Successfully doing this locally ensures that hello-gbdx will run on GBDX. [hello-gbdx/test_inputs](https://github.com/kostasthebarbarian/platform_stories/tree/master/create_task/hello-gbdx/test_inputs) in this repo contains the two inputs required by hello-gbdx: (a) the directory [data_in](https://github.com/kostasthebarbarian/platform_stories/tree/master/create_task/hello-gbdx/test_inputs/data_in), the contents of which will be written to *out.txt* (in this example, this is simply the file *data_file.txt*) (b) the file [*ports.json*](https://github.com/kostasthebarbarian/platform_stories/tree/master/create_task/hello-gbdx/test_inputs/ports.json) which
contains the message to be written to *out.txt*. Keep in mind that *ports.json* is automatically created by GBDX based on the task definition and the values of the string input ports provided by the user when the task is executed.

Run hello-gbdx-docker-image and mount inputs to the container.

```bash
docker run -v ~/path/to/hello-gbdx/test_inputs:/mnt/work/input -it <your_username>/hello-gbdx-docker-image
```

The inputs are mounted onto the container under the directory mnt/work/input. This is where GBDX actually places the inputs specified by the input ports when the task is executed. Note the important distinction between mounting data to the container and adding data to the image using the ADD command in the Dockerfile: when you exit the container, this data 'disappears' (i.e., it is not saved onto the image).

Confirm that the inputs are mounted by exploring the container.

```bash
# Look at the contents of the input directory. data_in should be mounted.
root@3ad24b35e32e:/ ls /mnt/work/input/
>>> data_in  ports.json
```

To test hello-gbdx, simply run the *hello_gbdx.py* script.

```bash
root@3ad24b35e32e:/ python hello_gbdx.py
```

If the script completes successfully you shouldn't see anything written to STDOUT and the file *out.txt* should be found under mnt/work/output/. Here is how you can confirm this:

```bash
# Navigate to the output directory, ensure that 'data_out' lives there
root@3ad24b35e32e:/ cd mnt/work/output
root@3ad24b35e32e:/ ls
>>> data_out/

# Make sure that data_out contains the expected output file
root@3ad24b35e32e:/ cd data_out
root@3ad24b35e32e:/ ls
>>> out.txt
```

You can also make sure *out.txt* contains the expected content by typing ```vim out.txt```. The file should look like this:

```bash
data_file.txt
This is my message!
```

Congratulations, your task is working as expected! The next step is to [create a task definition](#defining-the-task), which will be used to [register](#registering-a-task-on-gbdx) hello-gbdx on the platform.


## Registering a Task on GBDX

Now that we have hello-gbdx-docker-image working locally, we can finally define hello-gbdx and then register it to the GBDX task registry.

### Defining the Task

The task definition is a [json file](https://github.com/kostasthebarbarian/platform_stories/blob/master/create_task/hello-gbdx/hello-gbdx_definition.json) that contains a description of the task functionality, a list of its inputs and outputs, and the Docker image that needs to be run when the task is executed.

```json
{
    "name": "hello-gbdx",
    "description": "Writes list of the  input file names and a user defined message to output file out.txt.",
    "properties": {
        "isPublic": true,
        "timeout": 7200
    },
    "inputPortDescriptors": [
        {
            "name": "message",
            "type": "string",
            "description": "User defined message.",
            "required": true
        },
        {
            "name": "data_in",
            "type": "directory",
            "description": "S3 location of input files.",
            "required": true
        }
    ],
    "outputPortDescriptors": [
        {
            "name": "data_out",
            "type": "directory",
            "description": "S3 location of the output file out.txt."
        }
    ],
    "containerDescriptors": [
        {
            "type": "DOCKER",
            "properties": {
                "image": "naldeborgh/hello-gbdx-docker-image"
            },
            "command": "python /hello_gbdx.py",
            "isPublic": true
        }
    ]
}

```

We review the four parts of this definition below.

<b>Task properties</b>:    

```
{
    "name": "hello-gbdx",
    "description": "Writes list of the  input file names and a user defined message to output file out.txt.",
    "properties": {
        "isPublic": true,
        "timeout": 7200
```

- <b>name</b>: The task name.
- <b>description</b>: A brief, high-level description of the task.
- <b>isPublic</b>: A boolean. If true, the task is publicly available.
- <b>timeout</b>: Amount of time (in seconds) for the task to run before it is terminated by the platform. The max value is 36000 (i.e., 10 hours).  

<b> Input Port Descriptors</b>: This is where the task input ports are defined.  

```
"inputPortDescriptors": [
    {
        "name": "message",
        "type": "string",
        "description": "User defined message.",
        "required": true
    },
    {
        "name": "data_in",
        "type": "directory",
        "description": "S3 location of input files.",
        "required": true
    }
```

- <b>name</b>: The input port name.
- <b>type</b>: The input port type. Currently the only options are 'directory' and 'string'. A directory input port is used to point to an S3 where input files are stored. A string input port is used to port is used to pass a string parameter to the task. Note that integers, floats and booleans must all be provided to a task in string format!
- <b>description</b>: Description of the input port.
- <b>required</b>: A boolean. 'true'/'false' indicate required/optional input, respectively.

<b> Output Port Descriptors</b>: This is where the task output ports are defined.

```
"outputPortDescriptors": [
    {
        "name": "data_out",
        "type": "directory",
        "description": "S3 location of the output file out.txt."
    }
```

- <b>name</b>: The output port name.
- <b>type</b>: The output port type. Currently, the only options are 'directory' and 'string'.
- <b>description</b>: Description of the output port.  

<b>Container Descriptors</b>:  

```
"containerDescriptors": [
    {
        "type": "DOCKER",
        "properties": {
            "image": "naldeborgh/hello-gbdx-docker-image"
        },
        "command": "python /hello_gbdx.py",
        "isPublic": true
```

- <b>type</b>: The domain on which the task is run. Typical tasks are run on the 'DOCKER' domain. Change this to 'GPUDOCKER' for tasks that require a GPU to run (more on this [later](#the-gpu-definition)).
- <b>image</b>: The name of the Docker image that is pulled from DockerHub.
- <b>command</b>: The command to run within the container.


### GBDX Task Registry

We now have all the required material to register hello-gbdx: [a Docker image on DockerHub](#creating-a-docker-image) and the [task definition](https://github.com/kostasthebarbarian/platform_stories/blob/master/create_task/hello-gbdx/hello-gbdx_definition.json).

Open an iPython terminal, import gbdxtools and start up a GBDX Interface.  

```python
from gbdxtools import Interface

gbdx = Interface()
```

Call the ```register()``` method of the **TaskRegistry** class with the name of the definition JSON. (Make sure *hello-gbdx_definition.json* is in your working directory).

```python
gbdx.task_registry.register(json_filename = 'hello-gbdx_definition.json')
>>> u'hello-gbdx successfully registered.'
```

There's a good chance that hello-gbdx already exists in the registry. You can try using a different name after appropriaterly modifying the definition.

Congratulations, you have just registered hello-gbdx! You can run it with sample data we have provided on S3 as follows:

```python
from gbdxtools import Interface
from os.path import join
import string, random
gbdx = Interface()

# get input location
bucket = gbdx.s3.info['bucket']
prefix = gbdx.s3.info['prefix']
story_prefix = 's3://' + join(bucket, prefix, 'platform_stories', 'create_task', 'hello_gbdx')

# create the task and set inputs
hello_task = gbdx.Task('hello-gbdx')
hello_task.inputs.data_in = join(story_prefix, 'data_in')
hello_task.inputs.message = 'This is my message!'

# create unique location name to save output
output_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_loc = join('platform_stories/create_task/hello_gbdx/user_outputs', output_str)

# define a single-task workflow
workflow = gbdx.Workflow([hello_task])

# specify location to save the output data, execute the workflow
workflow.savedata(hello_task.outputs.data_out, output_loc)
workflow.execute()

# Once the workflow is complete download the output as follows:
gbdx.s3.download(join('platform_stories/create_task/hello_gbdx/user_outputs', output_str))
```

What ```workflow.savedata``` does behind the scenes is set the value of the data_out port to output_loc.
When the workflow completes, the contents of data_out are moved to output_loc.

To delete hello-gbdx from the registry:

```python
gbdx.task_registry.delete('hello-gbdx')
>>> u'hello-gbdx successfully deleted.'
```

You have created a basic yet fully functional GBDX task using Docker and gbdxtools. The next section covers the process of creating more complicated tasks that can run machine learning algorithms.


## Machine Learning on GBDX

In the last section, we created a simple task that generates a text file with a list of the contents of the input directory and a user defined message. Chances are you're looking to do a bit more with your task. In this section, you will learn how to create a task that runs a standard machine learning (ML) algorithm such as a random forest classifier, and how to setup a ML task that utilizes a Convolutional Neural Network (CNN) so that it can be executed on a worker with a GPU.

### Random Forest Classifier

In this example, we will create the task 'rf-pool-classifier' that trains a [random forest classifier](https://en.wikipedia.org/wiki/Random_forest) to classify polygons of arbitrary geometry into those that contain swimming pools and those that don't. For more information on this algorithm see [here](https://github.com/DigitalGlobe/mltools/tree/master/examples/polygon_classify_random_forest) and [here](http://blog.tomnod.com/crowd-and-machine-combo).

![rf_img.png]({{ site.baseurl }}/images/create-task/rf_img.png)
*Figure 4: Inputs and output of rf-pool-classifier.*

rf-pool-classifier has two directory input ports: geojson and image. Within the S3 locations specified by geojson and image the task expects to find a file *train.geojson*, which contains labeled polygons from both classes, and a tif image file from which the task will extract the pixels corresponding to each polygon, respectively (Figure 4). The task also has the input string port n_estimators that determines the number of trees in the random forest; specifying a value is optional and the default is '100'. The task produces a trained model in [pickle](https://docs.python.org/2/library/pickle.html) format, which is saved in the S3 location specified by the output port trained_classifier.

#### The Code

The code of *rf_pool_classifier.py* is shown below; the structure is the same as *hello_gbdx.py*.

```python
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')   # suppress annoying warnings

from shutil import move
from mltools import features
from mltools import geojson_tools as gt
from mltools import data_extractors as de
from gbdx_task_interface import GbdxTaskInterface
from sklearn.ensemble import RandomForestClassifier

class RfPoolClassifier(GbdxTaskInterface):

    def invoke(self):

        # Get inputs
        n_estimators = int(self.get_input_string_port('n_estimators', default = '100'))
        img_dir = self.get_input_data_port('image')
        img = os.path.join(img_dir, os.listdir(img_dir)[0])

        geojson_dir = self.get_input_data_port('geojson')
        geojson = os.path.join(geojson_dir, os.listdir(geojson_dir)[0])


        # Move geojson to same dir as img
        move(geojson, img_dir)

        # Navigate to directory with input data
        os.chdir(img_dir)

        # Create output directory
        output_dir = self.get_output_data_port('trained_classifier')
        os.makedirs(output_dir)

        # Get training data from the geojson input
        train_rasters, _, train_labels = de.get_data('train.geojson', return_labels=True, mask=True)

        # Compute features from each training polygon
        compute_features = features.pool_basic
        X = []
        for raster in train_rasters:
            X.append(compute_features(raster))

        # Create classifier object.
        c = RandomForestClassifier(n_estimators = n_estimators)

        # Train the classifier
        X, train_labels = np.nan_to_num(np.array(X)), np.array(train_labels)
        c.fit(X, train_labels)

        # Pickle classifier and save to output dir
        with open(os.path.join(output_dir, 'classifier.pkl'), 'wb') as f:
            pickle.dump(c, f)


if __name__ == "__main__":
    with RfPoolClassifier() as task:
        task.invoke()

```

Here is what's going on in the script:

We define the **RfPoolClassifier** class that inherits from **GbdxTaskInterface**, and read the input ports.  

```python
class RfPoolClassifier(GbdxTaskInterface):

    def invoke(self):

        # Get inputs
        n_estimators = int(self.get_input_string_port('n_estimators', default = '100'))
        img_dir = self.get_input_data_port('image')
        img = os.path.join(img_dir, os.listdir(img_dir)[0])

        geojson_dir = self.get_input_data_port('geojson')
        geojson = os.path.join(geojson_dir, os.listdir(geojson_dir)[0])
```

We move all the input files in the same directory (this particular implementation wants them in one place), and create the output directory.  

```python
# Move geojson to same dir as img
move(geojson, img_dir)

# Navigate to directory with input data
os.chdir(img_dir)

# Create output directory
output_dir = self.get_output_data_port('trained_classifier')
os.makedirs(output_dir)
```

Using the ```mltools.data_extractors``` modules, the pixels corresponding to each polygon in *train.geojson* are extracted and stored in a masked numpy array. For each array, a 4-dim feature vector is computed by the function ```features.pool_basic``` and stored in the list X.

```python
# Get training data from the geojson input
train_rasters, _, train_labels = de.get_data('train.geojson', return_labels=True, mask=True)

# Compute features from each training polygon
compute_features = features.pool_basic
X = []
for raster in train_rasters:
    X.append(compute_features(raster))
```

We create an instance of the [sklearn Random Forest Classifier class](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and train it using X and corresponding labels.  

```python
# Create classifier object.
c = RandomForestClassifier(n_estimators = n_estimators)

# Train the classifier
X, train_labels = np.nan_to_num(np.array(X)), np.array(train_labels)
c.fit(X, train_labels)
```

We save the trained classifier to the output directory port.

```python
# Pickle classifier and save to output dir
with open(os.path.join(output_dir, 'classifier.pkl'), 'wb') as f:
    pickle.dump(c, f)
```

Finally, we call the ```invoke()``` function when the script is run.

```python
if __name__ == "__main__":
    with RfPoolClassifier() as task:
        task.invoke()
```

#### The Docker Image

rf-pool-classifier requires more libraries than hello-gbdx, such as [numpy](http://www.numpy.org/) and [mltools](https://github.com/DigitalGlobe/mltools). We build the Docker image rf-pool-classifier-docker-image by pulling [naldeborgh/gdal_base](https://hub.docker.com/r/naldeborgh/gdal_base/) and installing the required libraries.

```bash
# Pull and tag the Docker image
docker pull naldeborgh/gdal_base
docker tag naldeborgh/gdal_base <your_username>/gdal_base

# Run the container
docker run -it <your_username>/gdal_base

# Install packages
root@5d4ae93d26dd:/ apt-get update && apt-get install -y git
root@5d4ae93d26dd:/ pip install gdal numpy ephem psycopg2
root@5d4ae93d26dd:/ pip install git+https://github.com/DigitalGlobe/mltools

# Exit the container and commit changes to a new image name
root@5d4ae93d26dd:/ exit
docker commit -m 'install rf classifier packages' <container_id> <your_username>/rf-pool-classifier-docker-image
```

We are now ready to copy *rf_pool_classifier.py* and *gbdx_task_interface.py* to rf-pool-classifier-docker-image. Make sure to have the script saved to your working directory and execute the following:

```bash
# Run Docker in detached mode
docker run -itd <your_username>/rf-pool-classifier-docker-image

# Copy the script to the container and commit the changes
docker cp rf_pool_classifier.py <container_id>:/
docker cp gbdx_task_interface.py <container_id>:/
docker commit -m 'copy rf_pool_classifier script' <container_id> <your_username>/rf-pool-classifier-docker-image
```

You can also build rf-pool-classifier-docker-image from scratch using the DockerFile below. The DockerFile should be in your working directory and your scripts should reside in ./bin for this to work.

```bash
FROM ubuntu:14.04

# install python and gdal packages
RUN apt-get update && apt-get install -y\
   git\
   vim\
   python \
   ipython\
   build-essential\
   python-software-properties\
   software-properties-common\
   python-pip\
   python-scipy\
   python-dev\
   gdal-bin\
   python-gdal\
   libgdal-dev

# install ml dependencies
RUN pip install gdal numpy ephem psycopg2
RUN pip install git+https://github.com/DigitalGlobe/mltools

# put code into image
ADD ./bin /
```

To build the image:

```bash
docker build -t <your_username>/rf-pool-classifier-docker-image .
```

The DockerFile and scripts can be found [here](https://github.com/kostasthebarbarian/platform_stories/tree/master/create_task/rf-pool-classifier/rf-pool-classifier-build).

#### Testing the Docker Image

We can now test rf-pool-classifier-docker-image on our local machine before defining rf-pool-classifier and registering it on the platform. Just as in the case of [hello-gbdx](#testing-a-docker-image), we will mimic the platform by mounting sample input to a container and then executing *rf_pool_classifier.py*.

Create the directory rf_pool_classifier_test in which to download the task inputs from S3. You will need subdirectories for geojson and image.

```bash
# create input port directories
mkdir rf_pool_classifier_test
cd rf_pool_classifier_test
mkdir geojson
mkdir image
```

From an iPython terminal, download *1040010014800C00.tif* and *train.geojson* from their S3 locations as follows.
Note that the tif is large (~15 GB) so you will need adequate disk space.

```python
from gbdxtools import Interface
gbdx = Interface()

# download the image strip (will take a couple minutes)
gbdx.s3.download('platform_stories/create_task/rf_pool_classifier/image/1040010014800C00.tif', './image/')

# download the geojson with training data
gbdx.s3.download('platform_stories/create_task/rf_pool_classifier/geojson/train.geojson', './geojson/')

# exit iPython
exit
```

Run rf-pool-classifier-docker-image with rf_pool_classifier_test mounted to the input port.

```bash
docker run -v ~/<full/path/to/rf_pool_classifier_test>:/mnt/work/input -it <your_username>/rf-pool-classifier-docker-image
```

Within the container run *rf_pool_classifier.py*.

```bash
python /rf_pool_classifier.py
```

The script should run without errors. To confirm this, check the output port directory for *classifier.pkl*.

```bash
root@91d9d5cd9570:/ ls mnt/work/output/trained_classifier
>>> classifier.pkl
```

You can now define and register rf-pool-classifier!

#### Task Definition

The definition for rf-pool-classifier is provided below:

```json
{
    "name": "rf-pool-classifier",
    "description": "Train a Random Forest Classifier to identify polygons containing pools.",
    "properties": {
        "isPublic": true,
        "timeout": 7200
    },
    "inputPortDescriptors": [
        {
            "name": "image",
            "type": "directory",
            "description": "S3 location of image strip.",
            "required": true
        },
        {
            "name": "geojson",
            "type": "directory",
            "description": "S3 location of train.geojson. Each feature in train.geojson should contain a feature_id, image_id, and class name (either 'No swimming pool' or 'Swimming pool')",
            "required": true
        },
        {
            "name": "n_estimators",
            "type": "string",
            "description": "Number of trees in the random forest classifier. Defaults to 100.",
            "required": false
        }
    ],
    "outputPortDescriptors": [
        {
            "name": "trained_classifier",
            "type": "directory",
            "description": "S3 location of 'classifier.pkl' which contains the trained model."
        }
    ],
    "containerDescriptors": [
        {
            "type": "DOCKER",
            "properties": {
                "image": "naldeborgh/rf-pool-classifier-docker-image"
            },
            "command": "python /rf_pool_classifier.py",
            "isPublic": true
        }
    ]
}
```

Put *rf-pool-classifier_definition.json* in your working directory and register rf-pool-classifier as follows:

```python
from gbdxtools import Interface
gbdx = Interface()

# register the task using rf-pool-classifier_definition.json
gbdx.task_registry.register(json_filename = 'rf-pool-classifier_definition.json')
```

#### Executing the Task

We will now run through a sample execution of rf-pool-classifier using gbdxtools.
We have provided sample input data in platform_stories/create_task/rf_pool_classifier.

Open an iPython terminal, create a GBDX interface and get the task input location.

```python
from gbdxtools import Interface
from os.path import join
import random, string

gbdx = Interface()
bucket = gbdx.s3.info['bucket']
prefix = gbdx.s3.info['prefix']

# specify location
story_prefix = 's3://' + join(bucket, prefix, 'platform_stories', 'create_task', 'rf_pool_classifier')
```

Create an rf_task object and specify the inputs.

```python
rf_task = gbdx.Task('rf-pool-classifier')
rf_task.inputs.image = join(story_prefix, 'image')
rf_task.inputs.geojson = join(story_prefix, 'geojson')
rf_task.inputs.n_estimators = "1000"
```

Create a single-task workflow object and define where the output data should be saved.

```python
# create unique location name to save output
output_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_loc = join('platform_stories/create_task/rf_pool_classifier/user_outputs', output_str)

workflow = gbdx.Workflow([rf_task])
workflow.savedata(rf_task.outputs.trained_classifier, output_loc)
```

Execute the workflow and monitor its status as follows:

```python
workflow.execute()

# Monitor workflow status
workflow.status
```

Once the workflow has completed you may download the task output:

```python
gbdx.s3.download(join('platform_stories/create_task/rf_pool_classifier/user_outputs/', output_str))
```

Done! At this point we have created and executed a simple ML task on the platform.
In the next section, we will cover how to make use of the GPU for compute-intensive algorithms
that rely on it.


### Using the GPU

Until now, we have been running our tasks on a CPU device. For certain ML applications such as deep learning that are very compute-intensive, the GPU offers order-of-magnitude performance improvement compared to the CPU. GBDX provides the capability of running a task on a GPU worker. This requires configuring the Docker image and defining the task appropriately.

This section will walk you through setting up a local GPU instance for building and testing your Docker image, and then building a GPU-compatible Docker image, i.e., a Docker image that can access the GPU on the node on which it is run.

#### Setting up a GPU instance

Currently, all GPU devices on GBDX use NVIDIA driver version 346.46. Here are the steps for setting up an AWS instance with this driver, which you will need to test your GPU-compatible Docker image.

Launch an [EC2 g2.2xlarge](https://aws.amazon.com/ec2/instance-types/#g2) ubuntu instance on AWS. At least 20GB of storage is recommended.  

Now ssh into your instance and install build-essential.

```bash
ssh -i <path/to/key_pair> ubuntu@<instance_id>

sudo -s
apt-get update && apt-get install build-essential
```

Get CUDA installer.

```bash
wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
```

Extract CUDA installer.  

```bash
>> chmod +x cuda_6.5.14_linux_64.run
>> mkdir nvidia_installers
>> ./cuda_6.5.14_linux_64.run -extract=`pwd`/nvidia_installers
```

Download the *NVIDIA-Linux-x86_64-346.46.run* file provided in this repo and copy it to your AWS instance.

```bash
# copy the NVIDIA driver from your local machine to the AWS instance
scp -i <path/to/key_pair> <path/to/NVIDIA-Linux-x86_64-346.46.run> ubuntu@<instance_id>:/

# ssh back into the AWS instance, and move the driver file to the nvidia_installers directory
mv /path/to/NVIDIA-Linux-x86_64-346.46.run nvidia_installers/
cd nvidia_installers
```

Run NVIDIA driver installer. An 8-bit UI will ask you a number of questions. Agree to everything.  

```bash
./NVIDIA-Linux-x86_64-346.46.run
```

If you get a kernel error while trying to install the NVIDIA driver,
troubleshoot according to the black bullets.
Otherwise, skip these completely.

- Install linux image extra virtual and reboot.

```bash
sudo apt-get install linux-image-extra-virtual
reboot
```

- Once the instance has rebooted, ssh back into it and create a file to blacklist nouveau as follows.

```bash
sudo -s
vim /etc/modprobe.d/blacklist-nouveau.conf

# Add these lines to file and save
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off

# Disable the Kernel Nouveau and reboot
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
update-initramfs -u
reboot
```

- Once again ssh into the rebooted instance and install linux source and headers.

```bash
# ssh into the instance, get kernel source
sudo -s
apt-get install linux-source
apt-get install linux-headers-3.13.0-37-generic
```

- Finally, update grub and rerun the driver installation.

```bash
update-grub
apt-get install linux-headers-`uname -r`

# Rerun the NVIDIA installer
cd nvidia_installers
./NVIDIA-Linux-x86_64-346.46.run
```

Once the driver is successfully installed, reboot the machine.

```bash
sudo reboot
```

Load NVIDIA kernel module.  

```bash
modprobe nvidia
```

Run CUDA and samples installer.  

```bash
./cuda-linux64-rel-6.5.14-18749181.run
./cuda-samples-linux-6.5.14-18745345.run
```

Verify CUDA installation.  

```bash
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

If CUDA has been installed successfully you will see the following output:

```bash
deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 7.0, CUDA Runtime Version = 6.5, NumDevs = 1, Device0 = GRID K520
Result = PASS
```

#### Building a GPU-Compatible Image

In this section, we will build the [gbdx-caffe-ubuntu](https://github.com/ctuskdg/gbdx-gpu-docker/tree/master/ubuntu) Docker image created by Carsten Tusk, and then add the [Theano](http://deeplearning.net/software/theano/) and [Keras](https://keras.io/) Python libraries, as these are required by the example task presented in the next [section](#convolutional-neural-network). While still logged into the AWS instance, execute the following steps:

Clone [this repo](https://github.com/ctuskdg/gbdx-gpu-docker) and navigate to the ubuntu folder.  

```bash
git clone https://github.com/ctuskdg/gbdx-gpu-docker
cd gbdx-gpu-docker/ubuntu/
```

Build the image as follows. This can take a long time (~ 30 mins).

```bash
sudo bash build
```

If you are going to be running Caffe instead of Theano or Keras on your image you will be able to use this image as is. Simply tag the image as follows and skip the remaining steps in this section.

```bash
# FOR CAFFE USERS ONLY
docker tag gbdx-caffe-ubuntu <your_username>/gbdx-gpu

# To run the image with GPU devices mounted:
docker run --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 -it <your_username>/gbdx-gpu /bin/bash
```

The build will result in three new Docker images on your device. We will be working with the one entitled 'gbdx-caffe-ubuntu'. Use the following command to run it with the GPU devices mounted, and navigate to the root directory:  

```bash
# Run the image with the GPU devices mounted
docker run --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 -it gbdx-caffe-ubuntu /bin/bash
root@a28d273819ea:/build# cd /
root@a28d273819ea:/
```

From inside the Docker container we will now install CUDA, which will prevent errors when running Theano. Complete the following commands:

```bash
# Install build-essential
root@a28d273819ea:/ sudo -s
root@a28d273819ea:/ apt-get update && apt-get install build-essential

# Get and dpkg CUDA installer
root@a28d273819ea:/ wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
root@a28d273819ea:/ chmod +x cuda_6.5.14_linux_64.run
root@a28d273819ea:/ mkdir nvidia_installers
root@a28d273819ea:/ ./cuda_6.5.14_linux_64.run -extract=`pwd`/nvidia_installers

# Run CUDA installers
root@a28d273819ea:/ cd nvidia_installers
root@a28d273819ea:/ ./cuda-linux64-rel-6.5.14-18749181.run
root@a28d273819ea:/ ./cuda-samples-linux-6.5.14-18745345.run
```

Once CUDA installation is complete, exit the container and commit your changes to a new image called 'gbdx-gpu' under your username as follows:  

```bash
root@a28d273819ea:/ exit

# Commit your changes and tag the image under your username
docker commit -m 'update CUDA installation' <container id> <your_username>/gbdx-gpu
```

Run gbdx-gpu and install dependencies.  

```bash
docker run -it <your_username>/gbdx-gpu /bin/bash

# Install Python, gdal, and machine learning dependencies
root@a28d273819ea:/ apt-get update && apt-get -y install python build-essential python-software-properties software-properties-common python-pip python-scipy python-dev vim gdal-bin python-gdal libgdal-dev
root@a28d273819ea:/ pip install keras theano
```

Navigate to the home directory and create a *.theanorc* file. This will instruct Theano to use the GPU.

```bash
root@a28d273819ea:/ cd
root@a28d273819ea:# vim .theanorc
```

Paste the following into *.theanorc*:

```bash
[global]  
floatX = float32  
device = gpu  
optimizer = fast_run  

[lib]  
cnmem = 0.9

[nvcc]  
fastmath = True

[blas]  
ldflags = -llapack -lblas

[cuda]
root = /usr/local/cuda-6.5
```

Exit the container and commit your changes to gbdx-gpu.

```bash
root@a28d273819ea:# exit

docker commit -m 'install dependencies, add .theanorc' <container id> <your_username>/gbdx-gpu
```

We now have the Docker image gbdx-gpu that can run Theano or Caffe on a GPU. In the following section, we will create a GBDX task that trains a CNN classifier using the GPU.


### Convolutional Neural Network

We are going to use the tools we created [above](#using-the-gpu) to create the task 'train-cnn' that trains a [CNN](http://neuralnetworksanddeeplearning.com/chap6.html) classifier using the GPU. The task uses input images and labels to create a trained model (Figure 5).

![train_cnn_task.png]({{ site.baseurl }}/images/create-task/train_cnn_task.png)
*Figure 5: Inputs and output of train-cnn.*

train-cnn has a single directory input port [train_data](https://github.com/kostasthebarbarian/platform_stories/tree/master/create_task/train-cnn/test_input/train_data). The task expects to find the following two files within the S3 location specified by train_data:

- *X.npz*: Contains the training images as a numpy array in [npz format](http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html). The array should have the following dimensional ordering: (num_images, num_bands, img_rows, img_cols).
- *y.npz*: Contains a numpy array of numerical class labels for each training image in *X.npz*.

train-cnn also has the optional string input ports bit_depth and nb_epoch. The former specifies the bit depth of the imagery and defaults to '8' and the latter defines the number of training epochs with a default value of '10'. The task produces a trained model in the form of a model architecture file *model_architecture.json* and a trained weights file *model_weights.h5*. These two outputs will be stored in the S3 location specified by the output port trained_model.

#### The Code

The code of *train_cnn.py* is shown below.

```python
import os
import json
import numpy as np

from gbdx_task_interface import GbdxTaskInterface
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


class TrainCnn(GbdxTaskInterface):

    def invoke(self):

        # Get string inputs
        nb_epoch = int(self.get_input_string_port('nb_epoch', default = '10'))
        bit_depth = int(self.get_input_string_port('bit_depth', default = '8'))

        # Get training from input data dir
        train = self.get_input_data_port('train_data')
        X_train = np.load(os.path.join(train, 'X.npz'))['arr_0']
        y_train = np.load(os.path.join(train, 'y.npz'))['arr_0']
        nb_classes = len(np.unique(y_train))

        # Reshape for input to net, normalize based on bit_depth
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
        X_train = X_train.astype('float32')
        X_train /= float((2 ** bit_depth) - 1)

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)

        # Create basic Keras model
        model = Sequential()

        model.add(Convolution2D(32, 3, 3, border_mode='valid',
                                input_shape=(X_train.shape[1:])))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        # Fit model on input data
        model.fit(X_train, Y_train, batch_size=128, nb_epoch=nb_epoch,
              verbose=1)

        # Create the output directory
        output_dir = self.get_output_data_port('trained_model')
        os.makedirs(output_dir)

        # Save the model architecture and weights to output dir
        json_str = model.to_json()
        model.save_weights(os.path.join(output_dir, 'model_weights.h5'))
        with open(os.path.join(output_dir, 'model_architecture.json'), 'w') as arch:
            json.dump(json_str, arch)


if __name__ == '__main__':
    with TrainCnn() as task:
        task.invoke()
```

Here is what is happening in *train_cnn.py*:  

Define the **TrainCnn** class that inherits from **GbdxTaskInterface**, read the input ports, and load the training data as *X_train* and labels as *y_train*.

```python
class TrainCnn(GbdxTaskInterface):

    def invoke(self):

        # Get string inputs
        nb_epoch = int(self.get_input_string_port('nb_epoch', default = '10'))
        bit_depth = int(self.get_input_string_port('bit_depth', default = '8'))

        # Get training from input data dir
        train = self.get_input_data_port('train_data')
        X_train = np.load(os.path.join(train, 'X.npz'))['arr_0']
        y_train = np.load(os.path.join(train, 'y.npz'))['arr_0']
        nb_classes = len(np.unique(y_train))
```

Put X_train and y_train into a format that the CNN will accept during training.

```python
# Reshape for input to net, normalize based on bit_depth
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_train = X_train.astype('float32')
X_train /= float((2 ** bit_depth) - 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
```

Create a Keras CNN model. Layers are added to the model to define its architecture, then training parameters are set using ```model.compile```.  

```python
# Create basic Keras model
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid',
                        input_shape=(X_train.shape[1:])))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
```

Train the model.

```python
# Fit model on input data
model.fit(X_train, Y_train, batch_size=128, nb_epoch=nb_epoch,
      verbose=1)
```

Create *model_architecture.json* and *model_weights.h5* and save them to the output directory port.  

```python
# Create the output directory
output_dir = self.get_output_data_port('trained_model')
os.makedirs(output_dir)

# Save the model architecture and weights to output dir
json_str = model.to_json()
model.save_weights(os.path.join(output_dir, 'model_weights.h5'))
with open(os.path.join(output_dir, 'model_architecture.json'), 'w') as arch:
    json.dump(json_str, arch)
```

Call the ```invoke()``` function when the script is run.  

```python
if __name__ == '__main__':
    with TrainCnn() as task:
        task.invoke()
```

#### The Docker Image

train-cnn requires a Docker image that can access the GPU. We build the Docker image train-cnn-docker-image by pulling the Theano gbdx-gpu image that we created [above](#getting-a-gpu-compatible-image) and copying in *train_cnn.py* and *gbdx_task_interface.py*.  

Pull [naldeborgh/gbdx-gpu](https://hub.docker.com/r/naldeborgh/gbdx-gpu/) from DockerHub if you do not already have it. Tag the image under your username and rename it to train-cnn-docker-image.

```bash
docker pull naldeborgh/gbdx-gpu
docker tag naldeborgh/gbdx-gpu <your_username>/train-cnn-docker-image
```

Run train-cnn-docker-image in detached mode and copy *train_cnn.py* and *gbdx_task_interface.py*.

```bash
# Run train-cnn-docker-image in detached mode
docker run -itd <your_username>/train-cnn-docker-image
>>> <container_id>

# Copy code file to the image
docker cp train_cnn.py <container_id>:/
docker cp *gbdx_task_interface.py* <container_id>:/
```

Commit changes to train-cnn-docker-image and push it to DockerHub.

```bash
docker commit -m 'add train-cnn scripts' <container_id> <your_username>/train-cnn-docker-image
docker push <your_username>/train-cnn-docker-image
```

This image now has all of the libraries and scripts required by train-cnn. Continue on to the [next section](#testing-the-train-cnn-image) to test the image using the GPU instance created [above](#setting-up-a-device-to-test-your-gpu-image).

#### Testing the Docker Image

We will now test train-cnn-docker-image with sample input to ensure that *train_cnn.py* runs successfully AND that the GPU is utilized.

ssh into the AWS GPU instance and clone this repo so that the sample input is on the instance.

```bash
# ssh into the instance
ssh -i </path/to/key_pair> ubuntu@<your_instance_name>

# clone this repo
ubuntu@ip-00-000-00-000:~$ git clone https://github.com/kostasthebarbarian/platform_stories
```

Pull train-cnn-docker-image from your DockerHub account onto the instance.

```bash
docker pull <your_username>/train-cnn-docker-image
```

Run a container from train-cnn-docker-image. This is where testing on a GPU differs from our previous tests: you must specify which GPU devices the container should use with the ```--device``` flag:

```bash
docker run --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 -v ~/platform_stories/create_task/train-cnn/test_input:/mnt/work/input/ -it <your_username>/train-cnn-docker-image /bin/bash
```

Run *train_cnn.py*. In this step we confirm that the container is using the GPU and that the script runs without errors.

```bash
root@984b2508233b:/build# python /train_cnn.py
>>> Using Theano backend.
>>> Using gpu device 0: GRID K520 (CNMeM is enabled with initial size: 90.0% of memory, cuDNN not available)
>>> Epoch 1/2
>>> 60000/60000 [==============================] - 27s - loss: 0.3710 - acc: 0.8869
>>> Epoch 2/2
>>> 60000/60000 [==============================] - 27s - loss: 0.1292 - acc: 0.9622
```

The second line of STDOUT indicates that the container is indeed using the GPU! The lines that follow indicate that the model is being trained.

To confirm that the script was run successfully, check the output directory for the files *model_architecture.json* and *model_weights.h5*.  

```bash
# Look for the trained model in the output directory
root@984b2508233b:/build# ls /mnt/work/output/trained_model
>>> model_architecture.json    model_weights.h5
```

#### Task Definition

Defining a task that runs on a GPU is very similar to defining regular tasks. The one difference is in the containerDescriptors section: you must set the 'type' key to 'GPUDOCKER' (as opposed to 'DOCKER'), and the 'domain' property to 'gpu'. Here is the definition for train-cnn:

```json
{
    "name": "train-cnn",
    "description": "Train a convolutional neural network classifier on input imagery",
    "properties": {
        "isPublic": true,
        "timeout": 36000
    },
    "inputPortDescriptors": [
        {
            "name": "train_data",
            "type": "directory",
            "description": "S3 directory containing two files: training images (must be named 'X.npz') and associated training labels (named 'y.npz'). X.npz has shape (n_samples, n_bands, img_rows, img_cols). y.npz has shape (n_samples,)",
            "required": true
        },
        {
            "name": "nb_epoch",
            "type": "string",
            "description": "Int: number of training epochs to perform during training. Defualts to 10.",
            "required": false
        },
        {
            "name": "bit_depth",
            "type": "string",
            "description": "Int: bit depth of the input images. This parameter is necessary for proper normalization. Defaults to 8."
        }
    ],
    "outputPortDescriptors": [
        {
            "name": "trained_model",
            "type": "directory",
            "description": "The fully trained model with the architecture stored as model_arch.json and weights as model_weights.h5."
        }
    ],
    "containerDescriptors": [
        {
            "type": "GPUDOCKER",
            "properties": {
                "image": "naldeborgh/train-cnn-docker-image",
                "domain": "gpu"
            },
            "command": "python /train_cnn.py",
            "isPublic": true
        }
    ]
}
```

Now all we have to do is register train-cnn; follow the same steps as for hello-gbdx and rf-pool-classifier.

#### Executing the Task

It's time to try out train-cnn. We'll use the publicly available [MNIST dataset](http://yann.lecun.com/exdb/mnist/) which contains 60,000 images of handwritten digits to train a model to recognize handwritten digits. Figure 6 shows some example images.

![mnist.png]({{ site.baseurl }}/images/create-task/mnist.png)
*Figure 6: Sample digit images from the MNIST dataset.*

We have provided this dataset in train-cnn acceptable format on S3 under platform_stories/create_task/train_cnn.

Open an iPython terminal, create a GBDX interface and get the S3 location.

```python
from gbdxtools import Interface
from os.path
import random, string

gbdx = Interface()
bucket = gbdx.s3.info['bucket']
prefix = gbdx.s3.info['prefix']

# specify location
story_prefix = 's3://' + join(bucket, prefix, 'platform_stories', 'create_task', 'train_cnn')
```

Create a cnn_task object and specify the inputs.

```python
cnn_task = gbdx.Task('train-cnn')
cnn_task.inputs.train_data = join(story_prefix, 'train_data')
cnn_task.inputs.bit_depth = '8'
cnn_task.inputs.nb_epoch = '15'
```

Create a single-task workflow object and define where the output data should be saved.

```python
# create unique location name to save output
output_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_loc = join('platform_stories/create_task/train_cnn/user_outputs', output_str)

workflow = gbdx.Workflow([cnn_task])
workflow.savedata(cnn_task.outputs.trained_model, output_loc)
```

Execute the workflow and monitor the status as follows:

```python
workflow.execute()

# monitor workflow status
workflow.status
```

Once the workflow has completed you may download the task output and play with your trained model.

```python
gbdx.s3.download(join('platform_stories/create_task/train_cnn/user_outputs', output_str))
```

Congratulations, you have now created, registered, and executed a GBDX task on a GPU!

### Additional Resources

You may find the following links helpful:

+ [Object detection with Theano and Caffe](https://github.com/DigitalGlobe/gbdx-caffe)
+ [Building GPU docker containers for GBDX](https://github.com/ctuskdg/gbdx-gpu-docker)
+ [CUDA 6.5 on AWS GPU Instance Running Ubuntu 14.04](http://tleyden.github.io/blog/2014/10/25/cuda-6-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/)
+ [cloud-harness: Build custom GBDX tasks for beginners](http://cloud-harness.readthedocs.io/en/latest/)

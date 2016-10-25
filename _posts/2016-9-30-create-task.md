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

# Contents

1. [Background](#background)
2. [Hello GBDX](#hello-gbdx)
3. [Dockerizing](#dockerizing)
    - [About Docker](#about-docker)
    - [Creating a Docker Image](#creating-a-docker-image)
    - [Testing a Docker Image](#testing-a-docker-image)
4. [Registering a Task on GBDX](#registering-a-task-on-gbdx)
    - [Defining the Task](#defining-the-task)
    - [GBDX Task Registry](#gbdx-task-registry)
5. [Machine Learning on GBDX](#machine-learning-on-gbdx)
    - [Random Forest Classifier](#random-forest-classifier)
    - [Using the GPU](#using-the-gpu)
    - [Convolutional Neural Network](#convolutional-neural-network)
6. [Additional Resources](#additional-resources)    


# Background

A GBDX [**task**](http://gbdxdocs.digitalglobe.com/docs/task-and-workflow-course) is a process that performs a specific action on its inputs and generates a set of outputs. In the vast majority of cases, inputs and outputs consist of satellite image files (usually in tif format), vector files (shapefile, geojson), text files and various metadata files (XML, IMD and other).

Tasks can be chained together in a **workflow** where one task's outputs can be the inputs to one or more different tasks. In this manner, more complicated processes can be executed than what is possible within a single task. For example, you can imagine a ship detection workflow consisting of the following tasks: (a) pansharpen the raw satellite image (b) create a sea mask (c) look for boats in the sea.   

When a workflow is executed, tasks are scheduled appropriately by a scheduler and the system generates status indicators that are available via the GBDX API. Using the task definition in the **task registry**, each task is executed by a worker node within a docker container that contains the task code and its dependencies in an encapsulated environment. You can find additional information on GBDX in the [GBDX documentation](http://gbdxdocs.digitalglobe.com/).


# Hello GBDX

In this section, we will write a Python script for our Hello GBDX task, hello-gbdx.
The script [hello-gbdx.py](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/code/hello-gbdx.py) does the following: it obtains a list of the task input files and prints this list
in the file out.txt, along with a user defined message.
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
data_in is the task directory input port. What ```get_input_data_port('data_in')``` does behind the scenes is return the string 'mnt/work/input/data_in'.

The value of the input string port message is obtained using the ```get_input_string_port``` function (also inherited from **GbdxTaskInterface**).  

```python
message = self.get_input_string_port('message', default='No message!')
```

message is one of possibly many task string input ports. What ```get_input_string_port('message', default='No message!')``` does behind the scenes is read the value of message from the file ports.json which is found under mnt/work/input/. (Keep in mind that you don't have to worry about these inner workings if you don't want to!) If the value is not specified, it returns a default value.

The name of the output directory is obtained using the ```get_output_data_port``` function (inherited from **GbdxTaskInterface**) and the output directory is created.  

```python
# Get output
output_dir = self.get_output_data_port('data_out')
os.makedirs(output_dir)
```

data_out is the task directory output port. What ```get_output_data_port('data_out')``` does behind the scenes is return the string 'mnt/output/data_out'.  Note that **it is the responsibility of the script** to create the output directory.

out.txt is created and saved in the output directory:

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

How is data input to and output from hello-gbdx when it is executed on GBDX?
If hello-gbdx is the first in a series of tasks comprising a workflow, data_in is assigned a string value by the user which is the S3 location that contains the task input files (it will soon become apparent how to do this). These files are automatically copied to mnt/work/input/data_in of the docker container which runs hello-gbdx.py (we will cover docker in the next section). When the execution of hello-gbdx.py is concluded, out.txt is found in mnt/work/input/data_out of the docker container (our script explicitly saved it there). GBDX permits saving the contents of mnt/work/input/data_out
at a user-specified S3 location, as well as feed those to the directory input port of another task in order to chain the two tasks together. If neither of those actions are performed, the contents of mnt/work/input/data_out are lost when the task executed. In this walkthrough, we will only consider single-task workflows; you can explore Platform Stories for examples of more complicated workflows involving multiple tasks.

In the next section we go through the steps of creating a Docker image for hello-gbdx.

# Dockerizing

Dockerizing is a crucial step to get your code to run as a task on GBDX.
In this section, we will provide a high level overview of [Docker](https://www.docker.com/) and review its terminology. Next, we'll show you how to build your own hello-gbdx Docker image and run a container locally. At the end of this section, you will have an image that can be used to execute hello-gbdx, and have all of the materials necessary to register hello-gbdx on GBDX.

## About Docker

Docker is a software containerization platform that allows developers to package up an application with its dependencies, and deliver it to a user in a single, self-sufficient package (referred to as a container).

![docker_workflow.png]({{ site.baseurl }}/images/create-task/docker_workflow.png)
*Figure 1: Docker allows you to deliver various libraries and scripts in a lightweight package. Docker is required to create a task on GBDX.*

### Docker and GBDX

When a task is run on GBDX, a Docker container containing the task code, all its required dependencies, and the operating system is run by a worker node. Docker provides an efficient method for delivering the task code and its dependencies to the worker in an encapsulated environment.

### Docker Lingo

Because Docker can be confusing if you are not used to it, we will define some terms you will encounter in this tutorial. You can consult the [Docker glossary](https://docs.docker.com/engine/reference/glossary/#base-image) for more information.

* <b>Image</b>: Docker images are the basis of containers. An image is a static, immutable file that describes the container environment. Running an image produces a container.

* <b>Container</b>: A container is a runtime instance of a Docker image (similar to an object being an instantiation of a class in Python).

* <b>DockerFile</b>: A Dockerfile is a text document that contains all the commands you would normally execute manually in order to build a Docker image. Docker can build images automatically by reading the instructions from a Dockerfile.
It is only necessary to create a DockerFile if you elect to create your image from scratch.

* <b>DockerHub</b>: A repository of images. You can pull and edit these images for your own use, analogous to cloning or forking a GitHub repo.

## Creating a Docker Image

Before creating your image [sign up](https://hub.docker.com/) for a DockerHub account and follow [these](https://docs.docker.com/engine/getstarted/step_one/) instructions to install Docker toolbox on your machine. This will allow you to create, edit, and push your Docker images locally from the command line. In this section, we review two methods for creating a Docker image:

- [Pulling an existing image from DockerHub](#pulling-an-image-from-dockerhub) and [editing it](#adding-your-code-to-the-image) to include additional packages and your code.

- [Building an image from scratch](#building-an-image-with-a-dockerfile) using a DockerFile. If you elect to go this route you can skip the next two sections. (You may wish to read through them anyway as it is useful to know how to edit a Docker image without use of a DockerFile.)

### Pulling an Image from DockerHub

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

Finally, you can tag the image under your username. This enables you to edit the image and push it to your personal DockerHub repository. Name the image hello-gbdx-docker-image, as we will be moving our code into it shortly.

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

### Adding Your Code to the Image

![docker_commit_wf.png]({{ site.baseurl }}/images/create-task/docker_commit_wf.png)
*Figure 2: Adding code to a docker image.*

The following steps (shown in Figure 2) walk you through adding [hello-gbdx.py](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/code/hello-gbdx.py) and [gbdx_task_interface.py](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/code/gbdx_task_interface.py) to hello-gbdx-docker-image. Before getting started ensure that both scripts are saved to your current working directory.

First we run the image using the ```docker run``` command in detached mode (using the -d flag). This tells the container to run in the background so we can access the files on our local machine.

```bash
docker run -itd <your_username>/hello-gbdx-docker-image
>>> ff567ca72fa0ed6cdfbe0a5c02ea3e04f88ec49239344f217ce1049651d01344
```

The value returned is the container id. Make note of this because we will need it.

We now use the ```docker cp``` command to copy our scripts into the container. The format of this command is as follows: ```docker cp <filename> <container_id>:<container_dest_path>```.

```bash
# Copy hello-gbdx.py to the root directory of the container
docker cp hello-gbdx.py <container_id>:/
docker cp gbdx_task_interface.py <container_id>:/
```

Our scripts are now in the container. You may confirm this by attaching to the container (bringing it back to the foreground) as follows:

```bash
docker attach <container_id>

# Notice that the scripts now live in the root directory of the container
root@ff567ca72fa0:/ ls
gbdx_task_interface.py   hello-gbdx.py   boot  etc   lib    media  opt   root  sbin  sys  usr  bin  dev   home  lib64  mnt    proc  run   srv   tmp  var
```

You may detach from the container (sending it back to background) without stopping it using the following escape sequence: <kbd>Ctrl</kbd>-<kbd>p</kbd> + <kbd>Ctrl</kbd>-<kbd>q</kbd>  

If we were to stop the container now, all of our changes would be lost and hello-gbdx-docker-image image would remain unchanged. To permanently update hello-gbdx-docker-image, we must commit our changes to it.

```bash
# Commit the changes from the container to the image
docker commit -m 'add scripts to root' <container_id> <your_username>/hello-gbdx-docker-image
```

Now when you run hello-gbdx-docker-image, hello-gbdx.py and gbdx_task_interface.py will be in the root directory.

You can also push your new image to DockerHub in case you need to pull it in the future.

```bash
# Push the changes up to DockerHub
docker push <your_username>/hello-gbdx-docker-image
```

Keep in mind that, although hello-gbdx does not require any additional libraries to run, often times you will need to install a package that is not provided in the image that you pulled. Let's say our task requires numpy to run; the process of adding it to the image is similar:

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

### Building an Image with a DockerFile

Feeling a little ambitious? You can build your own image from scratch with a DockerFile.
For more information on DockerFiles see [here](https://docs.docker.com/engine/reference/builder/) and [here](https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/).

![dockerfile.png]({{ site.baseurl }}/images/create-task/dockerfile.png)
*Figure 3: Building a docker image with a DockerFile.*

In this example, we will build hello-gbdx-docker-image.

We begin by making the directory [hello-gbdx-build](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/hello-gbdx-build), which will contain our DockerFile, and the subdirectory bin in which we copy hello-gbdx.py and gbdx_task_interface.py.

```bash
# Make build an bin directories
mkdir hello-gbdx-build/
cd hello-gbdx-build/
mkdir bin/

# Copy both scripts for hello-gbdx into the bin/ directory
cp path/to/hello-gbdx.py bin/
cp path/to/gbdx_task_interface.py bin/
```

Now we can create our DockerFile. From within the hello-gbdx-build directory, type ```vim Dockerfile``` (or whatever code editor you prefer) to create a blank document. The first line of the DockerFile is the OS that your image uses. We will be working with Ubuntu. Type the following in the first line of the file:

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

We instruct Docker to place the contents of [bin](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/hello-gbdx-build/bin) into the image root directory. Add the following line to the end of Dockerfile:

```bash
# Add all scripts in bin to the image root directory
ADD ./bin /
```

Our DockerFile is now [complete]((https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/hello-gbdx-build/Dockerfile)). Exit vim with the ```:wq``` command.

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

## Testing a Docker Image

At this point you should have hello-gbdx-docker-image which includes hello-gbdx.py.
In this section, we will run this image with actual input data. Successfully doing this locally ensures that hello-gbdx will run on GBDX. [hello-gbdx/sample-input in this repo](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/sample-input) contains the two inputs required by hello-gbdx: (a) the directory [data_in](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/sample-input/data_in), the contents of which will be written to out.txt (in this example, this is simply the file data_file.txt) (b) the file [ports.json](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/sample-input/ports.json) which
contains the message to be written to out.txt. Keep in mind that ports.json is automatically created by GBDX based on the task definition and the values of the string input ports provided by the user when the task is executed.

Run hello-gbdx-docker-image and mount inputs to the container under mnt/work/input; this is where GBDX will place the inputs when the task is executed.

```bash
docker run -v ~/path/to/hello-gbdx/sample-input:/mnt/work/input -it <your_username>/hello-gbdx-docker-image
```

Note the important distinction between mounting data to the container and adding data to the image using the ADD command in the Dockerfile: when you exit the container, this data 'disappears' (i.e., it is not saved onto the image).

Confirm that the inputs are mounted by exploring the container.

```bash
# Look at the contents of the input directory. data_in should be mounted.
root@3ad24b35e32e:/ ls /mnt/work/input/
>>> data_in  ports.json
```

To test hello-gbdx, simply run the hello-gbdx.py script.

```bash
root@3ad24b35e32e:/ python hello-gbdx.py
```

If the script completes successfully you shouldn't see anything written to STDOUT and the file out.txt should be found under mnt/work/output/. Here is how you can confirm this:

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

You can also make sure out.txt contains the expected content by typing ```vim out.txt```. The file should look like this:

```bash
data_file.txt
This is my message!
```

Congratulations, your task is working as expected! The next step is to [create a task definition](#defining-the-task), which will be used to [register](#registering-a-task-on-gbdx) hello-gbdx on the platform.


# Registering a Task on GBDX

Now that we have hello-gbdx-docker-image working locally, we can finally define hello-gbdx and then register it to the GBDX task registry.

## Defining the Task

The task definition is a [json file](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/hello-gbdx-definition.json) that contains a description of the task functionality, a list of its inputs and outputs, and the Docker image that needs to be run when the task is executed.

```json
{
    "name": "hello-gbdx",
    "description": "Writes list of the input file names and a user defined message to output file out.txt.",
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
            "description": "Input data directory.",
            "required": true
        }
    ],
    "outputPortDescriptors": [
        {
            "name": "data_out",
            "type": "directory",
            "description": "Output data directory."
        }
    ],
    "containerDescriptors": [
        {
            "type": "DOCKER",
            "properties": {
                "image": "naldeborgh/hello-gbdx-docker-image"
            },
            "command": "python /hello-gbdx.py",
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
    "description": "Writes list of the input file names and a user defined message to output file out.txt.",
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
        "description": "Input data directory.",
        "required": true
    }
```

- <b>name</b>: The input port name.
- <b>type</b>: The input port type. Currently the only options are 'directory' and 'string'. A directory input port is used to define an S3 location where input files are stored or to hook to the output directory port of a previous task. A string input port is used to pass a string parameter to the task. Note that integers, floats and booleans must all be provided to a task in string format!
- <b>description</b>: Description of the input port.
- <b>required</b>: A boolean. 'true'/'false' indicate required/optional input, respectively.

<b> Output Port Descriptors</b>: This is where the task output ports are defined.

```
"outputPortDescriptors": [
    {
        "name": "data_out",
        "type": "directory",
        "description": "Output data directory."
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
        "command": "python /hello-gbdx.py",
        "isPublic": true
```

- <b>type</b>: The domain on which the task is run. Typical tasks are run on the 'DOCKER' domain. Change this to 'GPUDOCKER' for tasks that require a GPU to run (more on this [later](#the-gpu-definition)).
- <b>image</b>: The name of the Docker image that is pulled from DockerHub.
- <b>command</b>: The command to run within the container.


## GBDX Task Registry

We now have all the required material to register hello-gbdx: [a Docker image on DockerHub](#creating-a-docker-image) and the [task definition](https://github.com/PlatformStories/create-task/tree/master/hello-gbdx/hello-gbdx-definition.json).

Open an iPython terminal, import gbdxtools and start up a GBDX Interface.  

```python
from gbdxtools import Interface

gbdx = Interface()
```

Call the ```register()``` method of the **TaskRegistry** class with the name of the definition JSON. (Make sure hello-gbdx-definition.json is in your working directory).

```python
gbdx.task_registry.register(json_filename = 'hello-gbdx-definition.json')
>>> u'hello-gbdx successfully registered.'
```

There's a good chance that hello-gbdx already exists in the registry. You can try using a different name after appropriaterly modifying the definition.

Congratulations, you have just registered hello-gbdx! You can run it with sample data as follows. Open an ipython terminal and copy in the following:

```python
from gbdxtools import Interface
from os.path import join
import string, random
gbdx = Interface()

# specify S3 location of input files
input_location = 's3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform-stories/create-task/hello-gbdx'

# create task object
hello_task = gbdx.Task('hello-gbdx')

# set the value of data_in
hello_task.inputs.data_in = join(input_location, 'data_in')

# set the value fo the input string port
hello_task.inputs.message = 'This is my message!'

# define a single-task workflow
workflow = gbdx.Workflow([hello_task])

# save contents of data_out in platform-stories/trial-runs/random_str within your bucket/prefix
output_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_location = join('platform-stories/trial-runs', output_str)
workflow.savedata(hello_task.outputs.data_out, output_location)
```

Execute the workflow and monitor its status as follows:

```python
workflow.execute()
workflow.status
```

When the workflow is complete, you can download out.txt locally as follows

```python
gbdx.s3.download(output_location)
```

To delete hello-gbdx from the registry:

```python
gbdx.task_registry.delete('hello-gbdx')
>>> u'hello-gbdx successfully deleted.'
```

You have created a basic yet fully functional GBDX task using Docker and gbdxtools. The next section covers the process of creating more complicated tasks that can run machine learning algorithms.


# Machine Learning on GBDX

In the last section, we created a simple task that generates a text file with a list of the contents of the input directory and a user defined message. Chances are you're looking to do a bit more with your task. In this section, you will learn how to create a task that runs a standard machine learning (ML) algorithm such as a random forest classifier, and how to setup a ML task that utilizes a Convolutional Neural Network (CNN) so that it can be executed on a worker with a GPU.

## Random Forest Classifier

In this example, we will create the task rf-pool-classifier that trains a [random forest classifier](https://en.wikipedia.org/wiki/Random_forest) to classify polygons of arbitrary geometry into those that contain swimming pools and those that don't. For more information on this algorithm see [here](https://github.com/DigitalGlobe/mltools/tree/master/examples/polygon_classify_random_forest) and [here](http://blog.tomnod.com/crowd-and-machine-combo).

![rf_img.png]({{ site.baseurl }}/images/create-task/rf_img.png)
*Figure 4: Inputs and output of rf-pool-classifier.*

rf-pool-classifier has two directory input ports: geojson and image. Within geojson and image, the task expects to find a file train.geojson, which contains labeled polygons from both classes, and a tif image file from which the task will extract the pixels corresponding to each polygon, respectively (Figure 4). The task also has the input string port n_estimators that determines the number of trees in the random forest; specifying a value is optional and the default is '100'. The task produces a trained model in [pickle](https://docs.python.org/2/library/pickle.html) format, which is saved in the S3 location specified by the output port trained_classifier.

### The Code

The code of rf-pool-classifier.py is shown below; the structure is the same as hello-gbdx.py.

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

Using the ```mltools.data_extractors``` modules, the pixels corresponding to each polygon in train.geojson are extracted and stored in a masked numpy array. For each array, a 4-dim feature vector is computed by the function ```features.pool_basic``` and stored in the list X.

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

### The Docker Image

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

We are now ready to copy rf-pool-classifier.py and gbdx_task_interface.py to rf-pool-classifier-docker-image. Make sure to have the script saved to your working directory and execute the following:

```bash
# Run Docker in detached mode
docker run -itd <your_username>/rf-pool-classifier-docker-image

# Copy the script to the container and commit the changes
docker cp rf-pool-classifier.py <container_id>:/
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

The DockerFile and scripts can be found [here](https://github.com/PlatformStories/create-task/tree/master/rf-pool-classifier/rf-pool-classifier-build).

### Testing the Docker Image

We can now test rf-pool-classifier-docker-image on our local machine before defining rf-pool-classifier and registering it on the platform. Just as in the case of [hello-gbdx](#testing-a-docker-image), we will mimic the platform by mounting sample input to a container and then executing rf-pool-classifier.py.

Create the directory rf_pool_classifier_test in which to download the task inputs from S3. You will need subdirectories for geojson and image.

```bash
# create input port directories
mkdir rf_pool_classifier_test
cd rf_pool_classifier_test
mkdir geojson
mkdir image
```

From an iPython terminal, download 1040010014800C00.tif and train.geojson from their S3 locations as follows.
Note that the tif is large (~15 GB) so you will need adequate disk space.

```python
from gbdxtools import Interface
from os.path import join
gbdx = Interface()

input_location = 's3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform-stories/create-task/rf-pool-classifier'

# download the image strip (will take a couple minutes)
gbdx.s3.download(join(input_location, 'image'), './image/')

# download the geojson with training data
gbdx.s3.download(join(input_location, 'geojson'), './geojson/')

# exit iPython
exit
```

Run rf-pool-classifier-docker-image with rf_pool_classifier_test mounted to the input port.

```bash
docker run -v ~/<full/path/to/rf_pool_classifier_test>:/mnt/work/input -it <your_username>/rf-pool-classifier-docker-image
```

Within the container run rf-pool-classifier.py.

```bash
python /rf-pool-classifier.py
```

The script should run without errors. To confirm this, check the output port directory for classifier.pkl.

```bash
root@91d9d5cd9570:/ ls mnt/work/output/trained_classifier
>>> classifier.pkl
```

You can now define and register rf-pool-classifier!

### Task Definition

The definition for rf-pool-classifier is provided below:

```json
{
    "name": "rf-pool-classifier",
    "description": "Train a random forest classifier to classify polygons in those that contain pools and those that do not.",
    "properties": {
        "isPublic": true,
        "timeout": 7200
    },
    "inputPortDescriptors": [
        {
            "name": "image",
            "type": "directory",
            "description": "Contains the image strip where the polygons are found.",
            "required": true
        },
        {
            "name": "geojson",
            "type": "directory",
            "description": "Contains a geojson with labeled polygons. Each polygon has the properties feature_id, image_id, and class_name (either 'No swimming pool' or 'Swimming pool')",
            "required": true
        },
        {
            "name": "n_estimators",
            "type": "string",
            "description": "Number of trees to use in the random forest classifier. Defaults to 100.",
            "required": false
        }
    ],
    "outputPortDescriptors": [
        {
            "name": "trained_classifier",
            "type": "directory",
            "description": "Contains the file 'classifier.pkl' which is the trained random forest classifier."
        }
    ],
    "containerDescriptors": [
        {
            "type": "DOCKER",
            "properties": {
                "image": "naldeborgh/rf-pool-classifier-docker-image"
            },
            "command": "python /rf-pool-classifier.py",
            "isPublic": true
        }
    ]
}
```

Put rf-pool-classifier-definition.json in your working directory and register rf-pool-classifier as follows:

```python
from gbdxtools import Interface
gbdx = Interface()

# register the task using rf-pool-classifier-definition.json
gbdx.task_registry.register(json_filename = 'rf-pool-classifier-definition.json')
```

### Executing the Task

We will now run through a sample execution of rf-pool-classifier using gbdxtools.

Open an iPython terminal, create a GBDX interface and specify the task input location.

```python
from gbdxtools import Interface
from os.path import join
import random, string

gbdx = Interface()

# specify location
input_location = 's3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform-stories/create-task/rf-pool-classifier'
```

Create an rf_task object and specify the inputs.

```python
rf_task = gbdx.Task('rf-pool-classifier')
rf_task.inputs.image = join(input_location, 'image')
rf_task.inputs.geojson = join(input_location, 'geojson')
rf_task.inputs.n_estimators = "1000"
```

Create a single-task workflow object and define where the output data should be saved.

```python
# set output location to platform-stories/trial-runs/random_str within your bucket/prefix
random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_location = join('platform-stories/trial-runs', random_str)

workflow = gbdx.Workflow([rf_task])
workflow.savedata(rf_task.outputs.trained_classifier, output_location)
```

Execute the workflow and monitor its status as follows:

```python
workflow.execute()
workflow.status
```

Once the workflow is completed, you can download classifier.pkl locally as follows:

```python
gbdx.s3.download(output_location)
```

Done! At this point we have created and executed a simple ML task on the platform.
In the next section, we will cover how to make use of the GPU for compute-intensive algorithms
that rely on it.


## Using the GPU

Until now, we have been running our tasks on a CPU device. For certain ML applications such as deep learning that are very compute-intensive, the GPU offers order-of-magnitude performance improvement compared to the CPU. GBDX provides the capability of running a task on a GPU worker. This requires configuring the Docker image and defining the task appropriately.

This section will walk you through setting up a local GPU instance for building and testing your Docker image, and then building a GPU-compatible Docker image, i.e., a Docker image that can access the GPU on the node on which it is run.

### Setting up a GPU instance

All GBDX GPU workers use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#nvidia-docker) to allow running containers to leverage their GPU devices. Here are the steps for setting up an AWS instance with nvidia-docker, which you will need to test your GPU-compatible Docker image.

On AWS launch ami-d05e75b8. Choose a GPU instance of type [EC2 g2.2xlarge](https://aws.amazon.com/ec2/instance-types/#g2). At least 20GB of storage is recommended. Then ssh into your instance and install CUDA repository.

```bash
ssh -i <path/to/key_pair> ubuntu@<instance_id>

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
```

Update APT and install dependencies.

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y opencl-headers build-essential protobuf-compiler \
    libprotoc-dev libboost-all-dev libleveldb-dev hdf5-tools libhdf5-serial-dev \
    libopencv-core-dev  libopencv-highgui-dev libsnappy-dev libsnappy1 \
    libatlas-base-dev cmake libstdc++6-4.8-dbg libgoogle-glog0 libgoogle-glog-dev \
    libgflags-dev liblmdb-dev git python-pip gfortran

# Clean up
sudo apt-get clean
```

Specify DRM module version.

```bash
sudo apt-get install -y linux-image-extra-`uname -r` linux-headers-`uname -r` linux-image-`uname -r`
```

Install CUDA.  

```bash
sudo apt-get install -y cuda
sudo apt-get clean
```

Verify CUDA installation.  

```bash
nvidia-smi

# You should see the following output:
+------------------------------------------------------+
| NVIDIA-SMI 361.93.02  Driver Version: 361.93.02      |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GRID K520           Off  | 0000:00:03.0     Off |                  N/A |
| N/A   30C    P0    36W / 125W |     11MiB /  4095MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Install Docker on your image, beginning with a couple prerequisites.

```bash
sudo apt-get -f install
sudo apt-get install apt-transport-https ca-certificates

# Add the new GPG key
sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
```

Create a docker.list file:

```bash
sudo vim /etc/apt/sources.list.d/docker.list
```

Replace any existing entries in the file with the following:

```bash
deb https://apt.dockerproject.org/repo ubuntu-trusty main
```

Purge the old repo if it exists.

```bash
sudo apt-get purge lxc-docker
```

Install Docker.

```bash
sudo apt-get update
sudo apt-get install docker-engine
```

Instruct docker to run without sudo privileges.

```bash
sudo service docker start
sudo groupadd docker
sudo usermod -aG docker $USER
```

Ensure Docker was installed properly by running the 'hello-world' container. You should see a message indicating that the installation was successful.

```bash
docker run hello-world
```

Finally, install nvidia-docker on the instance.

```bash
# Install nvidia-docker and nvidia-docker-plugin
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# Test nvidia-smi
nvidia-docker run --rm nvidia/cuda nvidia-smi

# You should see the following output:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 361.93.02              Driver Version: 361.93.02                 |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GRID K520           Off  | 0000:00:03.0     Off |                  N/A |
| N/A   30C    P8    17W / 125W |      0MiB /  4036MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Building a GPU-Compatible Image

Now that we have an instance on which to test our GPU tasks, we can create a Docker image that can access the GPU. To do this we simply use the Docker image [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/) as a base and install any necessary dependencies. nvidia/cuda is set up such that it can run seamlessly on any device using the nvidia-docker plugin, saving us the headache of matching drivers between the Docker image, GBDX worker nodes, and our GPU instance.

Here we will create a Docker image 'gbdx-gpu' with the [Theano](http://deeplearning.net/software/theano/) and [Keras](https://keras.io/) Python libraries installed, as these are required by the example task presented in the next [section](#convolutional-neural-network). While still logged into the AWS instance, execute the following steps:

Login to Docker on the running instance.

```bash
docker login -u <your_username> -p <your_password>
```

Pull the nvidia/cuda image created for CUDA 7.5 from DockerHub and tag it under a new name.

```bash
docker pull nvidia/cuda:7.5-cudnn4-devel
docker tag nvidia/cuda:7.5-cudnn4-devel <your_username>/gbdx-gpu
```

Run a container from the image.

```bash
docker run -it <your_username>/gbdx-gpu /bin/bash
```

Install Python libraries and other machine learning dependencies.

```bash
root@a28d273819ea:/ apt-get update && apt-get -y install python build-essential python-software-properties software-properties-common python-pip python-scipy python-dev vim gdal-bin python-gdal libgdal-dev
root@a28d273819ea:/ pip install keras theano h5py
```

Create a .theanorc file in the root directory. This will instruct Theano to use the GPU.

```bash
root@a28d273819ea:# vim /root/.theanorc
```

Paste the following into .theanorc:

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
root = /usr/local/cuda
```

Save the .theanorc file and now update the Keras config file. This will instruct Keras to use the Theano backend.

```bash
root@a28d273819ea:# vim /root/.keras/keras.json
```

Remove any contents of the file and paste the following:

```bash
{
"image_dim_ordering": "th",
"epsilon": 1e-07,
"floatx": "float32",
"backend": "theano"
}
```

Exit the container and commit your changes to gbdx-gpu.

```bash
root@a28d273819ea:# exit

docker commit -m 'install dependencies, add .theanorc' <container id> <your_username>/gbdx-gpu
```

We now have the Docker image gbdx-gpu that can run Theano and Keras on a GPU. See [here](https://github.com/PlatformStories/create-task/tree/master/train-cnn/train-cnn-build) for how to build this image with a DcockerFile. In the following section, we will create a GBDX task that trains a CNN classifier using the GPU.


## Convolutional Neural Network

We are going to use the tools we created [above](#using-the-gpu) to create the task 'train-cnn' that trains a [CNN](http://neuralnetworksanddeeplearning.com/chap6.html) classifier using the GPU. The task uses input images and labels to create a trained model (Figure 5).

![train_cnn_task.png]({{ site.baseurl }}/images/create-task/train_cnn_task.png)
*Figure 5: Inputs and output of train-cnn.*

train-cnn has a single directory input port [train_data](https://github.com/PlatformStories/create-task/tree/master/train-cnn/sample-input/train_data). The task expects to find the following two files within train_data:

- **X.npz**: Training images as a numpy array saved in [npz format](http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html). The array should have the following dimensional ordering: (num_images, num_bands, img_rows, img_cols).
- **y.npz**: Class labels corresponding to training images as a numpy array saved in npz format.

train-cnn also has the optional string input ports bit_depth and nb_epoch. The former specifies the bit depth of the imagery and defaults to '8' and the latter defines the number of training epochs with a default value of '10'. The task produces a trained model in the form of a model architecture file model_architecture.json and a trained weights file model_weights.h5. These two outputs will be stored in the S3 location specified by the output port trained_model.

### The Code

The code of train-cnn.py is shown below.

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

Here is what is happening in train-cnn.py:  

Define the **TrainCnn** class that inherits from **GbdxTaskInterface**, read the input ports, and load the images and labels to X_train and y_train, respectively.

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

Create model_architecture.json and model_weights.h5 and save them to the output directory port.  

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

### The Docker Image

train-cnn requires a Docker image that can access the GPU. We build the Docker image train-cnn-docker-image by pulling the Theano gbdx-gpu image that we created [above](#getting-a-gpu-compatible-image) and copying in train-cnn.py and gbdx_task_interface.py.  

Pull [naldeborgh/gbdx-gpu](https://hub.docker.com/r/naldeborgh/gbdx-gpu/) from DockerHub if you do not already have it. Tag the image under your username and rename it to train-cnn-docker-image.

```bash
docker pull naldeborgh/gbdx-gpu
docker tag naldeborgh/gbdx-gpu <your_username>/train-cnn-docker-image
```

Run train-cnn-docker-image in detached mode and copy train-cnn.py and gbdx_task_interface.py.

```bash
# Run train-cnn-docker-image in detached mode
docker run -itd <your_username>/train-cnn-docker-image
>>> <container_id>

# Copy code file to the image
docker cp train-cnn.py <container_id>:/
docker cp gbdx_task_interface.py <container_id>:/
```

Commit changes to train-cnn-docker-image and push it to DockerHub.

```bash
docker commit -m 'add train-cnn scripts' <container_id> <your_username>/train-cnn-docker-image
docker push <your_username>/train-cnn-docker-image
```

This image now has all of the libraries and scripts required by train-cnn. See [here](https://github.com/PlatformStories/create-task/tree/master/train-cnn/train-cnn-build) to see a sample build of train-cnn-docker-image using a DockerFile. Continue on to the [next section](#testing-the-train-cnn-image) to test the image using the GPU instance created [above](#setting-up-a-device-to-test-your-gpu-image).

### Testing the Docker Image

We will now test train-cnn-docker-image with sample input to ensure that train-cnn.py runs successfully AND that the GPU is utilized.

ssh into the AWS GPU instance and clone [this repo](https://github.com/PlatformStories/create-task) so that the sample input is on the instance.

```bash
# ssh into the instance
ssh -i </path/to/key_pair> ubuntu@<your_instance_name>

# clone create-task repo
ubuntu@ip-00-000-00-000:~$ git clone https://github.com/PlatformStories/create-task
```

Pull train-cnn-docker-image from your DockerHub account onto the instance.

```bash
docker pull <your_username>/train-cnn-docker-image
```

Run a container from train-cnn-docker-image. This is where testing on a GPU differs from our previous tests: you must specify which GPU devices the container should use, defined by ```curl http://localhost:3476/v1.0/docker/cli```.

```bash
docker run `curl http://localhost:3476/v1.0/docker/cli` -v \
    ~/PlatformStories/create-task/train-cnn/sample-input:/mnt/work/input/ \
    -it <your_username>/train-cnn-docker-image /bin/bash
```

Run train-cnn.py. In this step we confirm that the container is using the GPU and that the script runs without errors.

```bash
root@984b2508233b:/build# python /train-cnn.py

# You should see the following output
Using Theano backend.
Using gpu device 0: GRID K520 (CNMeM is enabled with initial size: 90.0% of memory, cuDNN 4008)
Epoch 1/2
60000/60000 [==============================] - 7s - loss: 0.3776 - acc: 0.8827
Epoch 2/2
60000/60000 [==============================] - 7s - loss: 0.1431 - acc: 0.9570
```

The second line of STDOUT indicates that the container is indeed using the GPU! The lines that follow indicate that the model is being trained.

To confirm that the script was run successfully, check the output directory for the files model_architecture.json and model_weights.h5.  

```bash
# Look for the trained model in the output directory
root@984b2508233b:/build# ls /mnt/work/output/trained_model
>>> model_architecture.json    model_weights.h5
```

### Task Definition

Defining a task that runs on a GPU is very similar to defining regular tasks. The one difference is in the containerDescriptors section: you must set the 'domain' property to 'nvidiagpu'. Here is the definition for train-cnn:

```json
{
    "name": "train-cnn",
    "description": "Train a convolutional neural network classifier on the MNIST data set.",
    "properties": {
        "isPublic": true,
        "timeout": 36000
    },
    "inputPortDescriptors": [
        {
            "name": "train_data",
            "type": "directory",
            "description": "Contains training images X.npz and corresponding labels y.npz.",
            "required": true
        },
        {
            "name": "nb_epoch",
            "type": "string",
            "description": "Number of training epochs to perform during training. Defaults to 10.",
            "required": false
        },
        {
            "name": "bit_depth",
            "type": "string",
            "description": "Bit depth of the input images. This parameter is necessary for proper normalization. Defaults to 8."
        }
    ],
    "outputPortDescriptors": [
        {
            "name": "trained_model",
            "type": "directory",
            "description": "Contains the fully trained model with the architecture stored as model_arch.json and the weights as model_weights.h5."
        }
    ],
    "containerDescriptors": [
        {
            "type": "DOCKER",
            "properties": {
                "image": "naldeborgh/train-cnn-docker-image",
                "domain": "nvidiagpu"
            },
            "command": "python /train-cnn.py",
            "isPublic": true
        }
    ]
}

```

Now all we have to do is register train-cnn; follow the same steps as for hello-gbdx and rf-pool-classifier.

### Executing the Task

It's time to try out train-cnn. We'll use the publicly available [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 images of handwritten digits to train a model to recognize handwritten digits. Figure 6 shows some example images.

![mnist.png]({{ site.baseurl }}/images/create-task/mnist.png)
*Figure 6: Sample digit images from the MNIST dataset.*

Open an iPython terminal, create a GBDX interface and get the input location.

```python
from gbdxtools import Interface
from os.path import join
import random, string

gbdx = Interface()

input_location = 's3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform-stories/create-task/train-cnn'
```

Create a cnn_task object and specify the inputs.

```python
cnn_task = gbdx.Task('train-cnn')
cnn_task.inputs.train_data = join(input_location, 'train_data')
cnn_task.inputs.bit_depth = '8'
cnn_task.inputs.nb_epoch = '15'
```

Create a single-task workflow object and specify where the output data should be saved.

```python
# set output location to platform-stories/trial-runs/random_str within your bucket/prefix
random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_location = join('platform-stories/trial-runs', random_str)

workflow = gbdx.Workflow([cnn_task])
workflow.savedata(cnn_task.outputs.trained_model, output_location)
```

Execute the workflow and monitor the status as follows:

```python
workflow.execute()

# monitor workflow status
workflow.status
```

Once the workflow is completed, download the trained model:

```python
gbdx.s3.download(output_location)
```

Congratulations, you have now created, registered, and executed a GBDX task on a GPU!

# Additional Resources

You may find the following links helpful:

+ [Object detection with Theano and Caffe](https://github.com/DigitalGlobe/gbdx-caffe)
+ [Building GPU docker containers for GBDX](https://github.com/DigitalGlobe/gbdx-gpu-docker)
+ [CUDA 6.5 on AWS GPU Instance Running Ubuntu 14.04](http://tleyden.github.io/blog/2014/10/25/cuda-6-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/)
+ [cloud-harness: Build custom GBDX tasks for beginners](http://cloud-harness.readthedocs.io/en/latest/)

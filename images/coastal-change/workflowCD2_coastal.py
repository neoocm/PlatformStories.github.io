#!/usr/bin/python

###############################################################################
#
#  Script     : workflowCD2_coastal.py
#
#  Created by : Georgios Ouzounis
#               Data Science Applications Team, DigitalGlobe
#
#  Version 1.0: Aug 29, 2016
#               - workflowCD2_coastal is the second of the two workflows  
#                 designed for binary change detection;
#               - delivers a coastal change map (tristate) and its division into gains and losses;
#               - delivers the distance map of the gains and losses layers
#               - it takes as input two images (pre & post) after going through
#                 the AOP;   
#               - base code: PROTOGEN vs.2.0.0
#
#  Usage      : python workflowCD2_coastal.py newimage oldimage myBucket
#
#  Example    : python workflowCD2_coastal.py 
#                	s3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform_stories/coastal_change/images/post 
#                	s3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform_stories/coastal_change/images/pre 
#               	platform_stories/coastal_change  #(no '/' at the end)
#
###############################################################################


import sys
import os
import fnmatch
import gbdxtools


if len(sys.argv) < 4 or len(sys.argv) > 4:
   sys.exit("USAGE: python workflowCD2_coastal.py indir1 indir2 myBucket\n")

# #############################################################################
# ### mange the inputs & get prefixes for future use                        ###
# #############################################################################

postimage             = sys.argv[1]
preimage              = sys.argv[2]
myBucket              = sys.argv[3]

# #############################################################################
# ### run CDREADY task                                                      ###
# #############################################################################

gbdx                  = gbdxtools.Interface()

ipa                   = gbdx.Task('protogenV2CD_READY')
ipa.inputs.raster     = postimage
ipa.inputs.slave      = preimage

# #############################################################################
# ### chain in the water layer creation tasks for the pre & post images     ###
# #############################################################################
				
water_pre                = gbdx.Task('protogenV2RAW')
water_pre.inputs.raster  = ipa.outputs.slave.value
				
water_post               = gbdx.Task('protogenV2RAW')
water_post.inputs.raster = ipa.outputs.data.value

# #############################################################################
# ### chain in the CD_LULC task                                             ###
# #############################################################################

exclusion_mask                = gbdx.Task('protogenV2CD_LULC')
exclusion_mask.domain         = 'raid'
exclusion_mask.inputs.raster  = ipa.outputs.data.value
exclusion_mask.inputs.slave   = ipa.outputs.slave.value

# #############################################################################
# ### chain in the binary change tasks: CD_BIN_TRI,CD_BIN_GAIN,CD_BIN_LOSS  ###
# #############################################################################

cd_tri = gbdx.Task('protogenV2CD_BIN_TRI')
cd_tri.inputs.raster = rawpost.outputs.data.value
cd_tri.inputs.slave  = water_pre.outputs.data.value
cd_tri.inputs.mask   = exclusion_mask.outputs.data.value

cd_gain = gbdx.Task('protogenV2CD_BIN_GAIN')
cd_gain.inputs.raster= water_post.outputs.data.value
cd_gain.inputs.slave = water_pre.outputs.data.value
cd_gain.inputs.mask  = exclusion_mask.outputs.data.value

cd_loss = gbdx.Task('protogenV2CD_BIN_LOSS')
cd_loss.inputs.raster= water_post.outputs.data.value
cd_loss.inputs.slave = water_pre.outputs.data.value
cd_loss.inputs.mask  = exclusion_mask.outputs.data.value

# #############################################################################
# ### chain in the CD_GDDT and CD_LDDT tasks                                ###
# #############################################################################

ddt_gain = gbdx.Task('protogenV2CD_GDDT')
ddt_gain.inputs.raster = water_post.outputs.data.value
ddt_gain.inputs.slave  = cd_gain.outputs.data.value

ddt_loss = gbdx.Task('protogenV2CD_LDDT')
ddt_loss.inputs.raster = water_post.outputs.data.value
ddt_loss.inputs.slave  = cd_loss.outputs.data.value


# #############################################################################
# ### Do the workflow thing ...                                             ###
# #############################################################################

# create workflow from constituent tasks
wf = gbdx.Workflow([ipa, water_pre, water_post, exclusion_mask, cd_tri, cd_loss, cd_gain, ddt_gain, ddt_loss])

# set output location to platform-stories/trial-runs/random_str within your bucket/prefix
random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
output_location = join('platform-stories/trial-runs', random_str)

# set which task outputs are to be stored and where
wf.savedata(water_pre.outputs.data, output_location + '/water_pre')
wf.savedata(water_post.outputs.data, output_location + '/water_post')
wf.savedata(exclusion_mask.outputs.data, output_location + '/exclusion_mask')
wf.savedata(cd_tri.outputs.data, output_location + '/cd_tristate')
wf.savedata(cd_gain.outputs.data, output_location + '/cd_bin_gain')
wf.savedata(cd_loss.outputs.data, output_location + '/cd_bin_loss')
wf.savedata(ddt_gain.outputs.data, output_location + '/ddt_gain')
wf.savedata(ddt_loss.outputs.data, output_location + '/ddt_loss')


print "Coastal Change Detection Workflow: \n\n"
print wf.status
print wf.id

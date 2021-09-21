import glob
import numpy as np
import os
import os.path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy import log
from PIL import Image
from astropy.visualization import AsymmetricPercentileInterval, simple_norm
import PIL.ImageOps 
import matplotlib.pyplot as plt
import csv
import json
from detectron2.structures import BoxMode
import random
random.seed(a=0)	#adding a seed here because I don't want the input data for the model to change every time I run this script
import shutil 

#defines an empty list for all dictionaries
listdict = []

#defines dimensions of the images
img_height = 4094
img_width = 2046

#defines current working directory
curdir = os.getcwd()

#gets a list of all regions files
regfiles = glob.glob('/mnt/c/Users/marzt/Documents/Research/MISHAPS_F1_*_r/MISHAPS_F1_*_r_*.reg')

#creates variable for the total number of objects in each classification to be used in the training set
num_ast = 0
num_var = 0
num_com = 0

#function that gets the images in the proper zscale
def _data_stretch(image, vmin=None, vmax=None, pmin=0.25, pmax=99.75,
                  stretch='linear', vmid=None, exponent=2):

	log.info("vmax = %10.3e" % vmax)

	if stretch == 'arcsinh':
		stretch = 'asinh'

	normalizer = simple_norm(image, stretch=stretch, power=exponent,
                             asinh_a=vmid, min_cut=vmin, max_cut=vmax, clip=False)

	data = normalizer(image, clip=True).filled(0)
	data = np.nan_to_num(data)
	data = np.clip(data * 255., 0., 255.)

	return data.astype(np.uint8)

for regfile in regfiles:
	print(regfile)

	#directory containing regions files, information files, and image files
	filedir = os.path.dirname(regfile)

	#splits name of region file into a list of strings
	regfile_name = os.path.basename(regfile)
	regfile_list = regfile_name.split('_')
	
	#defines chip, filter, and date from the name of the region file
	chip = regfile_list[2]
	filter = regfile_list[3]
	date = regfile_list[4].rstrip('.reg')

	#checks if image was already created; if True, skips process	
	filepath = curdir + '/data/' + chip + '_' + filter + '_' + date + '_RGB.png'
	if os.path.isfile(filepath) == False:
		#opens line 1 (command) of information file for reading
		info_file = open(filedir + '/' + chip + '_' + filter + '_' + date + '.txt', 'r')
		command = info_file.readline()
		
		#splits the command into a list of strings
		command_list = command.split(' ')
		
		#defines red, green, and blue images and creates an RGB image from them
		red_image = command_list[command_list.index('-red') + 1]
		green_image = command_list[command_list.index('-green') + 1]
		blue_image = command_list[command_list.index('-blue') + 1]
		
		#gets data from the red, green, and blue fits files
		#if os.path.isfile(red_image) == False:
		#	print('Error: fits file not found')
		#	break
		
		image_r = fits.getdata(filedir + '/' + red_image)
		image_g = fits.getdata(filedir + '/' + green_image)
		image_b = fits.getdata(filedir + '/' + blue_image)
	
		#creates red, green, and blue images
		zs = ZScaleInterval()
        
		log.info("Red:")
		vmin_r, vmax_r = zs.get_limits(image_r)
		image_r = Image.fromarray(_data_stretch(image_r, vmin=vmin_r, vmax=vmax_r, stretch='linear'))

		log.info("Green:")
		vmin_g, vmax_g = zs.get_limits(image_g)
		image_g = Image.fromarray(_data_stretch(image_g, vmin=vmin_g, vmax=vmax_g, stretch='linear'))

		log.info("Blue:")
		vmin_b, vmax_b = zs.get_limits(image_b)
		image_b = Image.fromarray(_data_stretch(image_b, vmin=vmin_b, vmax=vmax_b, stretch='linear'))

		#merges red, green, and blue images into rgb image
		img = Image.merge("RGB", (image_r, image_g, image_b)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
	
		#saves rgb image
		my_dpi = 120
		img_height = 4094
		img_width = 2046
		figsize = (img_width/my_dpi, img_height/my_dpi)
		fig = plt.figure(figsize = figsize, dpi = my_dpi)
		ax = plt.axes([0,0,1,1])
		plt.imshow(img, interpolation = 'nearest')
		plt.axis('off')
		#plt.show()
		plt.savefig(filepath, dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
	
		#inverts image colors
		image = Image.open(filepath)
		inverted_image = PIL.ImageOps.invert(image.convert('RGB'))
		#plt.imshow(inverted_image)
		#plt.show()
		inverted_image.save(filepath)
		
	imageitem = {"file_name" : filepath, "height" : img_height, "width" : img_width, "image_id" : os.path.basename(filepath)}
		
	#opens regions file for reading by lines
	file = open('/mnt/c/Users/marzt/Documents/Research/MISHAPS_F1_'+ chip +'_' + filter +'/MISHAPS_F1_' + chip + '_' + filter + '_' + date + '.reg', 'r')
	lines = file.readlines()[3:]
	
	#creates empty lists for notations for each class
	annotations0 = []
	annotations1 = []
	annotations2 = []
	
	#iterates over regions in each image
	for line in lines:
		#creates a list containting parameters of each region
		region_params = line[line.find("(")+1:line.find(")")]
		region_params = region_params.split(',')
		
		#excludes regions that are zero pixels in diameter
		if region_params[2] == '0':
			continue
			
		#finds the shape of the region
		list_shape = line.split('(')
		shape = list_shape[0]
			
		#finds the color of the region
		list_color = line.split('=')
		if len(list_color) == 2:
			color = list_color[1].rstrip('\n')
		elif len(list_color) == 1:
			color = 'green'
		
		#defines coordinates of subimage centroid
		x_coor = int(float(region_params[0]))
		y_coor = int(float(region_params[1])) 
			
		#defines dimensions of subimages
		if shape == 'circle':
			x_initial = float(region_params[0]) - float(region_params[2])
			x_final = float(region_params[0]) + float(region_params[2])
			y_initial = float(region_params[1]) - float(region_params[2])
			y_final = float(region_params[1]) + float(region_params[2])
		elif shape == 'ellipse':
			x_initial = float(region_params[0]) - float(region_params[2])
			x_final = float(region_params[0]) + float(region_params[2])
			y_initial = float(region_params[1]) - float(region_params[3])
			y_final = float(region_params[1]) + float(region_params[3])
			
			#defines angle of region in degrees
			angle = int(float(region_params[4]))
			#print(angle)
		
		#print(region_params)
		
		#converts dimensions into int data type	
		x_initial = int(x_initial)
		x_final = int(x_final)
		y_initial = img_height - int(y_initial)
		y_final = img_height - int(y_final)
		
		width = x_final - x_initial
		height = y_final - y_initial
		
		#dictionary for objects
		catdict={'white' : 2, 'red' : 2, 'yellow' : 1, 'blue' : 2, 'green' : 2, 'cyan' : 0, 'black' : 2}
		
		#appends region information to the corresponding class annotation list
		if catdict[color] == 0:
			annotations0.append({"bbox" : [x_initial, y_initial, width, height], "bbox_mode" : 1, "category_id" : catdict[color]})
		elif catdict[color] == 1:
			annotations1.append({"bbox" : [x_initial, y_initial, width, height], "bbox_mode" : 1, "category_id" : catdict[color]})
		elif catdict[color] == 2:
			annotations2.append({"bbox" : [x_initial, y_initial, width, height], "bbox_mode" : 1, "category_id" : catdict[color]})
	
	#shortens the lists to a set amount of regions to use in the final data file
	annotations1 = random.sample(annotations1, int(0.39*len(annotations1)))
	annotations2 = random.sample(annotations2, int(0.075*len(annotations2)))
	print(len(annotations0), len(annotations1), len(annotations2))
	
	#increases numbers of each category to find the total number
	num_ast = num_ast + len(annotations0)
	num_var = num_var + len(annotations1)
	num_com = num_com + len(annotations2)
	
	#combines annotations files into a larger annotations file
	annotations = annotations0 + annotations1 + annotations2
		
	imageitem["annotations"] = annotations	
	
	#defines list of dictionaries
	listdict.append(imageitem)
	
print('total number of asteroids: ', num_ast)
print('total number of variable stars: ', num_var)
print('total number of artifacts: ', num_com)

with open('train/json_data.json', mode='w') as outfile:
	json.dump(listdict, outfile, sort_keys= True, indent=4)
	
#adds images to the testing and training directories
test_dir = curdir + '/test'
train_dir = curdir + '/train'
source_dir = curdir + '/data'

move_to_train = glob.glob(source_dir + '/*.png')
move_to_test = random.sample(glob.glob(source_dir + '/*.png'), int(0.3*len(regfiles)))
print(len(regfiles))

for file in move_to_test:	#moves training images to train directory
	shutil.copy(file, test_dir)
	move_to_train.remove(file)
	
for file in move_to_train:
	shutil.copy(file, train_dir)

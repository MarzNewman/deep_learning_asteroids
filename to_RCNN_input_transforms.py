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
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
import sys

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

#####################################
############# FUNCTIONS #############
#####################################

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

#functions that transform the region annotations
def flipx(xi_og, xf_og, yi_og, yf_og):		#takes the original x and y coordinates of the bounding boxes as the parameters
	x_initial = img_width - xf_og
	x_final = img_width - xi_og
	y_initial = yi_og
	y_final = yf_og
	return [x_initial, x_final, y_initial, y_final]
	
def flipy(xi_og, xf_og, yi_og, yf_og):
	x_initial = xi_og
	x_final = xf_og
	y_initial = img_height - yf_og
	y_final = img_height - yi_og
	return [x_initial, x_final, y_initial, y_final]
	
#function that creates annotations
def create_annots(chip, date, transform):
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
			x_i = float(region_params[0]) - float(region_params[2])
			x_f = float(region_params[0]) + float(region_params[2])
			y_i = float(region_params[1]) - float(region_params[2])
			y_f = float(region_params[1]) + float(region_params[2])
		elif shape == 'ellipse':
			x_i = float(region_params[0]) - float(region_params[2])
			x_f = float(region_params[0]) + float(region_params[2])
			y_i = float(region_params[1]) - float(region_params[3])
			y_f = float(region_params[1]) + float(region_params[3])
			
			#defines angle of region in degrees
			angle = int(float(region_params[4]))
			#print(angle)
		
		#print(region_params)
		
		#converts dimensions into int data type	
		x_i = int(x_i)
		x_f = int(x_f)
		y_i = img_height - int(y_i)
		y_f = img_height - int(y_f)
		
		width = x_f - x_i
		height = y_f - y_i
		
		#dictionary for objects
		catdict={'white' : 2, 'red' : 2, 'yellow' : 1, 'blue' : 2, 'green' : 2, 'cyan' : 0, 'black' : 2}
		
		#appends region information to the corresponding class annotation list
		if transform == 'none':
			x_initial, x_final, y_initial, y_final = x_i, x_f, y_i, y_f
				
		elif transform == 'flipx':
			x_initial, x_final, y_initial, y_final = flipx(x_i, x_f, y_i, y_f)
		
		elif transform == 'flipy':
			x_initial, x_final, y_initial, y_final = flipy(x_i, x_f, y_i, y_f)
		
		elif transform == 'rotate':
			x_initial, x_final, y_initial, y_final = flipx(x_i, x_f, y_i, y_f)
			x_initial, x_final, y_initial, y_final = flipy(x_initial, x_final, y_initial, y_final)
			
		if catdict[color] == 0:
			annotations0.append({"bbox" : [x_initial, y_initial, width, height], "bbox_mode" : 1, "category_id" : catdict[color]})
		elif catdict[color] == 1:
			annotations1.append({"bbox" : [x_initial, y_initial, width, height], "bbox_mode" : 1, "category_id" : catdict[color]})
		elif catdict[color] == 2:
			annotations2.append({"bbox" : [x_initial, y_initial, width, height], "bbox_mode" : 1, "category_id" : catdict[color]})
		#print(annotations0)
		
	return [annotations0, annotations1, annotations2]

#function that saves processed rgb images into the data directory
def create_rgbimage(filepath, transform):	
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
		
		#gauss kernel for smoothing images
		fwhm = 4
		std = fwhm / (np.sqrt(8 * np.log(2)))		#std for gaussian kernel
		gauss2d = Gaussian2DKernel (x_stddev=std)
		gauss3d = np.stack([gauss2d, gauss2d, gauss2d])
		
		#smooths rgb images
		smooth_r = convolve(np.array(image_r), kernel = gauss2d)
		smooth_g = convolve(np.array(image_g), kernel = gauss2d)
		smooth_b = convolve(np.array(image_b), kernel = gauss2d)
		
		#creates red, green, and blue images
		zs = ZScaleInterval()
		
		log.info("Red:")
		vmin_r, vmax_r = zs.get_limits(image_r)
		#image_r = Image.fromarray(_data_stretch(image_r, vmin=vmin_r, vmax=vmax_r, stretch='linear'))
		smooth_r = Image.fromarray(_data_stretch(smooth_r, vmin=vmin_r, vmax=vmax_r, stretch='linear'))

		log.info("Green:")
		vmin_g, vmax_g = zs.get_limits(image_g)
		#image_g = Image.fromarray(_data_stretch(image_g, vmin=vmin_g, vmax=vmax_g, stretch='linear'))
		smooth_g = Image.fromarray(_data_stretch(smooth_g, vmin=vmin_g, vmax=vmax_g, stretch='linear'))

		log.info("Blue:")
		vmin_b, vmax_b = zs.get_limits(image_b)
		#image_b = Image.fromarray(_data_stretch(image_b, vmin=vmin_b, vmax=vmax_b, stretch='linear'))
		smooth_b = Image.fromarray(_data_stretch(smooth_b, vmin=vmin_b, vmax=vmax_b, stretch='linear'))
		#print ("shape ", smooth_b.size)

		#merges red, green, and blue images into rgb image
		#img = Image.merge("RGB", (image_r, image_g, image_b)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
		#img = Image.merge("RGB", (smooth_r, smooth_g, smooth_b)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
	
		#saves rgb image
		my_dpi = 120
		figsize = (img_width/my_dpi, img_height/my_dpi)
		#print("figsize ", figsize)
		fig = plt.figure(figsize = figsize, dpi = my_dpi)
				
		#ax = plt.axes([0,0,1,1])
		
		#merges red, green, and blue images into rgb image
		if transform == 'none':
			img = Image.merge("RGB", (smooth_r, smooth_g, smooth_b)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
			plt.imshow(img, interpolation = 'nearest')
			plt.axis('off')
			#plt.show()
			plt.savefig(filepath, dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
			
		elif transform == 'flipy':
			img = Image.merge("RGB", (smooth_r, smooth_g, smooth_b))
			plt.imshow(img, interpolation = 'nearest')
			plt.axis('off')
			#plt.show()
			plt.savefig(filepath, dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
			
		elif transform == 'flipx':
			img = Image.merge("RGB", (smooth_r, smooth_g, smooth_b)).transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT)
			plt.imshow(img, interpolation = 'nearest')
			plt.axis('off')
			#plt.show()
			plt.savefig(filepath, dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
			
		elif transform == 'rotate':
			img = Image.merge("RGB", (smooth_r, smooth_g, smooth_b)).transpose(PIL.Image.FLIP_LEFT_RIGHT)
			plt.imshow(img, interpolation = 'nearest')
			plt.axis('off')
			#plt.show()
			plt.savefig(filepath, dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
	
		print ("shape 1 ", img.size)
	
		#inverts image colors
		image = Image.open(filepath)
		inverted_image = PIL.ImageOps.invert(image.convert('RGB'))
		#plt.imshow(inverted_image)
		#plt.show()
		#inverted_image.save(filepath)
		
		#resizes images to (2046, 4094) aka the correct size
		resized_image = inverted_image.resize((2046, 4094))
		resized_image.save(filepath)
		
		print ("shape 2 ", resized_image.size)
		
		imageitem = {"file_name" : filepath, "height" : img_height, "width" : img_width, "image_id" : os.path.basename(filepath)}  ###### for running on *my* machine
		#imageitem = {"file_name" : '/project/marz746/label_asteroids/data/' + chip + '_' + filter + '_' + date + '_RGB.png', "height" : img_height, "width" : img_width, "image_id" : os.path.basename(filepath)}	######## for running on the LSU HPC
	return imageitem

#####################################
#####################################
#####################################

#iterates over regfiles for each image
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

	#defines filepaths for saving data images (original image, inverted in x and y, and rotated by 180 in x and y)
	filepath_og = curdir + '/data/' + chip + '_' + filter + '_' + date + '_RGB.png' 
	filepath_flipx = curdir + '/data/' + chip + '_' + filter + '_' + date + '_RGB_flipx.png' 
	filepath_flipy = curdir + '/data/' + chip + '_' + filter + '_' + date + '_RGB_flipy.png'
	filepath_rotate = curdir + '/data/' + chip + '_' + filter + '_' + date + '_RGB_rotate.png'
	filepaths = [filepath_og, filepath_flipx, filepath_flipy, filepath_rotate]
	transform_dict = {filepath_og : 'none', filepath_flipx : 'flipx', filepath_flipy : 'flipy', filepath_rotate : 'rotate'}
	
	#creates a list of image items and annotations for each transformed image
	imageitems = []
	trans_annots0 = []	#lists of all transformed annotations (0 is asteroids, 1 is variable stars, and 2 is combined)
	trans_annots1 = []
	trans_annots2 = []
	
	
	for filepath in filepaths:
		####### overrides all of the broken stuff that i don't feel like fixing
		if os.path.exists('train/json_data.json'):
			dictionary = json.load(open('train/json_data.json', 'r'))
			list_of_values = [value for elem in dictionary for value in elem.values()]
			value = os.path.basename(filepath)
			if value not in list_of_values:
				if os.path.isfile(filepath) == True:
					print ("Annotations not found in json file; you need to delete the image in the data folder because this part of the code doesn't work yet")
					sys.exit()
					
					
					
		print(os.path.basename(filepath))
		#saves rgb image to data directory only if image is NOT saved already
		if os.path.isfile(filepath) == False:	
			imageitems.append(create_rgbimage(filepath, transform_dict[filepath]))
			print("Image saved")
			
		#creates image annotations with bounding boxes only if annotations are NOT in the json file already
		if os.path.exists('train/json_data.json'):
			dictionary = json.load(open('train/json_data.json', 'r'))
			list_of_values = [value for elem in dictionary for value in elem.values()]
			value = os.path.basename(filepath)
			if value in list_of_values:
				append_annots = False
				print ("Annotations found in json file; new annotations will not be copied.")
			else:
				append_annots = True
				trans_annots0.append(create_annots(chip, date, transform_dict[filepath])[0])
				trans_annots1.append(create_annots(chip, date, transform_dict[filepath])[1])
				trans_annots2.append(create_annots(chip, date, transform_dict[filepath])[2])
				#print("AAAAAAAAAAAAAAAAAAAAAA")
				print(trans_annots0)
				print("Annotations not found in json file; new annotations will be copied")
			print("append_annots value = ", append_annots)#, type(append_annots))
						
		else:
			append_annots = True
			trans_annots0.append(create_annots(chip, date, transform_dict[filepath])[0])
			trans_annots1.append(create_annots(chip, date, transform_dict[filepath])[1])
			trans_annots2.append(create_annots(chip, date, transform_dict[filepath])[2])
			print("json file not found; a new json file will be created and annotations will be copied")
	
	#if annotations are not in json file, this section creates annotations variables with bounding boxes and appends them to image items
	if append_annots is True:
		print("Annotation loop entered")
		#shortens the lists to a set amount of regions to use in the final data file
		annotations0 = trans_annots0
		#print(annotations0)
		annotations1 = random.sample(trans_annots1, int(0.39*len(trans_annots1)))
		annotations2 = random.sample(trans_annots2, int(0.075*len(trans_annots2)))
		
		#decreases all annotations because there are too many
		#annotations0 = random.sample(annotations0, int(0.25*len(annotations0)))
		#annotations1 = random.sample(annotations1, int(0.25*len(annotations1)))
		#annotations2 = random.sccample(annotations2, int(0.25*len(annotations2)))
	
		print(len(annotations0), len(annotations1), len(annotations2))
		
		#increases numbers of each category to find the total number
		num_ast = num_ast + len(annotations0)
		num_var = num_var + len(annotations1)
		num_com = num_com + len(annotations2)
	
		#combines annotations files into a larger annotations file
		#annotations = annotations0 + annotations1 + annotations2
		
		#imageitem["annotations"] = annotations
		#imageitems["annotations"] = annotations0
		print(imageitems)	
		print (annotations0)
		#anns_per_image = len(annotations0)/len(imageitems)
		for i in range(0, len(imageitems)):
			#print(i)
			imageitems[i]["annotations"] = annotations0[i]
		#for image in imageitems:
		#	for annot in annotations0:
			
	
		#defines list of dictionaries
		#for item in imageitems:
		#	listdict.append(item)
		print("Annotations have been created")
	
		####################### testing my idea for ignoring images with annotations already ########################
		#dictionary = json.load(open('train/json_data.json', 'r'))
		#list_of_values = [value for elem in dictionary for value in elem.values()]
		#value = os.path.basename(filepath)
		#if value in list_of_values:
		#	print ("hell yeah")
		#else:
		#	print (":(")
		#############################################################################################################
	
		#writes or appends image items to the json file
		if os.path.exists('train/json_data.json'):
			#with open('train/json_data.json', mode='r') as outfile:
				#json.dump(imageitems, outfile, sort_keys= True, indent=4)
			#	outfile.write(imageitems)
			f = open('train/json_data.json', 'r')
			json_object = json.load(f)
			for item in imageitems:
				json_object.append(item)
			#json_update = json_object.update(imageitems)
			with open('train/json_data.json', mode='w') as outfile:
				json.dump(json_object, outfile, sort_keys= True, indent=4)
	
		else:		
			with open('train/json_data.json', mode='w') as outfile:
				json.dump(imageitems, outfile, sort_keys= True, indent=4)
		
		print("Annotations have been copied to json file")
		outfile.close()
	
print('total number of asteroids: ', num_ast)
print('total number of variable stars: ', num_var)
print('total number of artifacts: ', num_com)

#with open('train/json_data.json', mode='w') as outfile:
#	json.dump(listdict, outfile, sort_keys= True, indent=4)
	
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

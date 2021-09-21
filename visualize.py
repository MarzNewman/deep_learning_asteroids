#from detectron2.utils.visualizer import ColorMode
#from build_model.py import get_dicts

#dataset_dicts = get_dicts("balloon/test")
#for d in random.sample(dataset_dicts, 3):    
#    im = cv2.imread(d["file_name"])
#    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-#format
#    v = Visualizer(im[:, :, ::-1],
#                   metadata=balloon_metadata, 
#                   scale=0.5, 
#                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only #available for segmentation models
#    )
#    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#    cv2_imshow(out.get_image()[:, :, ::-1])


import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
random.seed(a=0)
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches

from detectron2.engine import DefaultTrainer

from detectron2.utils.visualizer import ColorMode
#from build_model.py import get_dicts

from detectron2.modeling import build_model

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils

import copy

#from build_model.py import predictor

#defines current working directory
curdir = os.getcwd()

#load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
#cfg.merge_from_file(curdir + '/output/model_final_MK2(3class).pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model
#cfg.MODEL.WEIGHTS = curdir + '/output/model_final.pth' # Set path model .pth
cfg.MODEL.WEIGHTS = curdir + '/output/model_final.pth' # Set path model .pth
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)
model = build_model(cfg)

#dataloader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True))

print (predictor)
#print(predictor["instances"].pred_classes)
#print(predictor["instances"].pred_boxes)

#prepare data in standard format
#register_coco_instances("my_dataset_train", {}, "data.csv", "./data")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

# read the csv file using read_csv function of pandas
train = pd.read_csv('data.csv')

#plots example of image with labels
fig = plt.figure()

#add axes to the image
ax = fig.add_axes([0,0,1,1])

# read and plot the image
#image = plt.imread('data/N22_r_190630_RGB.png')
#plt.imshow(image)

#iterates over image for different objects
for _,row in train[train.image_names == "N22_r_190630_RGB.png"].iterrows():
	xmin = row.xmin
	xmax = row.xmax
	ymin = row.ymin
	ymax = row.ymax
    
	width = xmax - xmin
	height = ymax - ymin
    
	# assign different color to different classes of objects
	if row.region_type == 'artifacts':
		edgecolor = 'black'
		ax.annotate('artifact', xy=(xmax-40,ymin+20))
	elif row.region_type == 'asteroids':
		edgecolor = 'cyan'
		ax.annotate('asteroid', xy=(xmax-40,ymin+20))
	elif row.region_type == 'variable_stars':
		edgecolor = 'yellow'
		ax.annotate('var_star', xy=(xmax-40,ymin+20))
	elif row.region_type == 'green_cosmic_rays':
		edgecolor = 'limegreen'
		ax.annotate('ray', xy=(xmax-40,ymin+20))
	elif row.region_type == 'red_cosmic_rays':
		edgecolor = 'red'
		ax.annotate('ray', xy=(xmax-40,ymin+20))
	elif row.region_type == 'blue_cosmic_rays':
		edgecolor = 'blue'
		ax.annotate('ray', xy=(xmax-40,ymin+20))
	elif row.region_type == 'object_of_interest':
		edgecolor = 'magenta'
		ax.annotate('unknown', xy=(xmax-40,ymin+20))
		
	# add bounding boxes to the image
	rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
	
	ax.add_patch(rect)
#plt.show()
plt.close()

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # See "Data Augmentation" tutorial for details usage
    auginput = T.AugInput(image)
    transform = T.Resize((800, 800))(auginput)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
       "image": image,
       "instances": utils.annotations_to_instances(annos, image.shape[1:])
    }

#removes previously registered dataset
#DatasetCatalog.remove(my_dataset_train)
#DatasetCatalog.clear()

def get_dicts(img_dir):
	#json file with data for each image
	json_file = os.path.join(img_dir, "json_data.json")
	
	#opens json file for reading
	with open(json_file) as f:
		imgs_anns = json.load(f)
	
	dataset_dicts = imgs_anns
	
	#dataset_dicts = []
	
	#creates an annotation for each labeled object in each image in the data taken from the json file
##	for idx, v in enumerate(imgs_anns):
##		record = {}

##		filename = os.path.join(img_dir, v["file_name"])
##		height, width = cv2.imread(filename).shape[:2]
        
        	#defines variables from json file
##		record["file_name"] = filename
##		record["image_id"] = idx
##		record["height"] = height
##		record["width"] = width
      
##		annos = v["annotations"]
##		objs = []
#		for _, anno in annos:
#			assert not anno["region_attributes"]
#			anno = anno["shape_attributes"]
#			px = anno["all_points_x"]
#			py = anno["all_points_y"]
#			poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#			poly = [p for x in poly for p in x]

#			obj = {
#				"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#				"bbox_mode": BoxMode.XYWH_ABS,
#				"segmentation": [poly],
#				"category_id": 0,
#			}
#		objs.append(obj)
#		record["annotations"] = objs
#		dataset_dicts.append(record)
	return dataset_dicts

for d in ["train", "test"]:
	print(type(predictor))
	#print(predictor(d))
	#print(predictor["instances"].pred_classes)
	DatasetCatalog.register("astdataset_" + d, lambda d=d: get_dicts( d))
#	MetadataCatalog.get("astdataset_" + d).set(thing_classes=['artifacts','red_cosmic_rays', 'variable_stars', 'blue_cosmic_rays', 'green_cosmic_rays', 'asteroids', 'object_of_interest'])
	MetadataCatalog.get("astdataset_" + d).set(thing_classes=['asteroids', 'variable_stars', 'combined'])
	#MetadataCatalog.get("astdataset_" + d)
	
ast_metadata = MetadataCatalog.get("astdataset_train")
print(ast_metadata)

cfg.DATASETS.TRAIN="astdataset_train"

#prints randomly selected images with annotations
dataset_dicts = get_dicts(curdir + "/train")

#print(dataset_dicts)

#shows a sample of labeled images
for d in random.sample(dataset_dicts, 1):
	#print(d)
	img = cv2.imread(d["file_name"])
	print(img.shape)
	visualizer = Visualizer(img[:, :, ::-1], metadata=ast_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
	out = visualizer.draw_dataset_dict(d)
	#cv2.imshow('', out.get_image()[:, :, ::-1])
	#cv2.waitKey(0)
	plt.imshow(out.get_image()[:, :, ::-1])
	plt.show()
plt.close()

dataloader_test = build_detection_train_loader("astdataset_test", mapper=custom_mapper, total_batch_size=76)

#visualize data with bounding boxes
#dataset_dicts = get_dicts(curdir + "/test")
#outputs = model(dataset_dicts)
#print(outputs["instances"].pred_classes)

#visualizes a random image with the labeled objects
for d in random.sample(dataset_dicts, 1):    
	im = cv2.imread(d["file_name"])
	print(d["file_name"])
#	outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-#format
#	print ('\n')
#	print(outputs) 
#	print('\n')
#	v = Visualizer(im[:, :, ::-1], metadata=ast_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
#	out = bbbv.draw_instance_predictions(outts["instances"].to("cpu"))
	#outputs = model(ast_metadata)
	outts = predictor(im[..., ::-1])
	print("instances: ", outts["instances"].pred_classes)
	#outputs = model(dataloader_test)
	#print(outputs["instances"].pred_classes)
	modelclasses = MetadataCatalog.get("astdataset_train").thing_classes
	print(modelclasses)
	df = pd.DataFrame(modelclasses,columns=['Model classes'])
	print(df)
	v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("astdataset_train"), scale=0.5, instance_mode=ColorMode.IMAGE_BW)
	out = v.draw_instance_predictions(outts["instances"].to("cpu"))
	#print(df)
	#cv2.imshow('2', out.get_image()[:, :, ::-1])
	#cv2.waitKey(10000)	
	#plt.imshow(out.get_image()[:, :, ::-1])
	plt.imshow(out.get_image()[..., ::-1])
	plt.show()
  
#Evaluate  
#evaluator = COCOEvaluator("astdataset_val", ("bbox", "segm"), False, output_dir="./output/")
#val_loader = build_detection_test_loader(cfg, "astdataset_val")
#print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`

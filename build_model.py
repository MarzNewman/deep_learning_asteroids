import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
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

#defines current working directory
curdir = os.getcwd()

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

#removes previously registered dataset
#DatasetCatalog.remove(my_dataset_train)
#DatasetCatalog.clear()

def get_dicts(img_dir):
	json_file = os.path.join(img_dir, "json_data.json")
	with open(json_file) as f:
		imgs_anns = json.load(f)
	
	dataset_dicts = imgs_anns
	
	#dataset_dicts = []
##	for idx, v in enumerate(imgs_anns):
##		record = {}

##		filename = os.path.join(img_dir, v["file_name"])
##		height, width = cv2.imread(filename).shape[:2]
        
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
	DatasetCatalog.register("astdataset_" + d, lambda d=d: get_dicts( d))
	#MetadataCatalog.get("astdataset_" + d).set(thing_classes=['artifacts','red_cosmic_rays', 'variable_stars', 'blue_cosmic_rays', 'green_cosmic_rays', 'asteroids', 'object_of_interest'])
	MetadataCatalog.get("astdataset_" + d).set(thing_classes=['asteroids', 'variable_stars', 'combined'])
ast_metadata = MetadataCatalog.get("astdataset_train")

#prints randomly selected images with annotations
dataset_dicts = get_dicts(curdir + "/train")

#print(dataset_dicts)

#shows a sample of labeled images
#for d in random.sample(dataset_dicts, 3):
#	img = cv2.imread(d["file_name"])
#	visualizer = Visualizer(img[:, :, ::-1], metadata=ast_metadata, scale=0.5)
#	out = visualizer.draw_dataset_dict(d)
#	plt.imshow(out.get_image()[:, :, ::-1])
#	plt.show()


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("astdataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (balloon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.DEVICE = 'cpu'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# Look at training curves in tensorboard:
#load_ext tensorboard
#tensorboard --logdir output

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

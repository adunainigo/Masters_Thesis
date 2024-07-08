#imports
import os
import sys

# Directorio que quieres añadir
new_path = "C:\\Users\\aduna\\Documents\\Master_KU_Leuven\\Master_Thesis\\program\\data\\Github Matthias\\robot_demonstrator\\src"

# Añadir el directorio al PYTHONPATH
if new_path not in sys.path:
    sys.path.append(new_path)
import _thread
import albumentations as A
import cv2
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as TF
from albumentations.pytorch import ToTensorV2
from PIL import Image
from robot_demonstrator.ABB_IRB1200 import ABB_IRB1200
from robot_demonstrator.Camera import *
from robot_demonstrator.image_processing import *
from robot_demonstrator.plot import *
from robot_demonstrator.transformations import *
from skimage import feature, measure, morphology
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm
import joblib
import warnings


# Suppress all warnings
warnings.filterwarnings('ignore')

# Create camera object
cam = Camera()

# Start camera
cam.start()

# Create camera object (online)
robot = ABB_IRB1200("192.168.125.1")

# Start robot
robot.start()

# Load T_bc (Transformation matrix from robot base frame to camera frame)
T_bc = np.load('./data/T_bc.npy')

# Load perspective matrix (calculated using the image_rectification_test.py file)
M = np.load('./data/perspective_transform.npy')

# Load error model
model = pickle.load(open('./data/error_model.sav', 'rb'))

# Define pick and place orientation
quat = list(quat_from_r(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))) # Quaternion of the pick and place orientation
quat = [quat[3], quat[0], quat[1], quat[2]] # Convert [x y z w] to [w x y z]

# Define fixed z-position to pick
grip_height = 7

# Define place position
pose_place = [450.0, 290.0, 220] 

# Define offsets
offset1 = np.array([0, 0, 40]) # Offset above the pick and place poses
tool_offset = np.array([-math.sqrt(200), -math.sqrt(200), 170]) # Tool offset (Translation from robot end effector to TCP)

# Define error
error = [0, 0, 0] # [7, 3, 0]  Error in system obtained from data collection --> to be reduced

# Robot boundaries
xmin = 350 # Minimal Cartesian x-position
xmax = 630 # Maximal Cartesian x-position
ymin = -250 # Minimal Cartesian y-position
ymax = 250 # Maximal Cartesian y-position

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Global params
global xyz_base
xyz_base = []

# Robot task thread function
def robot_task(name):

	# Define global variable
	global xyz_base

	# Loop
	while True:

		# Sleep
		time.sleep(1)

		# Get feasible positions
		xyz_base_feasible = [xyz for xyz in xyz_base if not (xyz[0] > xmax or xyz[0] < xmin or xyz[1] > ymax or xyz[1] < ymin)]

		# Check if there are feasible positions
		if len(xyz_base_feasible) < 1: continue

		# Get first element
		xyz_base_tmp = xyz_base_feasible[0]

		# Debug
		print("\nRobot - Start picking object!\n")

		# Set pick pose upper
		robot.con.set_cartesian([xyz_base_tmp + offset1, quat])
		time.sleep(1)

		# Set pick pose
		robot.con.set_cartesian([xyz_base_tmp, quat])
		time.sleep(1)

		# Set pick DIO
		robot.con.set_dio(1)
		time.sleep(1)

		# Set pick pose upper
		robot.con.set_cartesian([xyz_base_tmp + offset1, quat])
		time.sleep(1)

		# Set place pose upper
		robot.con.set_cartesian([pose_place + offset1, quat])
		time.sleep(1)

		# Set place pose
		robot.con.set_cartesian([pose_place, quat])
		time.sleep(1)

		# Set place DIO
		robot.con.set_dio(0)
		time.sleep(1)

		# Set home position
		robot.con.set_joints([0, 0, 0, 0, 0, 0])
		time.sleep(1)

		# Debug
		print("\nRobot - Finished picking object!\n")

def load_checkpoint(checkpoint, model):
	"""
	Loads the model state from a checkpoint file.

	Parameters:
		checkpoint (dict): The checkpoint containing model state as saved previously.
		model (torch.nn.Module): The model instance where the state will be loaded.
	"""
	print("=> Loading checkpoint")
	model.load_state_dict(checkpoint["state_dict"])
	
# Postprocessing

def get_pieces_features(mask):
	"""
	Given a mask path, returns a dictionary with the features of each piece in 
	the image including the radius of the circumcircle.
	"""
	# Read the image mask
	mask = mask.numpy().squeeze()
	mask = (mask*255).astype(np.uint8)
	# Apply Gaussian blur to reduce noise while preserving edges
	#blur = cv2.GaussianBlur(mask, (5, 5), 0)
	
	# Threshold the image to ensure only the pieces are in white
	_, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
	
	# Find all contours on the thresholded image
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Filter out very small contours that are likely noise
	contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
	
	# Initialize list to hold features of each piece
	pieces_features = []
	
	# Process each contour to extract features
	for piece_contour in contours:
		# Create a mask of the piece
		piece_mask = np.zeros_like(mask)
		cv2.drawContours(piece_mask, [piece_contour], -1, (255), thickness=cv2.FILLED)
		

		# Eccentricity (fitEllipse) if the contour has enough points
		if piece_contour.shape[0] >= 5:
			(x, y), (MA, ma), angle = cv2.fitEllipse(piece_contour)
			eccentricity = np.sqrt(1 - (MA / ma) ** 2)
		else:
			eccentricity = None

		# Centroid and orientation (moments)
		M = cv2.moments(piece_contour)
		cx = int(M['m10'] / M['m00'])
		cy = int(M['m01'] / M['m00'])
		centroid = (cx, cy)

		# Circumcircle (minEnclosingCircle)
		(x, y), radius = cv2.minEnclosingCircle(piece_contour)

		# Compile features into a dictionary
		features = {
			#'aspect_ratio': aspect_ratio,
			'eccentricity': eccentricity,
			#'area': area,
			#'perimeter': perimeter,
			'centroid': centroid,
			#'orientation': orientation,
			#'convexity': convexity,
			'radius': radius
		}
		# Add features of the current piece to the list
		pieces_features.append(features)

	return pieces_features

def save_annotated_image(image, save_path, pieces_features):

	"""
	Saves the image to the given path, annotated with the circumcircle and centroid mark for each piece in red for visibility.
	Converts grayscale images to RGB before annotation.
	Args:
	- image (numpy array): The image array.
	- save_path (str): Path to save the annotated image.
	- pieces_features (list): List of dictionaries containing features of each piece including the radius and centroid.
	"""
	image = image.numpy().transpose(1, 2, 0)
	image = (image * 255).astype(np.uint8)
	# Convert grayscale image to RGB
	if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	# Annotate each piece
	for features in pieces_features:
		# Draw the circumcircle in red
		cv2.circle(image, (int(features['centroid'][0]), int(features['centroid'][1])), int(features['radius']), (0, 0, 255), 2)
		# Draw the centroid as a red 'X'
		cv2.drawMarker(image, (int(features['centroid'][0]), int(features['centroid'][1])), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
	# Save the annotated image
	cv2.imwrite(save_path, image)

def stats(data, savedir):
	# Extraer todas las keys posibles (suponiendo que todas las features están en todos los diccionarios)
	keys = list(data[0][0].keys())
	keys.remove('centroid')
	#keys.remove('aspect_ratio')
	#keys.remove ('perimeter')
	# Crear un diccionario para acumular los valores de cada feature
	features = {key: [] for key in keys}
	
	# Recorrer cada lista de diccionarios y cada diccionario para acumular los valores de las features
	for sublist in data:
		for dic in sublist:
			for key, value in dic.items():
				# Omitir centroides porque son tuplas y no es trivial calcular media y desviación estándar de tuplas
				#if key != 'centroid'and key!= 'area'and key!= 'perimeter':
				if key != 'centroid' :
					features[key].append(value)
	
	# Calcular la media y la desviación estándar para cada feature
	stats_dict = {key: (np.mean(values), np.std(values)) for key, values in features.items()}
	
	# Crear el directorio si no existe
	if not os.path.exists(savedir):
		os.makedirs(savedir)
	
	# Guardar el diccionario en un archivo JSON
	with open(os.path.join(savedir, 'feature_stats.json'), 'w') as f:
		json.dump(stats_dict, f, indent=4)
	
	return stats_dict

def load_stats(filepath):
	"""
	Load the statistics from a JSON file.
	"""
	with open(filepath, 'r') as file:
		stats = json.load(file)
	return stats

def filter_pieces(stats_data, pieces_list, std=9):
	"""
	Filter pieces based on the statistical data provided.
	Args:
	stats_data (dict): A dictionary with keys as properties and values as (mean, sigma).
	pieces_list (list): A list of dictionaries, where each dictionary contains properties of a piece.
	Returns:
	list: A list of dictionaries, each containing 'centroid' and 'radius' of valid pieces.
	"""
	valid_pieces = []
	stats_data= load_stats(stats_data)
	# Iterate over each piece
	for piece in pieces_list:
		valid = True
		# Check each statistical property
		for key, (mean, sigma) in stats_data.items():
			if key in piece:  # Only check if the key exists in the piece's data
				value = piece[key]
				if not (mean - std * sigma <= value <= mean + std * sigma):#+-7\sigma
					#print(f"Not valid due to {key}")
					valid = False
					break
		if valid:
			# If all properties are valid, add the centroid and radius to the valid list
			valid_pieces.append({'centroid': piece['centroid'], 'radius': piece['radius']})
	return valid_pieces

def postprocess(tensor_prediction, area_threshold):
	# Convert tensor to numpy array
	image = tensor_prediction.squeeze().cpu().numpy()
	
	# Ensure the image is in 8-bit format
	if image.dtype != np.uint8:
		image = np.clip(image * 255, 0, 255).astype(np.uint8)

	# Handle color conversion if necessary
	if len(image.shape) == 2:  # It's a grayscale image
		image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	else:
		image_color = image  # It's already a BGR image

	# Define the kernel for morphological operations
	kernel = np.ones((5, 5), np.uint8)

	# Apply morphological opening and closing
	opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

	# Convert to binary image for contour detection
	_, binary = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	# Find contours
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	mask = np.zeros_like(image)

	# Filter and draw contours by area
	for contour in contours:
		if cv2.contourArea(contour) > area_threshold:
			cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

	# Further morphological cleaning
	sure_bg = cv2.dilate(mask, kernel, iterations=3)

	# Distance transformation for segmentation
	dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
	_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
	sure_fg = np.uint8(sure_fg)

	# Unknown region
	unknown = cv2.subtract(sure_bg, sure_fg)

	# Connected components to separate different objects
	_, markers = cv2.connectedComponents(sure_fg)
	markers = markers + 1
	markers[unknown == 255] = 0

	# Watershed algorithm to segment connected parts
	cv2.watershed(image_color, markers)
	image_color[markers == -1] = [255, 0, 0]  # Mark boundaries in red

	# Prepare final mask in the same format as input
	final_mask = np.zeros_like(image, dtype=np.uint8)
	final_mask[markers > 1] = 1
	
	# Convert final mask back to tensor format
	final_tensor = torch.from_numpy(final_mask).unsqueeze(0)  # Add batch dimension if necessary

	return final_tensor

def read_labels(labels_path):
	with open(labels_path, 'r') as file:
		line = file.readline().strip()
		values = np.array(list(map(int, line.split())))
	return values

def calculate_corrections(differences):
	total_diff_x = sum(diff[0] for diff in differences)
	total_diff_y = sum(diff[1] for diff in differences)
	count = len(differences)
	
	C_x = total_diff_x / count
	C_y = total_diff_y / count
	
	return C_x, C_y

class DoubleConv(nn.Module):
	"""
	A module to perform two consecutive convolution operations followed by batch normalization and ReLU activation.

	Attributes:
		conv (nn.Sequential): A sequential container of two convolutional blocks.

	Parameters:
		in_channels (int): Number of input channels.
		out_channels (int): Number of output channels.
	"""
	def __init__(self, in_channels, out_channels):
		super(DoubleConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		"""
		Defines the computation performed at every call of the DoubleConv module.

		Parameters:
			x (torch.Tensor): The input data.

		Returns:
			torch.Tensor: The output data after passing through the convolution blocks.
		"""
		return self.conv(x)
class UNET(nn.Module):
	"""
	U-Net architecture for image segmentation tasks.

	Attributes:
		ups (nn.ModuleList): List of modules used in the decoder path of U-Net.
		downs (nn.ModuleList): List of modules used in the encoder path of U-Net.
		pool (nn.MaxPool2d): Max pooling layer.
		bottleneck (DoubleConv): The bottleneck layer of U-Net.
		final_conv (nn.Conv2d): Final convolutional layer to produce the output segmentation map.

	Parameters:
		in_channels (int): Number of channels in the input image.
		out_channels (int): Number of channels in the output image.
		features (List[int]): Number of features in each layer of the network.
	"""
	def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
		super(UNET, self).__init__()
		self.ups = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		for feature in features:
			self.downs.append(DoubleConv(in_channels, feature))
			in_channels = feature

		for feature in reversed(features):
			self.ups.append(
				nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
			)
			self.ups.append(DoubleConv(feature*2, feature))

		self.bottleneck = DoubleConv(features[-1], features[-1]*2)
		self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

	def forward(self, x):
		"""
		Defines the forward pass of the U-Net using skip connections and up-sampling.

		Parameters:
			x (torch.Tensor): The input tensor for the U-Net model.

		Returns:
			torch.Tensor: The output tensor after processing through U-Net.
		"""
		skip_connections = []

		for down in self.downs:
			x = down(x)
			skip_connections.append(x)
			x = self.pool(x)

		x = self.bottleneck(x)
		skip_connections = skip_connections[::-1]

		for idx in range(0, len(self.ups), 2):
			x = self.ups[idx](x)
			skip_connection = skip_connections[idx//2]

			if x.shape != skip_connection.shape:
				x = TF.resize(x, size=skip_connection.shape[2:])

			concat_skip = torch.cat((skip_connection, x), dim=1)
			x = self.ups[idx+1](concat_skip)

		return self.final_conv(x)
	
def configuration_models(): 
	 # Define the number of input features
	NINPUT = 3
	# Define the number of output features
	NOUTPUT = 3
	# Define the number of neurons in the hidden layers
	NHIDDEN = 25
	# Set the activation function to Tanh
	ACTIVATION = nn.Tanh()

	# Model Initialization
	CHECKPOINT_PATH_UNET= "./inigoaduna/my_checkpoint.pth.tar"  # Path to the model checkpoint
	CHECKPOINT_PATH_MLP = "./inigoaduna/mlp_checkpoint.pth.tar"
	SCALER_X = "./inigoaduna/scaler_X.pkl"
	SCALER_Y = "./inigoaduna/scaler_y.pkl"
	STATSDIR = "./inigoaduna/feature_stats.json"
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Set device to CUDA if available, otherwise use CPU.    
	print(f'Device= {DEVICE}')
 
	model_mlp = nn.Sequential(
		nn.Linear(NINPUT, NHIDDEN),
		ACTIVATION,
		nn.Linear(NHIDDEN, NHIDDEN),
		ACTIVATION,
		nn.Linear(NHIDDEN, NOUTPUT)
	)
	model_mlp.load_state_dict(torch.load(CHECKPOINT_PATH_MLP))


	# Model initialization with specified input and output channels
	model_unet = UNET(in_channels=3, out_channels=1).to(DEVICE)
	load_checkpoint(torch.load(CHECKPOINT_PATH_UNET ), model_unet)
	model_unet.eval()  # Set the model to evaluation mode.       
	# Define a function for the thread
	scaler_X_loaded = joblib.load(SCALER_X)
	scaler_y_loaded = joblib.load(SCALER_Y)
	return scaler_X_loaded, scaler_y_loaded, model_unet, model_mlp, DEVICE,STATSDIR

 
def camera_task(name):

	IMAGE_HEIGHT = 270  
	IMAGE_WIDTH  = 480
	AREA_TRESHOLD = 9000
	NON_DETECTED_PIECES=0
	SIGMA = 3
	post_predict_resize = A.Resize(height=1080, width=1920, interpolation=1)  
	test_transforms = A.Compose(
	[
		A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),  # Resize images to defined dimensions.
		A.Normalize(
			mean=[0.0, 0.0, 0.0],  # Normalize images with a mean of 0.
			std=[1.0, 1.0, 1.0],    # Standard deviation for normalization.
			max_pixel_value=255.0,  # Maximum pixel value in input images.
		),
		ToTensorV2(),  # Convert images to tensor format compatible with PyTorch.
	],
	)

	scaler_x, scaler_y, model_unet, model_mlp, DEVICE, STATSDIR = configuration_models()
	
	# Global
	global xyz_base

	# Loop
	while True:

		# Read frame
		image, depth_image = cam.read()
  
		#image = cv2.imread(image)
		#depth = np.load(depth_image)

		# Undistort image
		h, w = image.shape[:2]
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam.mtx, cam.dist, (w,h), 1, (w,h))
		mapx, mapy = cv2.initUndistortRectifyMap(cam.mtx, cam.dist, None, newcameramtx, (w, h), 5)
		#image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

		# Warp image as if the camera took the image from above, perpendicular to the table
		warped_image = cv2.warpPerspective(image, M, (np.shape(image)[1], np.shape(image)[0]))
		image= Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
		transformed = test_transforms(image=np.array(image))
		image = transformed["image"].unsqueeze(0).to(DEVICE)
		with torch.no_grad():
			prediction = model_unet(image)
		prediction = torch.sigmoid(prediction)
		prediction = (prediction > 0.5).float()

    	# Resize prediction and save
		prediction = prediction.squeeze().cpu().numpy()
		resized_prediction = post_predict_resize(image=prediction)['image']
		tensor_prediction = torch.from_numpy(resized_prediction).unsqueeze(0)
	
		#Postprocessing
		tensor_prediction = postprocess(tensor_prediction, AREA_TRESHOLD)    
    
    	#Get Pieces features
		pieces_features = get_pieces_features(tensor_prediction)
		#Filter objects not similar to images
		pieces_features = filter_pieces(STATSDIR , pieces_features, SIGMA)
		robot_locations = []
		if len(pieces_features)!=0:
			for piece in pieces_features: 
            	#Get Data
				center = piece['centroid']
				radius = piece['radius']
            	# Transform pixel on warped image back to original image
				new_pixel = np.dot(np.linalg.inv(M), np.array([[center[0]], [center[1]], [1]]))
				center = [int(new_pixel[0][0]/new_pixel[2][0]), int(new_pixel[1][0]/new_pixel[2][0])]
            
            	## Get pixel depth 
				pixel_depth = depth_image[center[1], center[0]]
            
				x_wc = center[0]
				y_wc = center[1]
				z_wc = pixel_depth 
            	# Prepare the model for mlp
				input_mlp_raw = np.array([[x_wc, y_wc, z_wc]])  # Convert it to numpy and add a dimension
				input_mlp_scaled = scaler_x.transform(input_mlp_raw)  # Apply The scaling
				input_mlp = torch.tensor(input_mlp_scaled, dtype=torch.float32)  # Convert to tensor
				with torch.no_grad():
					output_mlp = model_mlp(input_mlp)
				x_pred, y_pred, z_pred = scaler_y.inverse_transform(output_mlp.numpy().reshape(1, -1)).squeeze()

				xyz= np.array ([x_pred, y_pred, 177])
				robot_locations.append(list(xyz))
				xyz_base = robot_locations
				print(f'Robot Frame: ({x_pred},{y_pred},{z_pred})')
    
				### The code below is to show the results on the screen


				# Display the raw image with bounding box
				final_image = cv2.resize(warped_image, (int(1920/2), int(1080/2)))  
				cv2.imshow('frame1', final_image)
				cv2.resizeWindow("frame1", (int(1920/2), int(1080/2)))  
				cv2.moveWindow("frame1", 0, 0)
				if cv2.waitKey(10) & 0xFF == ord('q'):
					break 


				# Display the mask
				final_image = cv2.resize(resized_prediction, (int(1920/2), int(1080/2)))  
				cv2.imshow('frame3', final_image)
				cv2.resizeWindow("frame3", (int(1920/2), int(1080/2)))  
				cv2.moveWindow("frame3", int(1920/2), 0)
				if cv2.waitKey(10) & 0xFF == ord('q'):
					break 

				# Display the ldepth color map
				final_image = cv2.resize(cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET), (int(1920/2), int(1080/2)))  
				cv2.imshow('frame4', final_image)
				cv2.resizeWindow("frame4", (int(1920/2), int(1080/2)))  
				cv2.moveWindow("frame4", int(1920/2), int(1080/2))
				if cv2.waitKey(10) & 0xFF == ord('q'):
					break 
				
if __name__ == "__main__":

	# Create two threads
	try:
		_thread.start_new_thread(camera_task, ("Thread-1", ) )
		_thread.start_new_thread(robot_task, ("Thread-2", ) )
	except:
		print ("Error: unable to start thread")

	# Loop threads
	while True:
		pass
import numpy as np
import cv2
import os
import argparse

# Model: https://github.com/richzhang/colorization/tree/caffe/colorization/models
# Numpy File: https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy

# Defining file paths

prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"

# Defining test images path

image_path = ""
tweak_value = ""

def parse_arguments():
	parser = argparse.ArgumentParser(description='Process command line arguments.')
	parser.add_argument('-image')
	parser.add_argument('-v', '--l_val', nargs = '*', help='lightness value', type=Lightness_value)
	parser.add_argument('-i', '--image', nargs = '*', help='image path', type=Image_path)

	return parser.parse_args()


def Lightness_value(value):
	global tweak_value
	if not value:
		if int(value):
			return
	try:
		tweak_value = int(value)
		return "value "+str(value)
	except ValueError:
		raise argparse.ArgumentTypeError("Given Lightness is not valid!")


def Image_path(img):
	global image_path
	if not img:
		return
	try:
		image_path = img
		return "img "+str(img)
	except ValueError:
		raise argparse.ArgumentTypeError("Given Lightness is not valid!")


parsed_args = parse_arguments()

def main():
	parsed_args = parse_arguments()
	print(image_path)
	print(tweak_value)

# image_path = "test_img/lion.jpg"

# net (variable): stores the neural network from caffemodel
# points (variable/numpy_obj): stores cluster center points called kernels

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

# transpose cnn kernel into size - 1x1

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

"""
class8_ab is the ID for color scheme. This is very different from RGB. Here we are using LAB and loading channel quantization for a & b.

LAB -> Lightness a* b*
"""

# Loading Image -> Normalizing it -> Changing cholor scheme from BGR to LAB

print(image_path) # For testing purpose

bw_image = cv2.imread(image_path)
normalized = bw_image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB) # BGR -> LAB: Because imread reads image in BGR format

# Restricted dimensions for model: 224px x 224px

resized = cv2.resize(lab, (224, 224))

# Splitting channels and storing them
# L: takes the "Lightness" value after splitting
L = cv2.split(resized)[0]
L -= tweak_value # Tip: Tweak this to get better results

# Passing "Lightiness" value to neural network which will predict coloured values of a* & b* channel of the image.

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resizing Coloured image back to it's original shape
ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
L = cv2.split(lab)[0] # getting back the original "Lightness" level

# Joining channels of image together
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Changing channels from LAB to RGB
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = (255.0 * colorized).astype("uint8") # Scaling image to it's original dimensions

# cv2.imshow("Before", bw_image)
# cv2.imshow("After", colorized)

img_name = image_path.split("/")[-1].split(".")[0] + "_L" + str(tweak_value) + "_colorized." + image_path.split(".")[-1]
cv2.imwrite(img_name, colorized)

cv2.waitKey(0)
cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
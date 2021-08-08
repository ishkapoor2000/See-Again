import os
import argparse

L_val = "nope!L"
imgs = "nope!img"

def parse_arguments():
	parser = argparse.ArgumentParser(description='Process command line arguments.')
	parser.add_argument('-image')
	parser.add_argument('-v', '--l_val', nargs = '*', help='lightness value', type=Lightness_value)
	parser.add_argument('-i', '--image', nargs = '*', help='image path', type=Image_path)

	return parser.parse_args()


def Lightness_value(value):
	global L_val
	if not value and int(value):
		return
	try:
		L_val = value
		return "value "+str(value)
	except ValueError:
		raise argparse.ArgumentTypeError("Given Lightness is not valid!")


def Image_path(img):
	global imgs
	if not img:
		return
	try:
		imgs = img
		return "img "+str(img)
	except ValueError:
		raise argparse.ArgumentTypeError("Given Lightness is not valid!")


def main():
	parsed_args = parse_arguments()
	print(imgs)
	print(L_val)

if __name__ == "__main__":
	main()
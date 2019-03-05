#	============================================================
#	Purpose: To mark a specific area (i.e., ROI: Region Of 
#	Interest) of a retinal image and save co-ordinates of two 
#	points of the selcted rectangle (left-top, right-bottom).
#	------------------------------------------------------------
#	Took Help From:
#		https://matplotlib.org/gallery/widgets/rectangle_selector.html#rectangle-selector
#	------------------------------------------------------------
#	Tasks:
#		1.	Make a list of all files in DB_DIR directory.
#		2.	Load images from the list one-by-one when 'next'
#			button is clicked.
#		3.	Save image file name and co-ordinates of marked of
#			ROI area in a file in DIR directory.
#
#	Information:
#		1.	Task of marking ROI area by this program cannot be
#			paused. Each time it displays the first image
#			of the image list.
#
#			It is better to create directories of reasonable
#			amount of images. Or, complete marking task for
#			all images. 
#
#		 	Maybe in the next version this problem will be
#			solved.:)))))
#	------------------------------------------------------------
#	Sangeeta Biswas
#	Post-Doc Researcher
#	Brno University of Technology, Czech Republic
#	26.2.2019
#	============================================================

'''	Import necessary modules.	'''
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
from tensorflow.keras.preprocessing.image import load_img
from matplotlib.widgets import RectangleSelector
import numpy as np
import datetime

'''	Declare constants.	'''
DIR = '/home/linux-mint/Desktop/TFRecord_Dataset/' 
DB_DIR = DIR + "Images_Val/" #'/media/sangeeta/Data/RetinaDatabase/kaggle/Kaggle_TrainingSet/Data/Sorted/MildDR/Clean/'
IMG_H = 3888
IMG_W = IMG_H
LABEL = "macula"

def rdButton_action(label):
	global index, x1, x2, y1, y2, imgFile

	if (label == 'Close'):
		'''	Save the latest marked ROI info.	'''
		if (x1 != -1):	
			print("%s : (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (imgFile, x1, y1, x2, y2))
			fp.write("- boxes:\n")
			fp.write("  - {label: '%s', occluded: false, x_max: %3.4f, x_min: %3.4f, y_max: %3.4f, y_min: %3.4f}\n" % (LABEL, x1, x2, y1, y2))
			fp.write("  path: '%s'" % (imgFile))
		fp.close()
		plt.close()

		
	elif (label == 'Next'):
		if (index != len(imgList)):
			'''	Load an image and mark ROI'''
			imgFile = DB_DIR + imgList[index]		
			mark_ROI()

			'''	Save file name and '''
			if (x1 != -1):
				print("%s : (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (imgFile, x1, y1, x2, y2))
				fp.write("- boxes:\n")
				fp.write("  - {label: '%s', occluded: false, x_max: %3.4f, x_min: %3.4f, y_max: %3.4f, y_min: %3.4f}\n" % (LABEL, x1, x2, y1, y2))
				fp.write("  path: '%s'\n" % (imgFile))

			'''	For next use. '''
			index += 1
			x1 = -1
			x2 = -1
			y1 = -1
			y2 = -1

def toggle_selector(event):
	print(' Key pressed.')
	if event.key in ['Q', 'q'] and toggle_selector.RS.active:
		print(' RectangleSelector deactivated.')
		toggle_selector.RS.set_active(False)
		if event.key in ['A', 'a'] and not toggle_selector.RS.active:
			print(' RectangleSelector activated.')
			toggle_selector.RS.set_active(True)

def line_select_callback(eclick, erelease):
	global x1, y1, x2, y2

	'eclick and erelease are the press and release events'
	x1, y1 = eclick.xdata, eclick.ydata
	x2, y2 = erelease.xdata, erelease.ydata
	print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

def mark_ROI():
	'''	Load image. '''
	img =	load_img(
				imgFile,
				grayscale = False,	
				target_size = [IMG_H, IMG_W],
				interpolation = 'bicubic'	
			)
	img = np.array(img)

	'''	Display image. '''
	title = '(' + str(index + 1) + '/'+ str(len_imgList) + ') '+ imgFile
	current_ax = fig.add_subplot(1, 1, 1)	
	plt.cla()
	plt.title(title)
	plt.axis('off')
	plt.imshow(img)

	'''	Refresh canvas to see the loaded image immediately. '''
	fig.canvas.draw()
	fig.canvas.flush_events()

	'''	Select Region-Of-Interest (ROI) '''
	rectprops = dict(facecolor = 'none', edgecolor = 'black', alpha = 2.0, fill = False)
	toggle_selector.RS = RectangleSelector(
							current_ax, line_select_callback,
							drawtype = 'box', useblit = False,
							button = [1, 3],  # don't use middle button
							minspanx = None, minspany = None,
							spancoords = 'pixels', rectprops = rectprops,
							interactive = True)

'''	Display images one by one for manual categorization.	'''
def display_canvas():
	plt.gcf().canvas.set_window_title('ROI Marker')
	rax = plt.axes([0.9, 0.4, 0.1, 0.15])
	rdButton = RadioButtons(rax, ('Next', 'Close'),(False, False))
	rdButton.on_clicked(rdButton_action)

	plt.show()

def makeImgList():
	imgList = []
	i = 0
	if (os.path.isdir(DB_DIR)):
		for dirPath, subDirList, imgFileList in os.walk(DB_DIR):
			for imgFile in imgFileList:
				print('{}. {}'.format(i, imgFile))
				imgList.append(imgFile)
				i += 1
	else:
		print('{} does not exist.'.format(DB_DIR))

	return imgList

if __name__ == '__main__':
	'''	Make a list of all images in the database directory.	'''
	imgList = makeImgList()
	len_imgList = len(imgList)

	'''	Make an empty tuple to hold the file name which will be
		loaded and whose ROI area will be marked. '''
	imgFile = []

	'''	Open a file to save image file name and co-ordinates of ROI area. '''
	now = datetime.datetime.now()
	fileName = DIR + 'ROI_' + now.strftime("%Y-%m-%d-%H:%M") + '.yaml'
	fp = open(fileName,'w+')

	'''	Initialise co-ordinates of mouse click and mouse relase. '''
	x1 = -1
	x2 = -1
	y1 = -1
	y2 = -1

	'''	Display frame for loading images and marking ROI area.	'''
	index = 0
	fig = plt.figure(figsize = [20,20])
	display_canvas()

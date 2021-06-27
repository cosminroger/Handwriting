# -*- coding: utf-8 -*-
import time
start_time = time.process_time()
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.filters import sobel
from scipy.signal import find_peaks
from skimage.morphology import skeletonize
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import shutil
import sys
import glob



#Projections for Line and character Segmentation
def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)  
def vertical_projections(sobel_image):
	return np.sum(sobel_image, axis = 0)


#Changing size of image to 32x32 and changing it up for model
def prepare(filepath):
	size = 32
	img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	new = cv2.resize(img,(size,size))
	return new.reshape(-1,size,size,1)

debug = False     #True to view the process visually -> Graph plots
square = False		#If converted image is needed to be converted to 32x32 size
curr_dir = os.getcwd()    #Get address of current directory

if os.path.exists(curr_dir+"\\Segmented_Lines"):
	shutil.rmtree(curr_dir+"\\Segmented_Lines")
if os.path.exists(curr_dir+"\\Segmented_Characters_Per_Lines"):
	shutil.rmtree(curr_dir+"\\Segmented_Characters_Per_Lines")
if os.path.exists(curr_dir+"\\results"):
	shutil.rmtree(curr_dir+"\\results")


path = curr_dir+"/Segmented_Lines"        #Folder to keep the cropped lines
os.mkdir(path)
path = curr_dir+"/Segmented_Characters_Per_Lines"		#Folder to keep single character images folder wise
os.mkdir(path)
path = curr_dir+"/results"		#Folder to keep text files results
os.mkdir(path)


file = sys.argv[1]
imgfldr_path = file

img_list = image_array = sorted(glob.glob(file+'/*'))

img_list_len = len(img_list)
print(img_list_len)



for img_number in range(img_list_len):

	img = rgb2gray(imread(img_list[img_number]))    #Read image
	height, width = img.shape   #Get size of image

	sobel_image = sobel(img)   #Apply Sobel edge detection filter on image  ///// For later


	hpp = horizontal_projections(sobel_image)    #Find horizontal projection of pixels per row in image
	#Block only for debugging
	if debug == True:
		plt.plot(hpp)
		plt.show()


	peaks, _ = find_peaks(hpp, distance=150, height = 10)    #Find local maxima peaks in plot
	#Block only for debugging
	if debug == True:
		plt.plot(hpp)
		plt.plot(peaks, hpp[peaks], "x")
		plt.show()


	#Block only for debugging
	if debug == True:
		for x in range(len(peaks)):                     #Plot peaks on image
			plt.axline((0,peaks[x]),(width,peaks[x]))
		plt.imshow(img, cmap="gray")
		plt.show()


	h_w = 100   #Window for cropping around the centre line

	os.mkdir(curr_dir+"/Segmented_Lines/"+str(img_number))

	for x in range(len(peaks)):     #Crop each line and save as different image
		img3 = img[peaks[x] - h_w:peaks[x] + h_w, 0:width]        #Up to down, Right to left
		plt.imshow(img3, cmap="gray")
		plt.axis('off')
		plt.savefig('Segmented_Lines/'+str(img_number)+'/Line_'+str(x+1)+'_image.png', bbox_inches='tight',pad_inches = 0,dpi=300)


	######################################################################################################################
	#Once lines are cropped, we crop the characters to save them line wise, folder wise



	image_array = sorted(glob.glob('Segmented_Lines\\'+str(img_number)+'\\*.png'))
	#Get address of all lines present in image


	no_of_lines = len(image_array)
	print(no_of_lines)

	for x in range(no_of_lines):
		os.makedirs(curr_dir+"/Segmented_Characters_Per_Lines/"+str(img_number)+"/"+str(x+1))
		#Make specific folders per lines to store characters per line

	counter = 1   #Tracking current line

	#Going through each line to seperate characters
	for x in image_array:
		img = rgb2gray(imread(x))
		height, width = img.shape
		sobel_image = sobel(img)

		#Converting image to binary for skeletization
		im_gray = np.array(Image.open(x).convert('L')) #'L' for single channel grayscale image
		thresh = 128
		maxval = 1
		im_bool = (im_gray > thresh) * maxval
		inv_bin = np.invert(im_bool) + 2
		skeleton = skeletonize(inv_bin)

		#Taking vertical projection and inverting graph
		vpp = vertical_projections(skeleton)
		vpp = vpp * -1

		#Character segmentation through peaks( Trough as we inverted graph)
		peaks, values = find_peaks(vpp, distance=30, height = -4, width = -7)
		
		#Block only for debugging
		if debug == True:
			plt.plot(vpp)
			plt.plot(peaks, vpp[peaks], "x")
			for x in range(len(peaks)):
				plt.axline((peaks[x],0),(peaks[x],height))
			plt.imshow(skeleton, cmap = 'gray')
			plt.show()
			plt.close()

		#Adding '0' and width of image for better crop of character
		if (peaks[0]-50 < 0) or (peaks[-1]+50 > width):
			peaks = np.concatenate(([0],peaks))
			peaks = np.append(peaks,[width])
		else:
			peaks = np.concatenate(([peaks[0]-50],peaks))
			peaks = np.append(peaks,[peaks[-1]+50])


		#Cropping each alphabet and saving per line
		for y in range(len(peaks)-1):
			crop = img[0:height, peaks[y]:peaks[y+1]] #Height of crop -> Width of crop
			plt.imshow(crop, cmap="gray") #Convert to grayscale
			plt.axis('off')
			
			if square:
				name = 'Segmented_Characters_Per_Lines\\'+str(img_number)+'\\'+str(counter)+'\\char_'+str(y+1)+'.png'
				plt.savefig(name, bbox_inches='tight',pad_inches = 0,dpi=100)
				crop_image = cv2.resize(cv2.imread(name,cv2.IMREAD_GRAYSCALE),(32,32))
				cv2.imwrite(name , crop_image)
			else:
				plt.savefig('Segmented_Characters_Per_Lines\\'+str(img_number)+'\\'+str(counter)+'\\char_'+str(y+1)+'.png', bbox_inches='tight',pad_inches = 0,dpi=100)
			
			print("Cropped line: "+str(counter)+'/'+str(len(image_array))+" character: "+str(y+1)+"/"+str(len(peaks)-1))



		counter += 1



	########################################################################################################

	names = ["Alef","Ayin","Bet","Dalet","Gimel","He","Het","Kaf","Kaf-final","Lamed","Mem","Mem-medial","Nun-final","Nun-medial","Pe","Pe-final","Qof","Resh","Samekh","Shin","Taw","Tet","Tsadi-final","Tsadi-medial","Waw","Yod","Zayin"]
	icons = ["א","ע","ב","ד","ג","ה","ח","כ","ך","ל","מ","ם","ן","נ","פ","ף","ק","ר","ס","ש","ת","ט","ץ","צ","ו","י","ז"]
	symbols = ['\u05D0','\u05E2','\u05D1','\u05D3','\u05D2','\u05D4','\u05D7','\u05DB','\u05DA','\u05DC','\u05DD','\u05DE','\u05DF','\u05E0','\u05E4','\u05E3','\u05E7','\u05E8','\u05E1','\u05E9','\u05EA','\u05D8','\u05E5','\u05E6','\u05D5','\u05D9','\u05D6']
	style = ['Archaic','Hasmonean','Herodian']
	style_counter = [0,0,0]

	#Loading Pre Trained Model
	model = tf.keras.models.load_model('model_reco.h5')

	alpha_direct = glob.glob('Segmented_Characters_Per_Lines\\'+str(img_number)+'/*')
	#Counting total number of lines to go over


	text_recognition = open('results\\'+'img_'+str(img_number+1)+'_characters.txt', 'w', encoding="utf-8")


	#Counting number of alphabets per line to go over
	for x in range(len(alpha_direct)):
		beta_direct = glob.glob(alpha_direct[x]+'/*')
		for y in range(len(beta_direct)):
			values = [] #Saving prediction values in a list
			prediction = model.predict([prepare(beta_direct[y])])#This is a numpyarray
			prediction_list = list(prediction)

			for k in range(len(prediction_list[0])):
				values.append(prediction_list[0][k])

			max_value = max(values)
			max_index = values.index(max_value)
			print("Char:"+str(y+1)+str(symbols[max_index])+str(names[max_index]), end = " ")
			text_recognition.write("%s" % symbols[max_index])
		print("")
		text_recognition.write("\n")
	text_recognition.close()


	model = tf.keras.models.load_model('model_dialect.h5')
	alpha_direct = glob.glob('Segmented_Characters_Per_Lines\\'+str(img_number)+'/*')
	dialect_recognition = open('results\\'+'img_'+str(img_number+1)+'_style.txt', 'w', encoding="utf-8")
	for x in range(len(alpha_direct)):
		beta_direct = glob.glob(alpha_direct[x]+'/*')
		for y in range(len(beta_direct)):
			values = [] #Saving prediction values in a list
			prediction = model.predict([prepare(beta_direct[y])])#This is a numpyarray
			prediction_list = list(prediction)


			for k in range(len(prediction_list[0])):
				values.append(prediction_list[0][k])

			max_value = max(values)
			max_index = values.index(max_value)
			style_counter[max_index] += 1
			#print("Char:"+str(y+1)+str(symbols[max_index])+str(names[max_index]), end = " ")
			#dialect_recognition.write("%s" % symbols[max_index])
		#print("")

	max_value = max(style_counter)
	max_index = style_counter.index(max_value)
	dialect_recognition.write("%s" % style[max_index])
	dialect_recognition.close()








#To calculate execution time
print("--- %.2f seconds ---" % (time.process_time() - start_time))
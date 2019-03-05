"""
Purpose: prepare Dataset for training a classification Neural Network model
Load all images and labels in two lists and save them in a Dataset file to be loaded by the model

---
February 7, 2019
Authors:    Sangeeta Biswas
            Caio Oliveira (github.com/caduunb)
"""

#   Load Modules
# ---
from tensorflow.keras.layers     import Dense, Flatten
from tensorflow.keras.models     import Sequential
from tensorflow.keras.optimizers import Adam"""

#   Load dataset and create label list
# ---
path1 = '/home/user/path/to/image/Dataset'
label = 0
labelList = []
fileNameList = []
labelDict = {}
LABEL_NO = 5
for dirPath, subDirList, fileList in os.walk(path1):
    for subDir in subDirList:
        path2 = dirPath + '/' + subDir
        labelDict[label] = subDir
        #print (subDir)
        for subdirPath, subsubDirList, subfileList in os.walk(path2):
            for file in subfileList:
                fileName = path2 + '/' + file
                fileNameList.append(fileName)
                labelList.append(label)
                #print (label, fileName)
        label += 1
        if label == LABEL_NO:
            break
print (labelDict)

#   Create image list
# ---
GRAYSCALE = False
n = len(fileNameList)
imgList = np.zeros([n,64,64,3])
i = 0
for file in fileNameList:
    print (file)
    img = load_img(
        file,
        grayscale = GRAYSCALE,	
        target_size = [64,64],	#	[IMG_H, IMG_W],	#	None / Tuple [h, w], e.g [64, 64]
        interpolation = 'bicubic'	#	Resampling methods such as 
	#	'bilinear', 'nearest', 'bicubic'
	)
    imgList[i] = np.array(img)
    i += 1
print (imgList.shape)

#   Save lists in a file .npz
# ---
saveFileName =  '/home/user/example/project/' + 'dataset_example.npz'
np.savez(
    saveFileName,
    imageList = imgList,
    labelList = labelList,
    labelDict = labelDict,)

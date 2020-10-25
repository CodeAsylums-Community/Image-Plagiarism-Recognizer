#use opencv contrib library
import cv2

#creating the instance of ORB algo class
orb = cv2.ORB_create(nfeatures=900)

#creating the instance of FREAK algo class
freakExtractor = cv2.xfeatures2d.FREAK_create()

#reading images
image1 = cv2.imread('images/original_golden_bridge.jpg')
image2 = cv2.imread('images/mixed_colors.jpg')

#converting the images to grayscale
img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

#findings the initial keypoints using ORB
keypoints1= orb.detect(img1,None)
keypoints2= orb.detect(img2,None)


#optimising the keypoints and finding their descriptors using FREAK
keypoints1,descriptors1 = freakExtractor.compute(img1,keypoints1)
keypoints2,descriptors2 = freakExtractor.compute(img2,keypoints2)

#creating the instance of BruteForce Matcher class for feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

#matching the features of both the images
matches = bf.knnMatch(descriptors1,descriptors2,k=2)

#creating the list for storing the good features
good = []
#iterating over the matching features of compare good features
for m,n in matches:
	#score = score +m.distance**2 + n.distance**2
	if m.distance < .796875*n.distance:
		good.append([m])


#calculating the score		
score = len(good) / ((len(keypoints1)+len(keypoints2))/2) * 100
print("Percent match: {:.2f} %".format(score))

if(float(score)<20.0):
	print("Images are not Palgarised")

elif(float(score)<50.0 and float(score)>20.0):
	print("Images are mildly palgarised")
else:
	print("Images are completely palgarised")

#plotting an image consiting of lines of matching features for both images
img3= cv2.drawMatchesKnn(img1,keypoints1,img2,keypoints2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#uncomment these lines to see the features identified for the image and change plt.imshow()accordingly
img4=cv2.drawKeypoints(img1,keypoints1,img1,color=(255,0,0))
img5=cv2.drawKeypoints(img2,keypoints2,img1,color=(255,0,0))

#displaying the image on console output
cv2.imshow("img3", cv2.resize(img3, None, fx=0.4, fy=0.4))
cv2.imwrite("feature_matching.jpg", img3)


cv2.imshow("image1", cv2.resize(img4, None, fx=0.4, fy=0.4))
cv2.imshow("image2", cv2.resize(img5, None, fx=0.4, fy=0.4))

cv2.waitKey(0)
cv2.destroyAllWindows()
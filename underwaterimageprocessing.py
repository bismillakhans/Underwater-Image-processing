
def mainFunction(img):
	import cv2
	import numpy as np
	#%matplotlib inline
	from skimage.io import imread
	from matplotlib import pyplot as plt
	from scipy.ndimage.filters import gaussian_filter
	from skimage import img_as_float
	from skimage.color import rgb2ycbcr
	from skimage.color import ycbcr2rgb

	#Normalize to 0-255
	def normalise(I):
	  N = np.uint8(255 * ((I - np.min(I))) / (np.max(I) - np.min(I)))
	  return N

	#white balance using gray world algorithm
	def white_balance(img):
	  row, col, ch = img.shape
	  output = np.zeros(np.shape(img))
	  for j in range(0,3):
	    scalVal = sum(sum(img))/(row*col)
	    #print(scalVal)
	    output[:,:,j] = img[:,:,j] * (.725/scalVal[j])
	  return output

	#generate gaussian pyramid
	def genrateGaussianPyr(A, level):
	  G = A.copy()
	  gpA = [G]
	  for i in range(level):
	    G = cv2.pyrDown(G)
	    gpA.append(G)
	  return gpA

	#generate laplacian pyramid
	def genrateLaplacianPyr(A, level):
	  G = A.copy()
	  gpA = [G]
	  for i in range(level):
	    G = cv2.pyrDown(G)
	    gpA.append(G)  

	  lpA = [G]
	  for i in range(level,0,-1):    
	    GE = cv2.pyrUp(gpA[i])
	    r,c,ch=gpA[i-1].shape
	    #print(gpA[i-1].shape)
	    #print(GE.shape)
	    GE=cv2.resize(GE,(c,r))
	    #print(GE.shape)
	    L = cv2.subtract(gpA[i-1], GE)
	    lpA.append(L)
	  return lpA


	#a10
	img = imread(img)
	#plt.imshow(img)
	#plt.show()


	#color compensation
	#normalize Image 0-1
	imgNormalised = np.divide((img-np.min(img)), (np.max(img)-np.min(img)))

	#split the channels
	r = imgNormalised[:,:,0]
	g = imgNormalised[:,:,1]
	b = imgNormalised[:,:,2]

	rmean = r.mean()
	gmean = g.mean()
	bmean = b.mean()

	alpha = 1
	#  the compensated red channel Irc at everypixel location(x)
	Irc = r + alpha*(gmean - rmean) * (1 - r) * g
	#  the compensated blue channel Ibc at everypixel location(x)
	Ibc = b + alpha*(gmean - bmean) * (1 - b) * g
	# New image
	newImg = np.zeros(np.shape(img))
	newImg[:, :, 0] = Irc 
	newImg[:, :, 1] = g
	newImg[:, :, 2] = Ibc

	#plt.imshow(newImg)
	#plt.show()

	#white balance
	wb_img = white_balance(newImg)
	wb_img =  normalise(wb_img)
	#plt.imshow(wb_img)
	#plt.show()

	#Gamma Correction
	from skimage import exposure
	gamma_img = exposure.adjust_gamma(wb_img, 3)
	#plt.imshow(gamma_img)
	#plt.show()


	#from skimage.filters import unsharp_mask




	def _unsharp_mask_single_channel(image, radius, amount, vrange):
	  blurred = gaussian_filter(image,sigma=radius,mode='reflect')

	  result = image + (image - blurred) * amount
	  if vrange is not None:
	    return np.clip(result, vrange[0], vrange[1], out=result)
	  return result


	def unsharp_mask(image, radius=1.0, amount=1.0, multichannel=False,preserve_range=False):
	    
	  vrange = None  
	  if preserve_range:
	    fimg = image.astype(np.float)
	  else:
	    fimg = img_as_float(image)
	    negative = np.any(fimg < 0)
	    if negative:
	      vrange = [-1., 1.]
	    else:
	      vrange = [0., 1.]

	  if multichannel:
	    result = np.empty_like(fimg, dtype=np.float)
	    for channel in range(image.shape[-1]):
	      result[..., channel] = _unsharp_mask_single_channel(fimg[..., channel], radius, amount, vrange)
	    return result
	  else:
	    return _unsharp_mask_single_channel(fimg, radius, amount, vrange)
	sha_img = unsharp_mask(wb_img, radius=3, amount=1)
	sha_img = normalise(sha_img)
	#plt.imshow(sha_img)
	#plt.show()


	#weight calculation

	#Local contrast weight map


	def local_contrast_weight(img):
	  local_con_wt = np.zeros(img.shape)
	  Ycbcr = rgb2ycbcr(img)
	  Y_factor = Ycbcr[:,:,0]
	  #print(max(np.ravel(Ycbcr)))
	  laplacian = cv2.Laplacian(Y_factor,cv2.CV_64F)
	  Y_new = np.abs(Y_factor+laplacian)
	  local_con_wt[:,:,0] = Y_new
	  local_con_wt[:,:,1] = img[:,:,1]
	  local_con_wt[:,:,2] = img[:,:,2]
	  local_new=ycbcr2rgb(local_con_wt)
	  return local_new


	#Saturation weight maps
	def saturation_weight(img):
	  sat_wt=np.zeros(img.shape)
	  R=img[:,:,0]
	  G=img[:,:,1]
	  B=img[:,:,2]
	  Ycbcr=rgb2ycbcr(img)
	  Y_factor=Ycbcr[:,:,0]
	  wght=np.sqrt(1/3*(np.square(R-Y_factor)+np.square(G-Y_factor)+np.square(B-Y_factor)))
	  sat_wt[:,:,0] = wght
	  sat_wt[:,:,1] = wght
	  sat_wt[:,:,2] = wght
	  return sat_wt


	#Saliency weight maps

	def saliency_weight(img):
	  img = 255 * img /(max(img.flatten()))
	  sal_wt=np.zeros(img.shape)
	  Igauss = cv2.GaussianBlur(img,(5,5),0)
	  Imean=img.mean()
	  #print(Imean)
	  normI = Igauss-Imean
	  R = normI[:,:,0]
	  G = normI[:,:,1]
	  B = normI[:,:,2]
	  NormR = R / (R+G+B)
	  NormG = G / (R+G+B)
	  NormB = B / (R+G+B)
	  sal_wt[:,:,0] = NormR
	  sal_wt[:,:,1] = NormG
	  sal_wt[:,:,2] = NormB
	  return (sal_wt)



	#Sum Weight Map

	def sum_weight(inp1,inp2,inp3):
	  inp1=np.double(inp1)
	  inp2=np.double(inp2)
	  inp3=np.double(inp3)
	  out=(inp1+inp2+inp3)
	  return out

	#normlized Weight Map
	def norm_weight(inp1,inp2):
	  #inp1max=np.max(inp1.flatten())
	  #inp2max=np.max(inp2.flatten())
	  #inp1=255*inp1/inp1max
	  #inp2=255*inp2/inp2max
	  inp1=np.double(inp1)
	  inp2=np.double(inp2)
	  suminp = inp1+inp2+0.0001
	  inp1 = np.divide(inp1,suminp)
	  inp2 = np.divide(inp2,suminp)
	  return inp1, inp2

	#finding weights
	fus_inp1 = gamma_img
	fus_inp2 = sha_img
	out_11 = local_contrast_weight(fus_inp1)
	out_11 =normalise(out_11)
	out_12 = saturation_weight(fus_inp1)
	out_12 =normalise(out_12)
	out_13 = (saliency_weight(fus_inp1))
	out_13 =normalise(out_13)

	out_21 = local_contrast_weight(fus_inp2)
	out_21 =normalise(out_21)
	out_22 = saturation_weight(fus_inp2)
	out_22 =normalise(out_22)
	out_23 = ( saliency_weight(fus_inp2))
	out_23 =normalise(out_23)
	out_1 = sum_weight(out_11, out_12, out_13)
	#out_1 =normalise(out_1)
	out_2 = sum_weight(out_21, out_22, out_23)
	#out_2 =normalise(out_2)

	out1, out2 = norm_weight(out_1, out_2)
	out1=normalise(out1)
	out2=normalise(out2)

	#generate pyramid
	fus_inp1 = (gamma_img) 
	fus_inp2 = (sha_img) 
	w_py1 = (genrateGaussianPyr(out1,2))
	w_py2 = (genrateGaussianPyr(out2,2))
	i_py1 = (genrateLaplacianPyr(fus_inp1,2))
	i_py2 = (genrateLaplacianPyr(fus_inp2,2))

	fus_inp1=np.double(fus_inp1)
	fus_inp2=np.double(fus_inp2)


	from skimage.transform import resize
	row, col, ch = fus_inp1.shape

	fused_image = np.zeros(fus_inp1.shape)
	for i in range(0,3):
	  fus_w1 = resize(w_py1[i], [row, col, ch], preserve_range= True)
	  fus_w2 = resize(w_py2[i], [row, col, ch], preserve_range= True)

	  fus_i1 = resize(i_py1[i], [row, col, ch], preserve_range= True)
	  fus_i2 = resize(i_py2[i], [row, col, ch], preserve_range= True)
	  #fus_w1 = im2double(fus_w1)
	  #fus_w2 = im2double(fus_w1)  
	  #fus_i1 = im2double(fus_w1)
	  #fus_i2 = im2double(fus_w1)
	  fused_image = fused_image + np.multiply(fus_w1,fus_i1) + np.multiply(fus_w2, fus_i2)


	#output_image
	fused_image=normalise(fused_image)
	
	res = cv2.resize(fused_image, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
	cv2.imshow("imG",res)
	
	#cv2.imwrite(os.path.join(path , 'out.jpg'), img)
	#cv2.imwrite("/home/arunima/flask/static/img/in.png",img)
	cv2.imwrite("/home/arunima/flask/aru_UI/static/out.png",fused_image)
	cv2.waitKey(0)
	#plt.show()
	
	#input_image
	#plt.imshow(img)
	#plt.show()
#mainFunction()
#def runSpeedFunction():
	#pass
	
#if __name__ == '__main__':
	
#	pass



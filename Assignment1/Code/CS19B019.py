
#Import Libraries Required
import numpy as np
from numpy import array
from numpy import linalg as LA
import matplotlib.pyplot as plt
from PIL import Image

#Convert Image into numpy array
#use Python Imaging Library and get matrix form of given gray scale image
image = Image.open(r"33.jpg")
Image_Matrix = array(image)

"""# **EIGEN VALUE DECOMPOSITION**"""
#compute the eigenvalues and right eigenvectors of a given square array with the help of numpy.linalg.eig().
eigval,eigvec = LA.eig(Image_Matrix)

#get the sorted order of the  above eigen values in decreasing order based on magnitude
#using argsort get the sorted order of the array.
#arrange the matrix coloumns according to the above produced sorted order
def myfn(x):
  return -abs(x)
order = np.argsort(myfn(eigval))
TotalMatrix=np.vstack([eigval,eigvec])
TotalMatrix = TotalMatrix[:,order]

#Eigen decomposition of Image_Matrix = UD(U inverse)
#U is matrix formed by eigen vectors as coloumns
#D is diagonal matrix with diagonal elements as eigen values sorted in decreasing order
#V is inverse of U
U = TotalMatrix[1:,0:256]
D = np.diag(TotalMatrix[0][0:256])
V = LA.inv(U)

# fn_evd is a function that takes first highest k eigen values in the D and calculates the norm
#between the original image and the new approximate image
def fn_evd(k):
    A = U[:,0:k]
    B = D[0:k,0:k]
    C = V[0:k,:]
    Q = np.dot(B,C)
    New_Image=np.dot(A,Q)
    Error = Image_Matrix - New_Image
    frob_norm = LA.norm(Error,'fro') #This calculates the frobenius norm
    New_Image=np.abs(New_Image)
    Error =np.abs(Error)
    New_Image = New_Image.astype(np.uint8)
    Error = Error.astype(np.uint8)
    #plot reconstructed and error images#
    if k==50 or k==100 or k==151 or k==200:
      plt.suptitle("EVD k="+str(k))
      plt.subplot(1,2,1)
      plt.imshow(New_Image,"gray")
      plt.title("Image")
      plt.subplot(1,2,2)
      plt.imshow(Error,"gray")
      plt.title("Error")
      plt.show()
    return frob_norm

#As in eigen decomposition , if a complex eigen value is included then the corresponding complex conjugate also need to be included.
#sum is a variable that stores the running sum of eigen values, consider k only if this running sum is real.
sum =0;
x_evd = np.array([])
y_evd = np.array([])
for i in range(1,257):
    sum=sum+D[i-1,i-1]
    #If sum is real then it implies that every eigen value and its conjugate both are included.
    #So, consider k for plotting only if imaginary part of sum is 0
    if sum.imag==0 :
         x_evd = np.append(x_evd,i+1)
         y_evd = np.append(y_evd,fn_evd(i))

plt.plot(x_evd,y_evd)         #plotting frobenius_norm vs k
plt.title("Eigen Value Decomposition")
plt.xlabel("k")
plt.ylabel("Frobenius norm")
plt.show()


"""## **SINGULAR VALUE DECOMPOSITION**"""

#change type to uint32 so that A@A.T will not fit in uint8
#Image_matrix = U@(Sigma)@(V.T)
#V is the matrix formed by eigen vectors of A@(A.T)
#Image_matrix = U@(Sigma)@(V.T) => (Image_Matrix)@V = U@Sigma (because V is orthogonal)
#As,U matrix is also orthonormal, make every coloumn of (Image_Matrix)@V to norm 1 to get U
#Image_matrix = U@(Sigma)@(V.T) => (Image_Matrix)@V = U@Sigma => (U.T)@(Image_Matrix)@V = Sigma
Image_Matrix = Image_Matrix.astype(np.uint32)
AtA = np.dot(Image_Matrix.T,Image_Matrix)
e,V = LA.eig(AtA)
U = Image_Matrix@V
for i in range(256):
  U[:,i]=U[:,i]/LA.norm(U[:,i])
Sigma = U.T@Image_Matrix@V

#Sort U,V,Sigma based on decreasing order of singular values#
order = np.diag(Sigma).argsort()
Sigma = np.diag(Sigma)[order[::-1]]
Sigma = np.diag(Sigma)
U=U[:,order[::-1]]
V=V[:,order[::-1]]

# fn_evd is a function that takes first highest k eigen values in the D and calculates
#the norm between the original image and the new approximate image
def fn_svd(k):
    A = U[:,0:k]
    B = Sigma[0:k,0:k]
    C = V[:,0:k].T
    Q = np.dot(B,C)
    New_Image=np.dot(A,Q)
    Error = Image_Matrix - New_Image
    frob_norm = LA.norm(Error,'fro')
    New_Image=np.abs(New_Image)
    Error =np.abs(Error)
    New_Image = New_Image.astype(np.uint8)
    Error = Error.astype(np.uint8)
    #plot reconstructed and error images#
    if k==50 or k==100 or k==151 or k==200:
      plt.gray()
      plt.suptitle("SVD k="+str(k))
      plt.subplot(1,2,1)
      plt.imshow(New_Image,"gray")
      plt.title("Image")
      plt.subplot(1,2,2)
      plt.imshow(Error,"gray")
      plt.title("Error")
      plt.show()
    return frob_norm

#form x_svd and y_svd and plot the y_svd vs x_svd#
x_svd = range(1,257)
y_svd = np.array([])
for i in range(1,257):
    y_svd = np.append(y_svd,fn_svd(i))
plt.plot(x_svd,y_svd)
plt.title("Singular Value Decomposition")
plt.xlabel("k")
plt.ylabel("Frobenius norm")
plt.show()

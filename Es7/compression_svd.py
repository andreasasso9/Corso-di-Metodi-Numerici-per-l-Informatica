from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

path = Path('Es7/img')
path.mkdir(parents=True, exist_ok=True)

A = imread('Es7/cane.png')
# A è una matrice  M x N x 3, essendo un'immagine RGB
# A(:,:,1) Red A(:,:,2) Blue A(:,:,3) Green
# su una scala tra 0 e 1
print(A.shape)


X = np.mean(A,-1); # media lungo l'ultimo asse, cioè 2
img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
#plt.show()
plt.savefig(f"{path}/original_image.png")

# If full_matrices=True (default), u and vT have the shapes (M, M) and (N, N), respectively.
# Otherwise, the shapes are (M, K) and (K, N), respectively, where K = min(M, N).
U, S_vals, VT = np.linalg.svd(X,full_matrices=False)
print(S_vals[5])
print(S_vals[20])
print(S_vals[100])
#i valori di S precedenti, in particolare S(100), forniscono una stima dell'errore commesso
# nella compressione dell'immagine con la SVD
S = np.diag(S_vals)

j=0
for r in (5,20,100):
	energy = S_vals[:r].sum() / S_vals.sum()
	print(f"Porzione energia (sum sigma) per k={r}: {energy:.6f} ({energy*100:.2f} %)")
	Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
	plt.figure(j+1)
	j +=1
	img = plt.imshow(Xapprox)
	img.set_cmap('gray')
	plt.axis('off')
	plt.title('r = ' + str(r))
	#plt.show()
	plt.savefig(f"{path}/compressed_image_r"+str(r)+".png")
	
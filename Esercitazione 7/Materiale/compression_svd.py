from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

def preserved_energy(S, total_energy, k):
	partial_energy = np.sum(S[:k])
	return partial_energy / total_energy * 100

def get_minimum_k(S, threshold):
	# Punto di partenza per il target e per il k da raggiungere
	target = 0
	k = 0
	# Accumuliamo sugli elementi di S finché non "superiamo" la soglia
	for i in range(len(S)):
		target += S[i]
		k += 1  # Avendo cura di salvare k
		if target >= threshold:
			break  # Usciamo dal ciclo
	return k, target

A = imread('cane.png')
# A è una matrice  M x N x 3, essendo un'immagine RGB
# A(:,:,1) Red A(:,:,2) Blue A(:,:,3) Green
# su una scala tra 0 e 1
print(A.shape)

X = np.mean(A,-1); # media lungo l'ultimo asse, cioè 2
img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.savefig('cane_gray.png')

# If full_matrices=True (default), u and vT have the shapes (M, M) and (N, N), respectively.
# Otherwise, the shapes are (M, K) and (K, N), respectively, where K = min(M, N).
U, S, VT = np.linalg.svd(X,full_matrices=False)

# Esercizio 3
total_energy = np.sum(S) # La somma dei valori della matrice singolare
print(f"Total energy: {total_energy}")
for i in (5,20,100):
	print(f"Preserved energy for k = {i}: {preserved_energy(S=S, total_energy=total_energy, k=i)} %")

# Esercizio Facoltativo
print("=" * 100)

threshold = total_energy * 0.8 # La soglia che vogliamo raggiungere
k, target = get_minimum_k(S = S, threshold=threshold)
print(f"Threshold: {threshold}")
print(f"Accumulated sum: {target}")
print(f"Minimum k to reach the threshold: {k}")

S = np.diag(S)

j = 0
for r in (5, 20, 100):
	Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
	plt.figure(j+1)
	j += 1
	img = plt.imshow(Xapprox)
	img.set_cmap('gray')
	plt.axis('off')
	plt.title('r = ' + str(r))
	plt.savefig('cane' + str(r) + '.png')

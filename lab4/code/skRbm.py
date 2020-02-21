import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import accuracy_score
plt.rcParams['image.cmap'] = 'gray'

def read_mnist(dim=[28,28],n_train=60000,n_test=1000):

    """
    Read mnist train and test data. Images are normalized to be in range [0,1]. Labels are one-hot coded.
    """    
    import scipy.misc

    train_imgs = load_idxfile("train-images-idx3-ubyte")
    train_imgs = train_imgs / 255.
    train_imgs = train_imgs.reshape(-1,dim[0]*dim[1])

    train_lbls = load_idxfile("train-labels-idx1-ubyte")
    train_lbls_1hot = np.zeros((len(train_lbls),10),dtype=np.float32)
    train_lbls_1hot[range(len(train_lbls)),train_lbls] = 1.

    test_imgs = load_idxfile("t10k-images-idx3-ubyte")
    test_imgs = test_imgs / 255.
    test_imgs = test_imgs.reshape(-1,dim[0]*dim[1])

    test_lbls = load_idxfile("t10k-labels-idx1-ubyte")
    test_lbls_1hot = np.zeros((len(test_lbls),10),dtype=np.float32)
    test_lbls_1hot[range(len(test_lbls)),test_lbls] = 1.

    return train_imgs[:n_train],train_lbls_1hot[:n_train],test_imgs[:n_test],test_lbls_1hot[:n_test]

def load_idxfile(filename):

    """
    Load idx file format. For more information : http://yann.lecun.com/exdb/mnist/ 
    """
    import struct
        
    with open(filename,'rb') as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0 :
           raise Exception('Invalid idx file: unexpected magic number!')
        dtype,ndim = ord(_file.read(1)),ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0] for _ in range(ndim)]
        data = np.fromfile(_file, dtype=np.dtype(np.uint8).newbyteorder('>')).reshape(shape)
    return data



def gen_mnist_image(X):
    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)

def gen_mnist_image1(X):
    return np.rollaxis(np.rollaxis(X[0].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)


image_size = [28,28]
train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(train_imgs))
#plt.show()
plt.imshow(train_imgs[0].reshape(28,28))
#plt.show()

epochs = np.arange(10,20,2)
errorTrain = []
errorTest = []
for e in epochs:
	rbm = BernoulliRBM(n_components=500, learning_rate=0.01, batch_size=20, n_iter= e, random_state=0, verbose=True)
	rbm.fit(train_imgs)

	plt.figure(figsize=(20, 20))
	for i, comp in enumerate(rbm.components_[:100]):
	    plt.subplot(10, 10, i + 1)
	    plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.RdBu,
	               interpolation='nearest', vmin=-2.5, vmax=2.5)
	    plt.axis('off')
	plt.suptitle('100 components extracted by RBM', fontsize=16);
	plt.show()

	Y_train = rbm.components_.dot(train_imgs)
	Y_test = rbm.components_.dot(test_imgs)

	eTrain = accuracy_score(train_imgs, Y_train)
	eTest = accuracy_score(test_imgs, Y_test)
	errorTrain.append(eTrain)
	errorTest.append(eTest)
	"""
	Y_train = rbm.predict(train_imgs)
	Y_test = rbm.predict(test_imgs)

	eTrain = accuracy_score(train_imgs, Y_train)
	eTest = accuracy_score(test_imgs, Y_test)
	errorTrain.append(eTrain)
	errorTest.append(eTest)

	r = np.randint(0,test_imgs.shape[0])
	Y = rbm.predict(test_imgs[r])"""

	"""
	fig = plt.figure()
	a = fig.add_subplot(1,2,1)
	plt.imshow(test_imgs[r])
	a.set_title('Ground Truth')
	a = fig.add_subplot(1,2,2)
	plt.imshow(Y)
	a.set_title('Prediction')
	plt.axis('off')
	plt.show()

	plt.figure(figsize=(4.2, 4))
	"""

plt.plot(epochs,errorTrain,label='Train Error')
plt.plot(epochs,errorTest,label='Test Error')
plt.legend()
plt.show()






"""xx = X_train[:40].copy()
for ii in range(1000):
    for n in range(40):
        xx[n] = rbm.gibbs(xx[n])

plt.figure(figsize=(10,20))
plt.imshow(gen_mnist_image(xx))
plt.show()"""
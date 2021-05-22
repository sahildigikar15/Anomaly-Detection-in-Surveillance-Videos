import cv2
import mxnet as mx
from mxnet import nd ,gluon, autograd,cpu
import glob
import numpy as np
import os
from PIL import Image
from scipy import signal
from matplotlib import pyplot as plt


class ConvolutionalAutoencoder(gluon.nn.HybridBlock):

    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()

        with self.name_scope():
            self.encoder = gluon.nn.HybridSequential()
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Conv2D(32, 5, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.Conv2D(32, 5, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.Dense(2000))

            self.decoder = gluon.nn.HybridSequential()
            with self.decoder.name_scope():
                self.decoder.add(gluon.nn.Dense(32 * 22 * 22, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(32, 5, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(1, kernel_size=5, activation='sigmoid'))

    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = self.decoder[0](x)
        x = x.reshape((-1, 32, 22, 22))
        #         print(self.decoder)
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        x = self.decoder[3](x)
        x = self.decoder[4](x)
        return x


# test_file = sorted(glob.glob(os.path.join('static', 'uploader/*')))
# #test_file_gt = sorted(glob.glob(UCSD_FOLDER +'/UCSDped1/Test/Test024_gt/*'))
# a = np.zeros((len(test_file),2,100,100))

# for idx,filename in enumerate(test_file):
#
#     im = Image.open(filename)
#     im = im.resize((100,100))
#     a[idx,0,:,:] = np.array(im, dtype=np.float32)/255.0
#
# dataset = gluon.data.ArrayDataset(mx.nd.array(a, dtype=np.float32))
# dataloader = gluon.data.DataLoader(dataset, batch_size=1)

print("Done Reading and preprocessing")

def plot_regularity_score(model,dataloader):
  """
  Calculated regularity score per frame:
  Regularity Score = 1 - (e_t - min@t(e_t))/max@t(e_t)
  where e_t = sum over pixelwise l2 loss for each frame
  """
  e_t = []
  for image in dataloader:
    img = image[:,0,:,:].reshape(1,1,image.shape[-2],image.shape[-1])
    img = img.as_in_context(mx.cpu())
    output = model(img)
    output = (output.asnumpy().squeeze()*255).reshape(100*100,1)
    img = (img.asnumpy().squeeze()*255).reshape(100*100,1)
    e_xyt = np.linalg.norm(output-img,axis=1,ord=2)
    e_t.append(np.sum(e_xyt))
  e_t_min = min(e_t)
  e_t_max = max(e_t)
  reg_scores = []
  for i in range(len(e_t)):
    reg_scores.append(1 - ((e_t[i]-e_t_min)/e_t_max))
  return reg_scores

# model =  ConvolutionalAutoencoder()
# model.load_parameters(os.path.join('static', 'autoencoder_ucsd.params'))
# reg_scores_cae = plot_regularity_score(model,dataloader)
#
# print("Done loading params")

def plot_anomaly(img, output, diff, H, threshold, counter):
  """
  Plots the images along the axis to show the input, output of the model,
  difference between the 2, and their predicted anomalies as red dots on
  the input image.
  """
  fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(10, 3))
  ax0.set_axis_off()
  ax1.set_axis_off()
  ax2.set_axis_off()
  ax0.set_title('input image')
  ax1.set_title('reconstructed image')
  ax2.set_title('diff ')
  ax3.set_title('anomalies')
  ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
  cv2.waitKey(1000)
  cv2.destroyAllWindows()
  ax1.imshow(output, cmap=plt.cm.gray, interpolation='nearest')
  cv2.waitKey(1000)
  cv2.destroyAllWindows()
  ax2.imshow(diff, cmap=plt.cm.viridis, vmin=0, vmax=255, interpolation='nearest')
  cv2.waitKey(1000)
  cv2.destroyAllWindows()
  ax3.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
  cv2.waitKey(1000)
  cv2.destroyAllWindows()


  x,y = np.where(H > threshold)
  ax3.scatter(y,x,color='red',s=0.1)
  plt.axis('off')
  fig.savefig('static/predicted_image/' + str(counter) + '.png')

def model_evaluation(model,dataloader):
  loss_l2_per_frame = []
  threshold = 4*255
  counter = 0
  test_loss_metric = gluon.loss.SigmoidBCELoss()
  loss_per_frame = 0
  im_list = []
  i = 0
  for image in dataloader:
    counter = counter + 1
    img = image[:,0,:,:].reshape(1,1,image.shape[-2],image.shape[-1])
    mask = image[:,1,:,:].as_in_context(mx.cpu())
    img = img.as_in_context(mx.cpu())
    output = model(img)
    output = output.transpose((0,2,3,1))
    img = img.transpose((0,2,3,1))
    output = output.asnumpy()*255
    img = img.asnumpy()*255
    diff = np.abs(output-img)
    tmp = diff[0,:,:,0]
    H = signal.convolve2d(tmp, np.ones((4,4)), mode='same')
    H_new = mx.nd.array(np.where(H>threshold,1,0).reshape((1,100,100)),ctx=cpu())
    loss = test_loss_metric(H_new, mask)
    loss_l2_per_frame.append(loss.asscalar())
    plot_anomaly(img[0,:,:,0], output[0,:,:,0], diff[0,:,:,0], H, threshold, counter)

  print("Total loss per frame for anomalies predicted = ",sum(loss_l2_per_frame)/len(dataloader))

## Evaluating the model using the anomaly predictions and regularity scores
#model_evaluation(model,dataloader)
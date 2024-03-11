import numpy as np
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2 as cv2
import scipy
from scipy.spatial.distance import cosine
from scipy.ndimage import zoom

import warnings
warnings.filterwarnings("ignore")


######################
# Vars to evaluate models
######################
e_eval = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None']
e_dict = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt', 8:'None'}
e_eval_8 = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
e_dict_8 = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}
e_dict_8_inv = {'Neutral':0, 'Happy':1, 'Sad':2, 'Surprise':3, 'Fear':4, 'Disgust':5, 'Anger':6, 'Contempt':7}
e_eval_8_wrongorder = ['Neutral', 'Happy', 'Anger', 'Sad', 'Fear', 'Surprise', 'Disgust', 'Contempt']
e_dict_8_wrongorder = {0: 'Neutral', 1:'Happy', 2:'Anger', 3:'Sad', 4:'Fear', 5:'Surprise', 6:'Disgust', 7:'Contempt'}
e_eval_8_alphabetic = ['Anger','Contempt','Disgust','Fear','Happy','Neutral','Sad','Surprise']
e_dict_8_alphabetic = {0:'Anger',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}

###########################
# Elliptical grid mapping from square to circle
###########################
def coords_to_circle(x,y, side):
    return ( (x*np.sqrt(1-((y**2)/2))) * side/2,
             (y*np.sqrt(1-((x**2)/2))) * side/2)
def getTextSize(text, textSize = 1):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, textSize,1)[0]

#######################
# Draw data point on VA sphere
#######################
def draw_point(img, x_offset, y_offset, side, boxX, boxY, transform_to_circle):
    if (transform_to_circle):
        cCoords = coords_to_circle(boxX,boxY, side)
    else:
        cCoords = [boxX*side/2, boxY*side/2]
    img = cv2.circle(img, (int(cCoords[0]+x_offset+side/2),int(-cCoords[1]+y_offset+side/2)) , int(np.sqrt(side)), (255,255,255), -1)
    return img

############################
# Draw the VA wheel annotation in OpenCV
# args:
#   transform_to_circle : transform VA value to within the unit bound
#   predict_emotion : predict expression category based on VA region
############################
def drawVAgraph(img, x_offset, y_offset, side, valence, arousal, predict_emotion = True, transform_to_circle = False, concat_axis_descriptions = False):
    center_coordinates = ( int(x_offset + side / 2) , int( y_offset + side / 2 ))
    radius = int (side / 2)
    color = (255, 255, 255)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.circle(img, center_coordinates, radius, color, thickness)
    img = draw_point(img, x_offset, y_offset, side, valence, arousal, transform_to_circle)
    val_desc = 'Val' if concat_axis_descriptions else 'Valence'
    img = cv2.putText(img, val_desc, (int(x_offset + side),
                                       int(y_offset+side/2)),
                      font, 1, (255,255,255), 1, cv2.LINE_AA)
    aro_desc = 'Aro' if concat_axis_descriptions else 'Arousal'
    img = cv2.putText(img, aro_desc, (int(x_offset + side/2 - getTextSize(aro_desc)[0]/2),
                                       int(y_offset+side + getTextSize(aro_desc)[1])),
                      font, 1, (255,255,255), 1, cv2.LINE_AA)
    img = cv2.putText(img, '-', (int(x_offset + side/2 - getTextSize('-')[0]/2),
                                   int(y_offset+side - getTextSize('-')[1])),
                  font, 1, (255,255,255), 1, cv2.LINE_AA)
    img = cv2.putText(img, '+', (int(x_offset + side/2 - getTextSize('+')[0]/2),
                                   int(y_offset + 1.5 * getTextSize('+')[1])),
                  font, 1, (255,255,255), 1, cv2.LINE_AA)
    img = cv2.putText(img, '+', (int(x_offset + side - 1.5 * getTextSize('+')[0]),
                                   int(y_offset+side/2)),
                  font, 1, (255,255,255), 1, cv2.LINE_AA)
    img = cv2.putText(img, '-', (int(x_offset + 0.5 * getTextSize('-')[0]),
                                   int(y_offset+side/2)),
                  font, 1, (255,255,255), 1, cv2.LINE_AA)
                  
def aspectTransform( paar , origDim , newSize ):
    return ( int(paar[0] / origDim[0] * newSize), int(paar[1] / origDim[1] * newSize ))

#####################
# Draws a percentage bar in OpenCV
#####################
def draw_progress_bar(img, x, y, width, height, percentage):
    if(percentage>1):
        percentage /= 100
    # base
    img = cv2.rectangle(img, (x,y), (x+width, y+height), color=(0,0,255), thickness=-1)
    # overlay
    img = cv2.rectangle(img, (x,y), (x+int(width*percentage), y+height), color=(0,255,0), thickness=-1)
    # counter
    fontsize = getTextSize("%d " % (int(percentage*100)) + "%")
    img = cv2.putText(img, "%d " % (int(percentage*100)) + "%", 
        (x+width//2  - fontsize[0]//2  ,  y+height//2 + fontsize[1]//2 ),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    return img

#############################
# Hybrid generator
# Re-formats the pre-processed data for training
# of hybrid classifiers
#############################
class hybrid_generator(tf.keras.utils.Sequence):
    def __init__(self,dfiterator, step_size, returnWeights = True):
        self.dfiterator = dfiterator
        self.step_size = step_size
        self.returnWeights = returnWeights
    def __len__(self):
        return self.step_size
    def __getitem__(self, integer):
        tmp = self.dfiterator.next()
        return (tmp[0], (tmp[1][:,2:], tmp[1][:,:2]), tmp[2]) if self.returnWeights else (tmp[0], (tmp[1][:,2:], tmp[1][:,:2]))
    
###############################
# Shirley-Chiu Equiareal Mapping for Square/Circle
###############################
def s2c(x,y):
    if (x == 0 and y == 0):
        return (0,0)
    a = x
    b = y
    if (a*a > b*b):
        r = a
        phi = (np.pi / 4) * (b / a)
    else:
        r = b
        phi = (np.pi/2)-(np.pi/4)*(a/b)
    return (r*np.cos(phi), r*np.sin(phi))

def c2s(u,v):
    if (u == 0 and v == 0):
        return (0,0)
    r = np.sqrt(u*u + v*v)
    if (math.atan2(v,u) >= (-np.pi/4)):
        phi = math.atan2(v,u)
    else:
        phi = math.atan2(v,u) + 2*np.pi
    if(phi < np.pi/4):
        return (r, (4/np.pi)*r*phi)
    elif(phi < np.pi*3/4):
        return ((-4/np.pi)*r*(phi-np.pi/2),r)
    elif(phi < np.pi*5/4):
        return (-r, (-4/np.pi)*r*(phi-np.pi))
    else:
        return ((4/np.pi)*r*(phi-(3*np.pi/2)),-r)

##########################
#  Metrics from EmoNet for comparison with the same implementation
#  URL:
#  https://github.com/face-analysis/emonet/blob/master/emonet/metrics.py
##########################

def ACC(ground_truth, predictions):
    """Evaluates the mean accuracy
    """
    return np.mean(ground_truth.astype(int) == predictions.astype(int))

def RMSE(ground_truth, predictions):
    """
        Evaluates the RMSE between estimate and ground truth.
    """
    return np.sqrt(np.mean((ground_truth-predictions)**2))


def SAGR(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    return np.mean(np.sign(ground_truth) == np.sign(predictions))


def PCC(ground_truth, predictions):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    return np.corrcoef(ground_truth, predictions)[0,1]


def CCC(ground_truth, predictions):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    mean_pred = np.mean(predictions)
    mean_gt = np.mean(ground_truth)

    std_pred= np.std(predictions)
    std_gt = np.std(ground_truth)

    pearson = PCC(ground_truth, predictions)
    return 2.0*pearson*std_pred*std_gt/(std_pred**2+std_gt**2+(mean_pred-mean_gt)**2)
import pathlib
from enum import Enum
from typing import Optional, Union, List
import onnxruntime as ort
from deepface import DeepFace
import numpy as np
import cv2
import datetime
import onnx
import os
from PIL import Image
from torchvision import transforms
from pathlib import Path
import glob
import math
import gc
import matplotlib.pyplot as plt
from IPython.display import display



def extract_square_roi(image, x, y, w, h):
    # Calculate the side length of the square ROI
    side_length = max(w, h)

    # Create a black image with the same number of channels as the original image
    black_image = np.zeros((side_length, side_length, image.shape[2]), dtype=np.uint8)

    # Calculate bounding points
    x_start = (side_length - w) // 2
    y_start = (side_length - h) // 2
    x_end = x_start + w
    y_end = y_start + h
    
#     print(image.shape)
#     print(black_image.shape)
#     print(image[y : y + h, x : x + w].shape)
#     print(black_image[y_start:y_end, x_start:x_end].shape)
#     print('x', x, 'y', y, 'w', w, 'h', h)
#     print('xs', x_start, 'ys', y_start, 'xe', x_end, 'ye', y_end)

    # insert the ROI
    black_image[y_start:y_end, x_start:x_end] = image[y : y + h, x : x + w]

    return black_image, y_start, y_end, x_start, x_end


def is_image(f_name: str):
    return f_name.split('.')[-1] in ['png', 'bmp', 'jpg', 'jpeg']

def plot_point_on_circumplex_model(axs, circumplex_image, valence, arousal, point_color='blue'):
    # Convert to pixel coordinates
    width, height = circumplex_image.size
    pct = 0.92
    padding = (1-pct) / 2
    x = (valence * pct + 1) / 2 * (width - 2 * padding * width) + padding * width
    pct = 0.955
    padding = (1-pct) / 2
    y = (1 - (arousal * pct + 1) / 2) * (height - 2 * padding * height) + padding * height  # Flip y-axis
    x += 0.003 * width
    # Plot the point
    axs[1].plot(x, y, 'ro', color=point_color)

def show_dashboard(image_np_array, predictions, target=None, save_dashboard_fp=None):
    circ_fp = 'skeleton/images/VAmodel.png'
    im_trgt_size = (720,720)
    font = {
            'family': 'serif',
            'color':  'darkblue',
            'weight': 'normal',
            'size': 16,
        }
    font_heading = {
            'family': 'serif',
            'color':  'darkblue',
            'weight': 'normal',
            'size': 26,
        }
    emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
    
    # -----------
    
    # Load the images
    face_image = Image.fromarray(image_np_array)  # Image.open(image_fp)
    circumplex_image = Image.open(circ_fp)
    
    # Resize the images
    face_image = face_image.resize(im_trgt_size)
    circumplex_image = circumplex_image.resize(im_trgt_size)
    
    # Create a figure with a light gray background
    fig, axs = plt.subplots(1, 3, figsize=(20, 10), gridspec_kw={'width_ratios': [4, 4, 1]})
    fig.patch.set_facecolor('lightgray')
    
    # Add padding around the images
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Add the face image to the left
    axs[0].imshow(face_image)
    axs[0].axis('off')  # Hide axes

    # Add the circumplex image to the middle
    axs[1].imshow(circumplex_image)
    axs[1].axis('off')  # Hide axes
    
    # Define predictions
    predicted_valence = predictions[1][0]
    predicted_arousal = predictions[1][1]
    predicted_percentages = [100 * p for p in predictions[0]]
    prediction = np.argmax(predicted_percentages)
    
    # Display predicted valence arousal label
    plot_point_on_circumplex_model(axs, circumplex_image, predicted_valence, predicted_arousal)
    
    # Add progress bars to the right
    bars = axs[2].barh(emotions, predicted_percentages, color='darkblue', height=0.5)
    axs[2].set_xlim(0, 100)
    axs[2].invert_yaxis()  # labels read top-to-bottom
    axs[2].axis('off')  # Hide axes
    axs[2].patch.set_visible(False)  # Make background transparent
    
    # Add percentages on the bars
    for bar, percentage, emotion in zip(bars, predicted_percentages, emotions):
        width = bar.get_width()
        axs[2].text(width, bar.get_y() + bar.get_height()/2, f' {percentage:.3f}%', ha='left', va='center')
        axs[2].text(0, bar.get_y() - 0.16, f'{emotion}', ha='left', va='center', fontdict=font)
        
    # Add heading of predicted emotion and valence arousal values
    axs[0].text(10,-20, 'pred: ', fontdict=font_heading)
    font_heading['color'] = ('darkgreen' if prediction == target[0] else 'darkred') if target else font_heading['color']
    if target:
        target_eval = f" | TOP {sorted(predicted_percentages, reverse=True).index(target[0])+1}"
    else:
        target_eval = ""
    axs[0].text(130,-20, str(emotions[prediction]).upper() + target_eval, fontdict=font_heading)
    axs[1].text(10, -20, f"valence: {predicted_valence:.3f} | arousal: {predicted_arousal:.3f}", fontdict=font)
    
    if target:
        font_heading['color'] = 'darkgreen'
        font['color'] = 'darkgreen'
        axs[0].text(10,height + 40, 'actual: ' + str(emotions[target[0]]).upper(), fontdict=font_heading)
        axs[1].text(10, height + 40, f"valence: {target[1][0]:.3f} | arousal: {target[1][1]:.3f}", fontdict=font)
        plot_point_on_circumplex_model(axs, circumplex_image, target[1][0], target[1][1], point_color='green')

    # Save the dashboard as an image
    if save_dashboard_fp:
        path = os.path.split(save_dashboard_fp)[0]
        if path:
            os.makedirs(path, exist_ok=True)
        plt.savefig(save_dashboard_fp, bbox_inches='tight', pad_inches=0.1, transparent=True)
    
    display(fig)
    return axs

def enlarge_detected_face(orig_image_np, detected_obj, target_size):
    out_objs = []
    
    for obj in detected_obj:
        x_s = int(obj['facial_area']['x'])
        y_s = int(obj['facial_area']['y'])
        x_e = int(obj['facial_area']['w'] + obj['facial_area']['x'])
        y_e = int(obj['facial_area']['h'] + obj['facial_area']['y'])
        width = int(obj['facial_area']['w'])
        height = int(obj['facial_area']['h'])
        bigger_dimension = max([width, height])
        width_padding = bigger_dimension - width
        height_padding = bigger_dimension - height
#         face = orig_image_np[int(y_s-height_padding//2):int(y_e+height_padding//2),
#                             int(x_s-width_padding//2):int(x_e+width_padding//2), :]
        ys = max(0, int(y_s-height_padding//2))
        ye = min(orig_image_np.shape[0], int(y_e+height_padding//2))
        xs = max(0, int(x_s-width_padding//2))
        xe = min(orig_image_np.shape[1], int(x_e+width_padding//2))
        face = orig_image_np[ys:ye,xs:xe, :]
        face = cv2.resize(face, target_size).astype(np.uint8)
        out_obj = {'face': face, "facial_area": obj["facial_area"], "confidence": obj["confidence"]}
        out_objs.append(out_obj)

    return out_objs

class MobileNet(object):
    emotion_labels_list = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt", "None"]
    emotion_labels_dict = {0: "Neutral",1: "Happy",2: "Sad",3: "Surprise",4: "Fear",5: "Disgust",6: "Anger",7: "Contempt",8: "None",}
    
    # resize strategies
    class ResizeStrategy(Enum):
        STRETCH = "A"
        FILL_BLACK = "B"

    def __init__(
        self,
        model_fp,
        verbose=False,
        debug=False,
        resize_strategy=ResizeStrategy.FILL_BLACK,
        square2circ=True,
    ):
        self.mobilenet_input_shape = None
        self.mobilenet = None
        self.model_fp = model_fp
        self.verbose = verbose
        self.debug = debug
        self.resize_strategy = resize_strategy
        self.square2circ = square2circ

        self.load_model()

    def load_model(self):
        self.mobilenet = ort.InferenceSession(self.model_fp)
        self.mobilenet_input_shape = self.mobilenet.get_inputs()[0].shape
        model = onnx.load(str(self.model_fp))
        in_names = [_input.name for _input in model.graph.input]
        self.mobilenet_input_name = in_names[0]
        if self.verbose:
            print(f"model '{self.model_fp}' loaded")

    def preprocess(self, image):
        if self.resize_strategy == MobileNet.ResizeStrategy.FILL_BLACK:
            resized, _, _, _, _ = extract_square_roi(image, 0, 0, image.shape[1], image.shape[0])
            resized = cv2.resize(resized, (224, 224), interpolation=cv2.INTER_AREA)
        elif self.resize_strategy == MobileNet.ResizeStrategy.STRETCH:
            resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        else:
            raise ValueError("Invalid strategy enum")

        if self.verbose:
            print(f"image resized from {image.shape} to {resized.shape}")

        return resized

    def run(self, image):
        output = self.mobilenet.run(
            None,
            {"x": np.expand_dims(image, axis=0).astype(np.float32)/255},
        )

        return output
    
    @staticmethod
    def magnitude(vector): 
        return math.sqrt(sum(pow(element, 2) for element in vector))

    @staticmethod
    def circumplex2square(a):
        a = np.array(a)
        a_magn = MobileNet.magnitude(a)

        if a_magn > 1:
            print("Not in range")
            return

        if a[0] == 0 or a[1] == 0:
            return a

        a1_sign = -1 if a[0] < 0 else 1
        a2_sign = -1 if a[1] < 0 else 1
        if abs(a[0] / a[1]) > 1:
            point_on_square = (1., abs(1/a[0]*a[1]))
        else:
            point_on_square = (abs(1/a[1]*a[0]), 1.)
        point_on_square = np.array([a1_sign*point_on_square[0], a2_sign*point_on_square[1]])

        b = a_magn * point_on_square

        return b

    @staticmethod
    def square2circumplex(b):
        b = np.array(b)
        if abs(b[0]) > 1 or abs(b[1]) > 1:
            print("Not in range")
            return

        # get point on circle
        b_magn = MobileNet.magnitude(b)
        point_on_circle = b / b_magn

        # get point on square
        b1_sign = -1 if b[0] < 0 else 1
        b2_sign = -1 if b[1] < 0 else 1
        if abs(b[0]) > abs(b[1]):
            point_on_square = (1., abs(b[1]/b[0]))
        else:
            point_on_square = (abs(b[0]/b[1]), 1.)
        point_on_square = (b1_sign*point_on_square[0], b2_sign*point_on_square[1])

        # get a fraction of lenght
        point_on_square_magn = MobileNet.magnitude(point_on_square)
        mult = b_magn / point_on_square_magn

        # calculate new point
        a = mult * point_on_circle

        return a
    
    def interpret_output(self, image, output):
        valence = (output[1][0][0] - 0.5) * 2
        arousal = (output[1][0][1] - 0.5) * 2
        if self.square2circ:
            valence, arousal = self.square2circumplex([valence, arousal])
            
        predictions = [output[0][0], [valence, arousal]]
        show_dashboard(image, predictions, target=None, save_dashboard_fp=None)
        
        # DISPLAY PREDICTIONS WITH ONLY TEXT OUTPUT - OLD
        # -----------------------------------------------
        # for i, (em, pred_val) in enumerate(sorted(zip(MobileNet.emotion_labels_list, output[0][0]), key=lambda x: x[1], reverse=True)):
        #     print(f"{em.upper() if i == 0 else em}: {pred_val:.3f}")
        
        # valence = (output[1][0][0] - 0.5) * 2
        # arousal = (output[1][0][1] - 0.5) * 2
        # print(f"VALENCE/AROUSAL: {valence:.3f} / {arousal:.3f}")

    def __call__(self, image: Union[str | np.ndarray]):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.preprocess(image)
        elif isinstance(image, np.ndarray):
            image = self.preprocess(image)
        else:
            raise ValueError("Unsupported input type")

        output = self.run(image)

        if self.debug:
            # OLD - WHEN NOT USING DASHBOARD
            # ------------------------------
            # try:
            #     display(Image.fromarray(image))
            # except:
            #     "__Could not display image__"
            self.interpret_output(image, output)

        return output
    
    
def infer_from_folder(folder: str, func):
    for i, im_name in enumerate(os.listdir(folder)):
        if not is_image(im_name):
            print("skipping file", str(os.path.join(folder, im_name)))
            continue

        print('='*30)
        print('FILE', im_name)
        out = func(os.path.join(folder, im_name))

class FFE_CycleGAN:
    # resize strategies
    class ResizeStrategy(Enum):
        STRETCH = "A"
        FILL_BLACK = "B"
    
    def __init__(
        self,
        model_fp,
        resize_strategy=ResizeStrategy.FILL_BLACK,
        output_as_gray=False,
        input_as_avg_grayscale=False,
        verbose=False,
        debug=False,
    ):
        self.resize_strategy = resize_strategy
        self.verbose = verbose
        self.model_fp = model_fp
        self.debug = debug
        self.input_as_avg_grayscale = input_as_avg_grayscale
        self.output_as_gray = output_as_gray
        
        self.model_input_shape = None
        self.model = None
        self.in_names = None
        
        self.load_model()
        
    def load_model(self):
        self.model = ort.InferenceSession(self.model_fp)
        self.model_input_shape = self.model.get_inputs()[0].shape
        model = onnx.load(str(self.model_fp))
        self.in_names = [_input.name for _input in model.graph.input]
        
        if self.verbose:
            print(f"FFE-CycleGAN model from '{self.model_fp}' loaded.")
    
    def preprocess(self, image):
        if self.input_as_avg_grayscale:
            grayscale = np.mean(image, axis=-1)
            image = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.uint8)

        if self.resize_strategy == FFE_CycleGAN.ResizeStrategy.FILL_BLACK:
            resized, _, _, _, _ = extract_square_roi(image, 0, 0, image.shape[1], image.shape[0])
            resized = cv2.resize(resized, self.model_input_shape[-2:], interpolation=cv2.INTER_AREA)
        elif self.resize_strategy == FFE_CycleGAN.ResizeStrategy.STRETCH:
            resized = cv2.resize(image, self.model_input_shape[-2:], interpolation=cv2.INTER_AREA)
        else:
            raise ValueError("Invalid strategy enum")

        if self.verbose:
            print(f"image resized from {image.shape} to {resized.shape}")
        
        t = transforms.Compose([
              transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          ])
        image = ((resized / 255) )
        return np.expand_dims(t(image).numpy(), axis=0).astype(np.float32)
    
    def run(self, _input):
        output = self.model.run(
            None,
            {self.in_names[0]: _input},
        )
        return output
        
    def postprocess(self, output: np.array, original_shape):
        original_shape = (original_shape[1], original_shape[0])
        
        image = np.swapaxes(np.swapaxes(np.array(output[0])[0], 0, 2), 0, 1)
        image = ((image + 1) / 2 * 255).astype(np.uint8)   
        output = image
        
        # remove the stripes
        output= cv2.resize(output, (max(original_shape), max(original_shape)), interpolation=cv2.INTER_AREA)        
        if self.resize_strategy == Orig_CycleGAN.ResizeStrategy.FILL_BLACK:
            x_padding = (max(original_shape) - original_shape[1]) // 2
            y_padding = (max(original_shape) - original_shape[0]) // 2
            output = output[y_padding:original_shape[0]+y_padding, x_padding:original_shape[1]+x_padding]

        if self.output_as_gray:
            grayscale = np.mean(output, axis=-1)
            output = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.uint8)
            
        return output

    def __call__(self, orig_image: Union[str | np.ndarray | Path]):
        if isinstance(orig_image, str) or isinstance(orig_image, Path):
            orig_image = cv2.imread(str(orig_image))
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        elif isinstance(orig_image, np.ndarray):
            pass
        else:
            raise ValueError("Unsupported input type")
            
        orig_image_copy = np.copy(orig_image)
        orig_image_shape = orig_image.shape[:2]
        image = self.preprocess(orig_image)
        output = self.run(image)
        predicted_image = self.postprocess(output, orig_image_shape)

        if self.debug:
            try:
#                 display(Image.fromarray(orig_image))
#                 display(Image.fromarray(predicted_image))
                if self.input_as_avg_grayscale:
                    grayscale = np.mean(orig_image, axis=-1)
                    gray_image = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.uint8)
                    display(Image.fromarray(np.concatenate([orig_image, gray_image, predicted_image], axis=1)))
                else:
                    display(Image.fromarray(np.concatenate([orig_image, predicted_image], axis=1)))
            except:
                print("__Could not display image__")

        return predicted_image, orig_image #np.concatenate([orig_image, predicted_image], axis=1)

class Orig_CycleGAN:
    # resize strategies
    class ResizeStrategy(Enum):
        STRETCH = "A"
        FILL_BLACK = "B"
    
    def __init__(
        self,
        model_fp,
        resize_strategy=ResizeStrategy.FILL_BLACK,
        output_as_gray=False,
        input_as_avg_grayscale=False,
        verbose=False,
        debug=False,
    ):
        self.resize_strategy = resize_strategy
        self.verbose = verbose
        self.model_fp = model_fp
        self.debug = debug
        self.output_as_gray = output_as_gray
        self.input_as_avg_grayscale = input_as_avg_grayscale
        
        self.model_input_shape = None
        self.model = None
        self.in_names = None
        
        self.load_model()
        
    def load_model(self):
        self.model = ort.InferenceSession(self.model_fp)
        self.model_input_shape = self.model.get_inputs()[0].shape
        model = onnx.load(str(self.model_fp))
        self.in_names = [_input.name for _input in model.graph.input]
        
        if self.verbose:
            print(f"Orig-CycleGAN model from '{self.model_fp}' loaded.")
    
    def preprocess(self, image):
        if self.input_as_avg_grayscale:
            grayscale = np.mean(image, axis=-1)
            image = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.uint8)
        
        # resize or/and fill stripes
        if self.resize_strategy == Orig_CycleGAN.ResizeStrategy.FILL_BLACK:
            resized, _, _, _, _ = extract_square_roi(image, 0, 0, image.shape[1], image.shape[0])
            resized = cv2.resize(resized, self.model_input_shape[-2:], interpolation=cv2.INTER_AREA)
        elif self.resize_strategy == Orig_CycleGAN.ResizeStrategy.STRETCH:
            resized = cv2.resize(image, self.model_input_shape[-2:], interpolation=cv2.INTER_AREA)
        else:
            raise ValueError("Invalid strategy enum")

        if self.verbose:
            print(f"image resized from {image.shape} to {resized.shape}")
        
        t = transforms.Compose([
              transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          ])
        image = ((resized / 255) )
        return t(image).numpy().astype(np.float32)
    
    def run(self, _input):
        output = self.model.run(
            None,
            {self.in_names[0]: _input},
        )
        return output

    def postprocess(self, output: np.array, original_shape):
        image = np.swapaxes(np.array(output[0]), 0, 2)
        image = ((image + 1) / 2 * 255).astype(np.uint8)
        output = cv2.rotate(image[::-1,:,:], cv2.ROTATE_90_CLOCKWISE)        
        
        # remove the stripes
        output= cv2.resize(output, (max(original_shape), max(original_shape)), interpolation=cv2.INTER_AREA)        
        if self.resize_strategy == Orig_CycleGAN.ResizeStrategy.FILL_BLACK:
            x_padding = (max(original_shape) - original_shape[1]) // 2
            y_padding = (max(original_shape) - original_shape[0]) // 2
            output = output[y_padding:original_shape[0]+y_padding, x_padding:original_shape[1]+x_padding]

        
        if self.output_as_gray:
            grayscale = np.mean(output, axis=-1)
            output = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.uint8)

        return output

    def __call__(self, orig_image: Union[str | np.ndarray]):
        if isinstance(orig_image, str) or isinstance(orig_image, Path):
            orig_image = cv2.imread(str(orig_image))
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        elif isinstance(orig_image, np.ndarray):
            pass
        else:
            raise ValueError("Unsupported input type")
            
        orig_image_shape = orig_image.shape[:2]
        image = self.preprocess(orig_image)
        output = self.run(image)
        predicted_image = self.postprocess(output, orig_image_shape)
        
        if self.debug:
            try:
                display(Image.fromarray(np.concatenate([orig_image, predicted_image], axis=1)))
            except:
                "__Could not display image__"

        return predicted_image, orig_image
###############################
# CenterFace Facial Detector - slightly adjusted
# Author's GitHub: https://github.com/Star-Clouds/CenterFace
###############################

class CenterFace(object):
    def __init__(self, landmarks=True):
        self.input_shape = (224, 224)
        self.landmarks = landmarks
        if self.landmarks:
            self.net = cv2.dnn.readNetFromONNX("models/pretrained/centerface.onnx")
        else:
            self.net = cv2.dnn.readNetFromONNX("models/pretrained/cface.1k.onnx")
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

    def __call__(self, img, threshold=0.5):
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            img = img
        else:
            raise ValueError("Unsupported input type")

        height = img.shape[0]
        width = img.shape[1]

        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(
            height, width
        )
        return self.inference_opencv(img, threshold)

    def inference_opencv(self, img, threshold):
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0,
            size=(self.img_w_new, self.img_h_new),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        begin = datetime.datetime.now()
        if self.landmarks:
            heatmap, scale, offset, lms = self.net.forward(["537", "538", "539", "540"])
        else:
            heatmap, scale, offset = self.net.forward(["535", "536", "537"])
        end = datetime.datetime.now()
        print("cpu times = ", end - begin)
        return self.postprocess(heatmap, lms, offset, scale, threshold)

    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        if self.landmarks:
            dets, lms = self.decode(
                heatmap,
                scale,
                offset,
                lms,
                (self.img_h_new, self.img_w_new),
                threshold=threshold,
            )
        else:
            dets = self.decode(
                heatmap,
                scale,
                offset,
                None,
                (self.img_h_new, self.img_w_new),
                threshold=threshold,
            )
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = (
                dets[:, 0:4:2] / self.scale_w,
                dets[:, 1:4:2] / self.scale_h,
            )
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = (
                    lms[:, 0:10:2] / self.scale_w,
                    lms[:, 1:10:2] / self.scale_h,
                )
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        if self.landmarks:
            return dets, lms
        else:
            return dets

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if self.landmarks:
            boxes, lms = [], []
        else:
            boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = (
                    np.exp(scale0[c0[i], c1[i]]) * 4,
                    np.exp(scale1[c0[i], c1[i]]) * 4,
                )
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(
                    0, (c0[i] + o0 + 0.5) * 4 - s0 / 2
                )
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
        if self.landmarks:
            return boxes, lms
        else:
            return boxes

    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep        

class Inference:
    # net types
    class net_type(Enum):
        SPECTRUM_TRANSLATOR_FFE_CYCLEGAN = "A"
        SPECTRUM_TRANSLATOR_ORIG_CYCLEGAN = "B"
#         SPECTRUM_TRANSLATOR_DENSEUNET = "C"
#         SPECTRUM_TRANSLATOR_CUSTOM_CYCLEGAN = "D"
        FER_MOBILENET = "E"
        FACE_DETECTOR_CENTERFACE = "F"
        FACE_DETECTOR_RETINAFACE = "G"

    @staticmethod
    def check_models_structure(models):
        assert set(models.keys()) == set(["face_detector", "spectrum_translator", "fer"]), \
            'dict must contain exactly: "face_detector", "spectrum_translator", "fer"'
        assert models["face_detector"]["net_type"] in \
            [Inference.net_type.FACE_DETECTOR_CENTERFACE, Inference.net_type.FACE_DETECTOR_RETINAFACE] or models["face_detector"]["net_type"] is None, \
            "face_detector's net_type must be in [net_type.FACE_DETECTOR_CENTERFACE, net_type.FACE_DETECTOR_RETINAFACE] or None"
        assert models["spectrum_translator"]["net_type"] in \
            [Inference.net_type.SPECTRUM_TRANSLATOR_FFE_CYCLEGAN, Inference.net_type.SPECTRUM_TRANSLATOR_ORIG_CYCLEGAN] or models["spectrum_translator"]["net_type"] is None, \
            "spectrum_translator's net_type must be in [net_type.SPECTRUM_TRANSLATOR_FFE_CYCLEGAN, net_type.SPECTRUM_TRANSLATOR_ORIG_CYCLEGAN] or None"
        assert models["fer"]["net_type"] in \
            [Inference.net_type.FER_MOBILENET] or models["fer"]["net_type"] is None, \
            "fer's net_type must be in [net_type.FER_MOBILENET] or None"
        
    def __init__(
        self,
        models: dict(),
        face_detector_output_shape: Optional[tuple],
        debug=False,
        verbose=False,
    ):
        self.check_models_structure(models)
        
        self.models = models
        self.debug = debug
        self.verbose = verbose
        
        # models
        self.spectrum_transfer_model = None
        self.fer_model = None
        self.face_detector_model = None
        
        self.num_detect_faces_run = 0

        # set models up
        if models["face_detector"]["net_type"]:
            self._setup_face_detector_model(models["face_detector"])
        if models["spectrum_translator"]["net_type"]:
            self._setup_spectrum_transfer_model(models["spectrum_translator"])
        if models["fer"]["net_type"]:
            self._setup_fer_model(models["fer"])
    
    def _setup_face_detector_model(self, model: dict()):
        print(f"Using '{model}' as face detector model")
        if model["net_type"] == Inference.net_type.FACE_DETECTOR_RETINAFACE:
            self.face_detector_model = 'retinaface'
        elif model["net_type"] == Inference.net_type.FACE_DETECTOR_CENTERFACE:
            self.face_detector_model = CenterFace()
        else:
            raise NotImplementedError

    def _setup_spectrum_transfer_model(self, model: dict()):
        print(f"Using '{model}' as spectrum transfer model")
        if model['net_type'] == Inference.net_type.SPECTRUM_TRANSLATOR_FFE_CYCLEGAN:
            self.spectrum_transfer_model = FFE_CycleGAN(model["pth_to_onnx"], \
                                                        input_as_avg_grayscale=self.models["spectrum_translator"]["input_as_avg_grayscale"], \
                                                        output_as_gray=self.models["spectrum_translator"]["output_as_avg_grayscale"], \
                                                        verbose=self.verbose, debug=self.debug)
        elif model['net_type'] == Inference.net_type.SPECTRUM_TRANSLATOR_ORIG_CYCLEGAN:    
            self.spectrum_transfer_model = Orig_CycleGAN(model["pth_to_onnx"], \
                                                         input_as_avg_grayscale=self.models["spectrum_translator"]["input_as_avg_grayscale"], \
                                                         output_as_gray=self.models["spectrum_translator"]["output_as_avg_grayscale"], \
                                                         verbose=self.verbose, debug=self.debug)
        else:
            raise NotImplementedError
            
    def _setup_fer_model(self, model: dict()):
        print(f"Using '{model}' as FER model")
        if model["net_type"] == Inference.net_type.FER_MOBILENET:
            self.fer_model = MobileNet(model["pth_to_onnx"], verbose=self.debug, debug=self.debug, resize_strategy=MobileNet.ResizeStrategy.FILL_BLACK, square2circ=model["va_to_circumplex_model"])
        else:
            raise NotImplementedError

    def detect_faces(
        self, image_fp: Optional[pathlib.Path | str], image_np_arr: Optional[np.array]
    ):
        assert not ((image_np_arr is None and image_np_arr is not None) or (
            image_np_arr is not None and image_np_arr is None
        )), "just one of 'image_fp' or 'image_np_arr' needs to be specified"

        image_fp = pathlib.Path(image_fp) if image_fp is not None else None
        try:
            if self.models["face_detector"]["net_type"] == Inference.net_type.FACE_DETECTOR_RETINAFACE:                
                face_objs = DeepFace.extract_faces(
                    img_path=str(image_fp) if image_np_arr is None else image_np_arr,
                    target_size=(224,224),
                    detector_backend="retinaface",
                    enforce_detection=False,
                    align=False
                )
                
                # adjust the output
                for f in face_objs:
                    f['face'] = (f['face'] * 255).astype(np.uint8)
                    
                # remove blackstripes
                if self.models["face_detector"]["remove_black_stripes"]:
                    # load if not loaded
                    if image_np_arr is None:
                        image_np_arr = cv2.imread(str(image_fp))
                        image_np_arr = cv2.cvtColor(image_np_arr, cv2.COLOR_BGR2RGB)

                    # expects 3 channels
                    if image_np_arr.shape[2] == 1:
                        image_np_arr = np.concatenate([image_np_arr] * 3, axis=-1)
                    face_objs = enlarge_detected_face(image_np_arr, face_objs, (224, 224))
                
                output = face_objs

            elif self.models["face_detector"]["net_type"] == Inference.net_type.FACE_DETECTOR_CENTERFACE:
                # load if not loaded
                if image_np_arr is None:
                    image_np_arr = cv2.imread(str(image_fp))
                    image_np_arr = cv2.cvtColor(image_np_arr, cv2.COLOR_BGR2RGB)

                # expects 3 channels
                if image_np_arr.shape[2] == 1:
                    image_np_arr = np.concatenate([image_np_arr] * 3, axis=-1)

                # predict
                dets, lms = self.face_detector_model(image_np_arr, threshold=0.35)
                
                # reformat output
                output = []
                for det in dets:
                    output.append({
                        'face': image_np_arr[int(det[1]):int(det[3]), int(det[0]):int(det[2]), :],
                        'facial_area': {'x': det[0], 'y': det[1], 'w': det[2] - det[0], 'h': det[3] - det[1]},
                        'confidence': det[4]
                    })
            else:
                raise NotImplementedError
            
            # remove blackstripes
            if self.models["face_detector"]["remove_black_stripes"]:
                output = enlarge_detected_face(image_np_arr, output, (224, 224))
            
            # display image
            if self.models["face_detector"]["display_images"]:
                display(Image.fromarray(output[0]["face"]))
                
            # save images
            if self.models['face_detector']['save_image_to_folder']:
                os.makedirs(self.models['face_detector']['save_image_to_folder'],exist_ok=True)
                im_arr = output[0]['face']
                if image_fp:
                    fn = Path(image_fp).name
                else:
                    fn = str(self.num_detect_faces_run) + '.jpg'
                Image.fromarray(im_arr).save(os.path.join(self.models['face_detector']['save_image_to_folder'], fn))
            
            self.num_detect_faces_run += 1
            
            return output
            
        except Exception as e:
                print(f"ERROR at image {image_fp if image_fp is not None else ''}", e)
                return None
            
    def translate_spectra(
        self,
        image_fp: Optional[pathlib.Path],
        image_np_arr: Optional[np.array],
    ):        
        if self.models["spectrum_translator"]["net_type"] == Inference.net_type.SPECTRUM_TRANSLATOR_ORIG_CYCLEGAN:
            output = self.spectrum_transfer_model(image_fp if image_np_arr is None else image_np_arr)
        elif self.models["spectrum_translator"]["net_type"] == Inference.net_type.SPECTRUM_TRANSLATOR_FFE_CYCLEGAN:
            output = self.spectrum_transfer_model(image_fp if image_np_arr is None else image_np_arr)
        else:
            raise NotImplementedError

        return output[0]
    
    def fer(self, image_fp: [pathlib.Path | str], image_np_arr: Optional[np.array]):
        if image_fp is not None:
            image_np_arr = cv2.imread(str(image_fp))
            image_np_arr = cv2.cvtColor(image_np_arr, cv2.COLOR_BGR2RGB)

        # predict
        output = self.fer_model(image_np_arr)
                
        return output
    
    # ==============================================================================
    def _from_array(self, images, func):
        detected = []
        for i, image in enumerate(images):
            if self.verbose:
                print(f"#{i} processing")
            if isinstance(image, list):
                detected.append(self._from_array(image, func))
            else:
                detected.append(func(None, image))
        return detected
    
    def _from_folder(self, folder: Union[str|pathlib.Path], func):
        # convert to Path
        folder = pathlib.Path(folder)
        detected = []
        i = 0
        
        for root, dirs, imgs in sorted(os.walk(str(folder)), key=lambda x: x[0]):
            for img in sorted(imgs):                    
                if img.split('.')[-1] not in ['png', 'bmp', 'jpeg', 'jpg']:
                    print(f'skipping {str(pathlib.Path(root) / img)} - not "png", "jpg", "jpeg" or "bmp"')
                    continue
                print(str(pathlib.Path(root) / img))
                detected.append(func(pathlib.Path(root) / img, None))
                
                if i%10 == 0:
                    gc.collect()
                
                i+=1

        return detected
                
    def _from_filenames(self, images: np.array, func):
        detected = []
        for i, image in enumerate(images):
            if self.verbose:
                print(f"#{i} - processing: {image}")
            detected.append(func(image, None))
        return detected
    
    def _infer(self, content, func):
        images_array = None
        face_detector_out = None
        spectrum_translator_out = None
        fer_out = None
        
        if self.face_detector_model is not None:
            face_detector_out = func(content, self.detect_faces)
            images_array = [[g['face'] for g in f] for f in face_detector_out]
        if self.spectrum_transfer_model is not None:
            if images_array is None:
                spectrum_translator_out = func(content, self.translate_spectra)
            else:
                spectrum_translator_out = self._from_array(images_array, self.translate_spectra)
            images_array = spectrum_translator_out
        if self.fer_model is not None:
            if images_array is None:
                fer_out = func(content, self.fer)
            else:
                fer_out = self._from_array(images_array, self.fer)
        
        return face_detector_out, spectrum_translator_out, fer_out
    
    def _infer_instant(self, content, func, save_to_folder=None):
        if save_to_folder:
            os.makedirs(save_to_folder, exist_ok=True)
        
        def _stacked_fns(_x1, _x2):
            x = _x1 if _x2 is None else _x2
            in1 = None if func == self._from_array else x
            in2 = x if func == self._from_array else None
            
            if self.face_detector_model is not None:
                x = [f['face'] for f in self.detect_faces(in1, in2)]
            if self.spectrum_transfer_model is not None:
                if self.face_detector_model is None:
                    x = [self.translate_spectra(in1, in2)]
                else:
                    x = self._from_array(x, self.translate_spectra)
                
                if save_to_folder:
                    idx = len(glob.glob(os.path.join(save_to_folder, "*.png")))
                    for i, _x in enumerate(x):
#                         Image.fromarray(_x).save(os.path.join(save_to_folder, f"{idx}_{i}.png"))
                        ###### TODO ########### AFTER USAGE GET BACK TO COMMENTED CODE
                        _x, _, _, _, _ = extract_square_roi(_x, 0, 0, _x.shape[1], _x.shape[0])
                        _x = cv2.resize(_x, (224, 224), interpolation=cv2.INTER_AREA)
                        Image.fromarray(_x).save(os.path.join(save_to_folder, str(_x1).split('/')[-1]))
                
            if self.fer_model is not None:
                if self.spectrum_transfer_model is None and self.face_detector_model is None:
                    x = self.fer(in1, in2)
                else:
                    x = self._from_array(x, self.fer)
            
            return x
        
        output = func(content, _stacked_fns)
        
        return output
    
    # ==============================================================================
    # =========================== CALLABLE FROM OUTSIDE ============================
    # ==============================================================================

    def infer_from_folder(self, folder: Union[str|pathlib.Path]):
        return self._infer(folder, self._from_folder)
        
    def infer_from_array(self, images: np.array):
        return self._infer(images, self._from_array)
    
    def infer_from_filenames(self, images: np.array):
        return self._infer(images, self._from_filenames)
    
    def infer_instant_from_folder(self, folder: Union[str|pathlib.Path], save_to_folder: Optional[str]=None):
        return self._infer_instant(folder, self._from_folder, save_to_folder)
    
    def infer_instant_from_array(self, images: np.array, save_to_folder: Optional[str]=None):
        return self._infer_instant(images, self._from_array, save_to_folder)
    
    def infer_instant_from_filenames(self, images: np.array, save_to_folder: Optional[str]=None):
        return self._infer_instant(images, self._from_filenames, save_to_folder)
        
    # -------------------------------------------------------------------------------

    @staticmethod
    def save_images_to_folder(images, folder: Union[str|pathlib.Path], img_format="png") -> None:
        # convert to Path and create
        folder = pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            cv2.imwrite(str(folder / f"{i}.{img_format}"), img)
            
    # ===============================================================================

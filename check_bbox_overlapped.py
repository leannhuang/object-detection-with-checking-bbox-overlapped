from ast import arguments
from asyncio.constants import SSL_HANDSHAKE_TIMEOUT
import cv2
import os
import numpy as np
import collections
import sys
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

class Bag:
    def __init__(self, bbox, items):
        self.bbox = bbox
        self.items = items

def assign_item():
    bag_left_item = ('Filet-O-Fish', 'Soda', 'French Fries', 'McChicken', 'Hot \'n Spicy McChicken')
    bag_right_item = ('Soda', 'French Fries', 'Double Cheeseburger', 'McNuggets')

    bag_left = Bag((0, 0, 0, 0), bag_left_item)
    bag_right = Bag((0, 0, 0, 0), bag_right_item)

    return bag_left, bag_right

def main():
    # define your object here 
    #object_list = ['Filet-O-Fish', 'Soda', 'French Fries', 'McNuggets']
    object_set = ('Filet-O-Fish', 'Soda', 'French Fries', 'McNuggets', 'Double Cheeseburger', 'Hot \'n Spicy McChicken', 'McChicken')

    # arguments
    argument_list = sys.argv
    print(argument_list)
    if len(argument_list) < 2:
        print('You did not assgin the probabily in the command argument. Set the probabilty to default value 50')
        prob_threshold = 0.5   
    else:
        prob_threshold = argument_list[1]
    
    # custom vision credentials information
    prediction_key = '<Your predition key>'
    ENDPOINT = 'https://mcdonalds-prediction.cognitiveservices.azure.com/'
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(endpoint=ENDPOINT, credentials=credentials)
    project_id = '<Your project id>'
    PUBLISH_ITERATION_NAME = '<Your project iteration>'
    
    # video path information
    video_path = './'
    video_name = '<Your video name>'
    parent_path = './'
    extract_img_folder = '_extracted_images'

    bag_left, bag_right = assign_item()
    frame_count = extract_frames_from_video(video_path, video_name, parent_path, extract_img_folder)
    ai_inference(frame_count, bag_left, bag_right, extract_img_folder, predictor, project_id, PUBLISH_ITERATION_NAME, prob_threshold, object_set, parent_path)
    compose_video(frame_count, PUBLISH_ITERATION_NAME, prob_threshold, project_id)


def extract_frames_from_video(video_path, video_name, parent_path, extract_img_folder):
    vidcap = cv2.VideoCapture(video_path + video_name)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    if success == 'False':
        print('1. Read the video FAILED. Check if your video file exists')
    else:
        print(f'1. Read the video SUCESSFULLY. The fps of the video is {fps}')
  
    img_path = parent_path + extract_img_folder
    os.mkdir(img_path)
    frame_count = 0
    
    while success:
        
        cv2.imwrite(os.path.join(img_path , f'frame_{frame_count}.jpg'), image)     
        success,image = vidcap.read()
        frame_count += 1
    
    print('2. Finish extracting the video to frames')
    return frame_count


def get_rectangle_start_end(pred, shape):

    x_start = int(pred.bounding_box.left * shape[1])
    y_start = int(pred.bounding_box.top * shape[0])

    x_end = x_start + int(pred.bounding_box.width * shape[1])
    y_end = y_start + int(pred.bounding_box.height * shape[0])

    return x_start, y_start, x_end, y_end

def get_bag_dict(results, prob, bag_left, bag_right, shape):
    bag_preds = [prediction for prediction in results.predictions if prediction.probability > prob and prediction.tag_name == 'Bag']
    bag_preds_rect = [get_rectangle_start_end(pred, shape) for pred in bag_preds]
    bag_preds_rect = sorted(bag_preds_rect)

    bag_left.bbox = bag_preds_rect[0]
    #print(f'bag_left: {bag_left.bbox}')
    bag_right.bbox = bag_preds_rect[1]
    #print(f'bag_right: {bag_right.bbox}')

    return bag_left, bag_right

def overlapped(bag_bbox, item_bbox):
    x_bag_start, y_bag_start, x_bag_end, y_bag_end = bag_bbox
    start_point, end_point = item_bbox
    x_item_start, y_item_start = start_point
    x_item_end, y_item_end = end_point 
    
    if max(x_bag_start, x_item_start) < min(x_bag_end, x_item_end) and max(y_bag_start, y_item_start) < min(y_bag_end, y_item_end):
        return True
    
    return False

def correct_item(bag_left, bag_right, item_name, start_point, end_point):
    item_bbox = (start_point, end_point)
    #print(item_bbox)
    bag_left_overlapped = overlapped(bag_left.bbox, item_bbox)
    bag_right_overlapped = overlapped(bag_right.bbox, item_bbox)
    if not bag_left_overlapped and not bag_right_overlapped:
        return (True, bag_left)

    if bag_left_overlapped:
        if item_name in bag_left.items:
            return (True, bag_left)
        else:
            return (False, bag_left)

    if bag_right_overlapped: 
        if item_name in bag_right.items:
            return (True, bag_right)
        else:
            return (False, bag_right)

def ai_inference(frame_count, bag_left, bag_right, extract_img_folder, predictor, project_id, PUBLISH_ITERATION_NAME, prob_threshold, object_set, parent_path):

    # Open the sample image and get back the prediction results.
    with open(os.path.join(extract_img_folder, "frame_0.jpg"), mode="rb") as test_data:
        results = predictor.detect_image(project_id, PUBLISH_ITERATION_NAME, test_data)
        print(results)
        print('3. Call the Custom Vision SUCESSFULLY')
    
    prob = float(prob_threshold)
    
    tagged_folder = f'{project_id}_tagged_images'
    path = os.path.join(parent_path, tagged_folder)
    os.mkdir(path)

    for i in range(frame_count):
        
        with open(os.path.join(extract_img_folder, "frame_%d.jpg" % i), mode="rb") as test_data:
            results = predictor.detect_image(project_id, PUBLISH_ITERATION_NAME, test_data)
            img = cv2.imread(f'{extract_img_folder}/frame_%d.jpg' % i)
            shape = img.shape

            bag_left, bag_right = get_bag_dict(results, prob, bag_left, bag_right, shape)
            filtered_preds = [prediction for prediction in results.predictions if prediction.probability > prob and prediction.tag_name in object_set] 

            for pred in filtered_preds:
                x_start, y_start, x_end, y_end = get_rectangle_start_end(pred, shape)

                start_point = (x_start, y_start)
                end_point = (x_end, y_end)

                item_bbox = (start_point, end_point)

                if not overlapped(bag_left.bbox, item_bbox) and not overlapped(bag_right.bbox, item_bbox):
                    img = cv2.rectangle(img, start_point, end_point, (0, 77, 255), 3)
                    img = cv2.putText(img, pred.tag_name, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 77, 255), 3)
                    continue

                if not correct_item(bag_left, bag_right, pred.tag_name, start_point, end_point)[0]:
                    bag = correct_item(bag_left, bag_right, pred.tag_name, start_point, end_point)[1]
                    x_bag_start, y_bag_start, x_bag_end, y_bag_end = bag.bbox
                    bag_start_point = (x_bag_start, y_bag_start)
                    bag_end_point = (x_bag_end, y_bag_end)
                    img = cv2.rectangle(img, bag_start_point, bag_end_point, (28, 41, 218), 3)
                    img = cv2.rectangle(img, start_point, end_point, (28, 41, 218), 3)
                    img = cv2.putText(img, pred.tag_name, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (28, 41, 218), 3)   
                
                else:
                    img = cv2.rectangle(img, start_point, end_point, (44, 190, 255), 3)
                    img = cv2.putText(img, pred.tag_name, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (44, 190, 255), 3)                           

            cv2.imwrite(f"{tagged_folder}/tagged{i:04d}.jpg", img)
    
    print('5. Finish inferencing the frames of the video')


def compose_video(frame_count, PUBLISH_ITERATION_NAME, prob_threshold, project_id):
    tagged_folder = f'{project_id}_tagged_images'
    video_name = f'tagged_{PUBLISH_ITERATION_NAME}_{prob_threshold}.avi'
    img = cv2.imread(f'{project_id}_tagged_images/tagged0000.jpg')
    shape = img.shape

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out_video = cv2.VideoWriter(video_name, fourcc, 25, (shape[1], shape[0]))

    for i in range(frame_count):
        file_name = f'{tagged_folder}/tagged{i:04d}.jpg'
        img = cv2.imread(file_name)
        out_video.write(img)
        
    out_video.release()

    print('6. Finish composing the video')
    print(f'7. Check the inferenced video - {video_name} under the ai-inerence-on-top-of-video folder')


def generate_color(object_set):
    object_list = list(object_set)
    object_count = len(object_list)
    object_to_color = collections.defaultdict()

    for i in range(object_count):
        color = np.random.choice(range(256), size=3)
        color = (int(color[0]), int(color[1]), int(color[2]))
        object = object_list[i]
        object_to_color[object] = color

    print('4. The colors of the objects are fully assigned')
    return object_to_color



if __name__ == "__main__":
    main()




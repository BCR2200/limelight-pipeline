import cv2
import numpy as np
import time

# global variables go here:
testVar = 0
last_frame_time = 0
fps = 0

# To change a global variable inside a function,
# re-declare it with the 'global' keyword
def incrementTestVar():
    global testVar
    testVar = testVar + 1
    if testVar == 100:
        print("test")
    if testVar >= 200:
        print("print")
        testVar = 0

def drawDecorations(image):
    cv2.putText(image, 
        'Limelight python script!', 
        (0, 700), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        2, (0, 255, 0), 3, cv2.LINE_AA)

def quarter_frame(img, lower, upper, roi_coords):
    start_y, end_y, start_x, end_x = roi_coords
    roi = img[start_y:end_y, start_x:end_x]

    img_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.GaussianBlur(img_hsv, (5,5), 0)

    mask = cv2.inRange(img_hsv, lower, upper)
    y_p_c = cv2.countNonZero(mask)

    return y_p_c

# runPipeline() is called every frame by Limelight's backend.
# takes in an image and some parameters from the robot (not used presently)
def runPipeline(image, llrobot):
    global last_frame_time, fps
    _ = llrobot
    
    current_time = time.time()
    if last_frame_time != 0:
        dt = current_time - last_frame_time
        if dt > 0:
            instant_fps = 1.0 / dt
            # Exponentially weighted moving average for smoothness
            fps = (0.9 * fps) + (0.1 * instant_fps)
    last_frame_time = current_time

    #print(f"{cv2.__version__}")
    llpython = [0,0,0,0,0,0,0,0]
    yellow_percentage = 0.0
    largestContour = np.array([[]])

    blob_params = cv2.SimpleBlobDetector_Params()

    blob_params.filterByColor = True
    blob_params.blobColor = 255
    blob_params.minThreshold = 10
    blob_params.maxThreshold = 800

    detector = cv2.SimpleBlobDetector_create(blob_params)

    height, width = image.shape[:2]
    total_pixels = height * width
    total_pixels_3rds = height * width / 6
    roi_coords1 = (int(height/2), int(height), int(0), int(width/3))
    roi_coords2 = (int(height/2), int(height), int(width/3), int(width*2/3))
    roi_coords3 = (int(height/2), int(height), int(width*2/3), int(width))

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.GaussianBlur(img_hsv, (5,5), 0)

    yellow_lower = (20, 100, 100)
    yellow_upper = (40, 255, 255)

    yellow_mask = cv2.inRange(img_hsv, yellow_lower, yellow_upper)
    yellow_pixel_count = cv2.countNonZero(yellow_mask)

    y_count_l = quarter_frame(image, yellow_lower, yellow_upper, roi_coords1)
    y_count_c = quarter_frame(image, yellow_lower, yellow_upper, roi_coords2)
    y_count_r = quarter_frame(image, yellow_lower, yellow_upper, roi_coords3)
    yellow_percentage = (yellow_pixel_count / total_pixels) * 100
    yellow_percentage_left = (y_count_l / total_pixels_3rds) * 100
    yellow_percentage_center = (y_count_c / total_pixels_3rds) * 100
    yellow_percentage_right = (y_count_r / total_pixels_3rds) * 100
    llpython[0] = round(yellow_percentage, 2)
    #llpython[1] = yellow_pixel_count
    #llpython[2] = total_pixels
    llpython[3] = round(yellow_percentage_left, 2)
    llpython[4] = round(yellow_percentage_center, 2)
    llpython[5] = round(yellow_percentage_right, 2)

    yellow_highlight = image.copy()
    yellow_highlight[yellow_mask > 0] = (0,255,255)
    #yellow_highlight[yellow_mask == 0] = (0,0,0)

    #key_points = detector.detect(yellow_highlight)

    #for kp in key_points:
        #x, y = kp.pt
        #size = kp.size
        #print(f"blob at {x}, {y}, {size}")
    
    #print(f"number of blobs: {len(key_points)}")
    #llpython[1] = len(key_points)
    
    output_image = cv2.addWeighted(image, 0.5, yellow_highlight, 0.5, 0)

    # TODO: These are fixed points on the image, dividing it into thirds (the bottom half of the image)
    cv2.line(output_image, (int(width/3), int(height/2)), (int(width/3), height), (0, 255, 0), 2)
    cv2.line(output_image, (int(width*2/3), int(height/2)), (int(width*2/3), height), (0, 255, 0), 2)

    #output_image = cv2.drawKeypoints(output_image, key_points, 
    #                None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    text = f"Yellow: {yellow_percentage:.2f}%"
    cv2.putText(output_image, text, (int(width*0.375), 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(output_image, text, (int(width*0.375), 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)

    text = f"left: {yellow_percentage_left:.2f}%"
    cv2.putText(output_image, text, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(output_image, text, (10,height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)

    text = f"center: {yellow_percentage_center:.2f}%"
    cv2.putText(output_image, text, (int(width/2 - 150), height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(output_image, text, (int(width/2 - 150), height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)

    text = f"right: {yellow_percentage_right:.2f}%"
    cv2.putText(output_image, text, (width - 200, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(output_image, text, (width - 200, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(output_image, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(output_image, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)

#    count_text = f"Pixels: {yellow_pixel_count} / {total_pixels}"
#    cv2.putText(output_image, count_text, (10, 360), 
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
#    cv2.putText(output_image, count_text, (10, 360),
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)

#    img_threshold = cv2.inRange(img_hsv, (24, 70, 70), (40, 255, 255))
    
   
#    contours, _ = cv2.findContours(img_threshold, 
#    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


#    largestContour = np.array([[]])
#    llpython = [0,0,0,0,0,0,0,0]

#    if len(contours) > 0:
#        cv2.drawContours(image, contours, -1, 255, 2)
#        largestContour = max(contours, key=cv2.contourArea)
#        x,y,w,h = cv2.boundingRect(largestContour)

#        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
#        llpython = [1,x,y,w,h,9,8,7]  
  
#    incrementTestVar()
#    drawDecorations(output_image)
       
    # make sure to return a contour,
    # an image to stream,
    # and optionally an array of up to 8 values for the "llpython"
    # networktables array
    return largestContour, output_image, llpython

if __name__ == "__main__":
    import argparse

    resolutions = {
        "320p": (320, 240),
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080)
    }

    parser = argparse.ArgumentParser(description='Limelight Pipeline Laptop Test')
    parser.add_argument('--res', type=str, default="480p", choices=resolutions.keys(),
                        help='Webcam resolution (default: 480p)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    width, height = resolutions[args.res]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Read back to verify
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Requested resolution: {width}x{height}")
    print(f"Actual resolution: {actual_width}x{actual_height}")

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # llrobot is not used in the current implementation of runPipeline
        # but we pass a dummy value to match the signature
        contour, processed_image, llpython = runPipeline(frame, None)

        cv2.imshow('Limelight Pipeline - Laptop Test', processed_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


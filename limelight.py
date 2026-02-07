import cv2
import numpy as np
import time


def count_in_roi(mask, roi_coords):
    start_y, end_y, start_x, end_x = roi_coords
    roi = mask[start_y:end_y, start_x:end_x]
    return cv2.countNonZero(roi)


yellow_lower = (17, 30, 130)
yellow_upper = (35, 240, 255)
height_percentage = 0.6 

# runPipeline() is called every frame by Limelight's backend.
# takes in an image and some parameters from the robot (not used presently)
def runPipeline(image, llrobot):
    _ = llrobot
    
    llpython = [0,0,0,0,0,0,0,0]

    # cropping image to look at only a certain percentage
    total_height, width = image.shape[:2]
    height = int(total_height*height_percentage)
    img_cropped = image[int(total_height-height):total_height, 0:width]
    
    
    # finding the percentage of yellow pixels in the image, and in each third of the image
    total_pixels = height * width
    total_pixels_3rds = height * width / 6
    left_roi = (0, height, int(0), int(width/3))
    middle_roi = (0, height, int(width/3), int(width*2/3))
    right_roi = (0, height, int(width*2/3), int(width))

    
    img_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(img_hsv, yellow_lower, yellow_upper)
    
    yellow_pixel_count = cv2.countNonZero(yellow_mask)

    y_count_l = count_in_roi(yellow_mask, left_roi)
    y_count_c = count_in_roi(yellow_mask, middle_roi)
    y_count_r = count_in_roi(yellow_mask, right_roi)
    yellow_percentage = (yellow_pixel_count / total_pixels) * 100
    yellow_percentage_left = (y_count_l / total_pixels_3rds) * 100
    yellow_percentage_center = (y_count_c / total_pixels_3rds) * 100
    yellow_percentage_right = (y_count_r / total_pixels_3rds) * 100
    llpython[0] = round(yellow_percentage, 2)
    llpython[3] = round(yellow_percentage_left, 2)
    llpython[4] = round(yellow_percentage_center, 2)
    llpython[5] = round(yellow_percentage_right, 2)

    cv2.line(image, (0, total_height-height), (width, total_height-height), (0, 255, 0), 2)

    # image = cv2.drawKeypoints(image, key_points,
    #                 None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # cv2.line(image, (int(width/3), int(height/2)), (int(width/3), height), (0, 255, 0), 2)
    # cv2.line(image, (int(width*2/3), int(height/2)), (int(width*2/3), height), (0, 255, 0), 2)

    # text = f"Yellow: {yellow_percentage:.2f}%"
    # cv2.putText(image, text, (int(width*0.375), 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    # cv2.putText(image, text, (int(width*0.375), 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)

    # text = f"left: {yellow_percentage_left:.2f}%"
    # cv2.putText(image, text, (10, height - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    # cv2.putText(image, text, (10,height - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)

    # text = f"center: {yellow_percentage_center:.2f}%"
    # cv2.putText(image, text, (int(width/2 - 150), height - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    # cv2.putText(image, text, (int(width/2 - 150), height - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)

    # text = f"right: {yellow_percentage_right:.2f}%"
    # cv2.putText(image, text, (width - 200, height - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    # cv2.putText(image, text, (width - 200, height - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)
    

    # for debugging, draw the HSV colours
    # cv2.rectangle(image, (width-100, 0), (width-50, 50), yellow_lower, -1)
    # cv2.rectangle(image, (width-50, 0), (width, 50), yellow_upper, -1)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cv2.drawContours(image, contours, -1, (255, 0, 0), 2, offset=(0, int(total_height-height)))
        largest_contour = max(contours, key=cv2.contourArea)+np.array([[[0, int(total_height-height)]]])
        x,y,w,h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        largest_contour = [x, y, w, h]
    else:
        largest_contour = [0, 0, 0, 0]

    # text = f"largest contour rect: {x},{y},{w},{h}"
    # cv2.putText(image, text, (10, 100),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    # cv2.putText(image, text, (10, 100),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3, cv2.LINE_AA)

    # make sure to return a contour,
    # an image to stream,
    # and optionally an array of up to 8 values for the "llpython"
    # networktables array
    return largest_contour, image, llpython

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

    last_frame_time = 0
    fps = 0
    pipeline_time_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # llrobot is not used in the current implementation of runPipeline
        # but we pass a dummy value to match the signature
        start_pipeline = time.perf_counter()
        _, processed_image, _ = runPipeline(frame, None)
        processed_image = cv2.flip(processed_image, 1)  # Flip horizontally
        end_pipeline = time.perf_counter()

        # Pipeline time in milliseconds
        duration_ms = (end_pipeline - start_pipeline) * 1_000
        # EWMA for pipeline time
        pipeline_time_ms = (0.9 * pipeline_time_ms) + (0.1 * duration_ms)

        current_time = time.time()
        if last_frame_time != 0:
            dt = current_time - last_frame_time
            if dt > 0:
                instant_fps = 1.0 / dt
                # Exponentially weighted moving average for smoothness
                fps = (0.9 * fps) + (0.1 * instant_fps)
        last_frame_time = current_time

        if fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(processed_image, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(processed_image, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)

        if pipeline_time_ms > 0:
            pipe_text = f"Pipe: {pipeline_time_ms:.2f} ms"
            cv2.putText(processed_image, pipe_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(processed_image, pipe_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)

        cv2.imshow('Limelight Pipeline - Laptop Test', processed_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


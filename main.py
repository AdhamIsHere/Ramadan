import cv2
import mediapipe as mp
import os
import numpy as np
import pygame
import threading
import time

# Initialize pygame mixer for sound
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Initialize MediaPipe hands for landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize base options and model for gesture recognition
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize Face Landmarker
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# Callback function (runs when gestures are detected)
current_gesture = "No gesture"
current_expression = "Neutral"
face_landmarks_for_drawing = None

def print_result(result, output_image, timestamp_ms):
    global current_gesture
    if result.gestures:
        gesture = result.gestures[0][0].category_name
        confidence = result.gestures[0][0].score
        print(f"Detected gesture: {gesture} (confidence: {confidence:.2f})")
        current_gesture = f"{gesture} ({confidence:.2f})"
    else:
        current_gesture = "No gesture"

def face_result_callback(result, output_image, timestamp_ms):
    global current_expression, face_landmarks_for_drawing
    if result.face_landmarks:
        # Analyze the first detected face
        face_landmarks = result.face_landmarks[0]
        current_expression = analyze_facial_expression(face_landmarks)
        # Store landmarks for drawing
        face_landmarks_for_drawing = face_landmarks
    else:
        current_expression = "No face detected"
        face_landmarks_for_drawing = None

def analyze_facial_expression(face_landmarks):
    """Analyze facial landmarks to detect expressions using task-based landmarks"""
    # Key landmark indices for expression analysis
    # Mouth landmarks
    mouth_left = face_landmarks[61]   # Left mouth corner
    mouth_right = face_landmarks[291] # Right mouth corner
    mouth_top = face_landmarks[13]    # Upper lip center
    mouth_bottom = face_landmarks[14] # Lower lip center
    
    # Eye landmarks
    left_eye_top = face_landmarks[159]
    left_eye_bottom = face_landmarks[145]
    right_eye_top = face_landmarks[386]
    right_eye_bottom = face_landmarks[374]
    
    # Calculate mouth curvature (smile/frown detection)
    mouth_center_y = (mouth_top.y + mouth_bottom.y) / 2
    mouth_curve = ((mouth_left.y + mouth_right.y) / 2) - mouth_center_y
    
    # Calculate mouth openness
    mouth_openness = abs(mouth_top.y - mouth_bottom.y)
    
    # Calculate eye openness
    left_eye_openness = abs(left_eye_top.y - left_eye_bottom.y)
    right_eye_openness = abs(right_eye_top.y - right_eye_bottom.y)
    avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
    
    # Expression classification
    if mouth_openness > 0.02:  # Mouth significantly open
        return "Surprised/Open"
    elif mouth_curve < -0.005:  # Mouth corners up (smiling)
        return "Happy/Smiling"
    elif mouth_curve > 0.005:   # Mouth corners down (frowning)
        return "Sad/Frowning"
    elif avg_eye_openness < 0.008:  # Eyes mostly closed
        return "Squinting/Tired"
    else:
        return "Neutral"

# Image mapping function
def get_reaction_image(gesture, expression):
    """
    Returns the appropriate reaction image based on current gesture and expression
    """
    # Clean up gesture and expression strings
    gesture_clean = gesture.split("(")[0].strip() if "(" in gesture else gesture.strip()
    expression_clean = expression.split("/")[0].strip()
    
    # Define image mappings for gesture + expression combinations
    image_mappings = {
        # Happy + Pointing = happy_point_up.png
        ("Pointing_Up", "Neutral"): "images/happy_point_up.png",
        
        # Sad face with no gesture = sad.jpg
        ("No gesture", "Sad"): "images/sad.jpg",
        (None, "Sad"): "images/sad.jpg",
        
        ("No gesture", "Happy"): "images/happy.png",
        ("Thumb_Up", "Neutral"): "images/thumbs_up.png",
        ("Open_Palm", "Neutral"): "images/open_palm.mp4",
      

        ("No gesture", "Neutral"): "images/neutral.png",
    }
    
    
    # Try to find exact match first
    key = (gesture_clean, expression_clean)
    if key in image_mappings:
        return image_mappings[key]
    
    # Try gesture-only match
    for (g, e), img_path in image_mappings.items():
        if g == gesture_clean:
            return img_path
    
    # Try expression-only match
    for (g, e), img_path in image_mappings.items():
        if e == expression_clean:
            return img_path
    
    # Return None if no match found
    return None

def get_reaction_sound(gesture, expression):
    """
    Returns the appropriate sound file based on current gesture and expression
    """
    # Clean up gesture and expression strings
    gesture_clean = gesture.split("(")[0].strip() if "(" in gesture else gesture.strip()
    expression_clean = expression.split("/")[0].strip()
    
    # Define sound mappings for gesture + expression combinations
    sound_mappings = {
        # Happy + Pointing = cheerful sound
        ("Pointing_Up", "Neutral"): "sound/happy_point_up.wav",
        
        # Sad face with no gesture = sad sound
        ("No gesture", "Sad"): "sound/sad.wav",
        (None, "Sad"): "sound/sad_music.mp3",
        
        ("No gesture", "Happy"): "sound/happy.wav",
        ("Thumb_Up", "Neutral"): "sound/thumb_up.wav",
        ("Open_Palm", "Neutral"): "sound/palm.wav",
        
        # Default fallbacks
        ("No gesture", "Neutral"): "sound/neutral_ambient.ogg",
    }
    
    # Try to find exact match first
    key = (gesture_clean, expression_clean)
    if key in sound_mappings:
        return sound_mappings[key]
    
    # Try gesture-only match
    for (g, e), sound_path in sound_mappings.items():
        if g == gesture_clean:
            return sound_path
    
    # Try expression-only match
    for (g, e), sound_path in sound_mappings.items():
        if e == expression_clean:
            return sound_path
    
    # Return None if no match found
    return None

# Global variables for sound management
current_sound_file = None
sound_thread = None

def play_sound_async(sound_path):
    """
    Play sound asynchronously without blocking the main thread
    Supports multiple audio formats
    """
    if not sound_path or not os.path.exists(sound_path):
        return
    
    try:
        # Get file extension to determine format
        file_ext = os.path.splitext(sound_path)[1].lower()
        
        # Supported audio formats
        supported_formats = ['.wav', '.mp3', '.ogg', '.aiff', '.flac']
        
        if file_ext not in supported_formats:
            print(f"Unsupported audio format: {file_ext}")
            print(f"Supported formats: {', '.join(supported_formats)}")
            return
        
        sound = pygame.mixer.Sound(sound_path)
        channel = sound.play(loops=-1)
        print(f"Playing sound: {os.path.basename(sound_path)}")
        
    except pygame.error as e:
        print(f"Pygame error playing sound {sound_path}: {e}")
    except Exception as e:
        print(f"Error playing sound {sound_path}: {e}")

def manage_sound_playback(sound_path):
    """
    Manage sound playback - stop current sound and play new one if different
    """
    global current_sound_file, sound_thread
    
    if sound_path == current_sound_file:
        return  # Same sound already playing
    
    # Stop current sound
    pygame.mixer.stop()
    current_sound_file = sound_path
    
    if sound_path and os.path.exists(sound_path):
        # Play new sound in background thread
        sound_thread = threading.Thread(target=play_sound_async, args=(sound_path,))
        sound_thread.daemon = True
        sound_thread.start()

# Global variables for video playback
current_video_cap = None
current_video_path = None

def display_reaction_image_window(media_path, window_name="Reaction"):
    """
    Display reaction image or video in a separate window
    """
    global current_video_cap, current_video_path
    
    if not media_path or not os.path.exists(media_path):
        # Show blank/default window if no media
        blank_img = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.putText(blank_img, "No Reaction", (150, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow(window_name, blank_img)
        return
    
    # Check if it's a video file
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    is_video = any(media_path.lower().endswith(ext) for ext in video_extensions)
    
    if is_video:
        # Handle video playback
        if current_video_path != media_path:
            # New video - close previous and open new one
            if current_video_cap is not None:
                current_video_cap.release()
            current_video_cap = cv2.VideoCapture(media_path)
            current_video_path = media_path
            # Increase video playback speed by skipping frames
            if current_video_cap is not None:
                fps = current_video_cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    # Set to play at 2x speed by adjusting frame rate
                    current_video_cap.set(cv2.CAP_PROP_FPS, fps * 2)
        
        if current_video_cap is not None and current_video_cap.isOpened():
            # Skip frames for faster playback (2x speed)
            ret, frame = current_video_cap.read()
            if ret:
                # Skip next frame for 2x speed
                current_video_cap.read()  # Skip one frame
            
            if not ret:
                # Video ended, restart from beginning
                current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = current_video_cap.read()
            
            if ret:
                # Resize video frame for bigger display
                height, width = frame.shape[:2]
                max_size = 600 
                
                if width > height:
                    new_width = max_size
                    new_height = int((max_size * height) / width)
                else:
                    new_height = max_size
                    new_width = int((max_size * width) / height)
                
                resized_frame = cv2.resize(frame, (new_width, new_height))
                
                # Create a bigger square canvas and center the video frame
                canvas = np.zeros((max_size, max_size, 3), dtype=np.uint8)
                y_offset = (max_size - new_height) // 2
                x_offset = (max_size - new_width) // 2
                canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
                
                cv2.imshow(window_name, canvas)
    else:
        # Handle static image
        # Close video if we're switching from video to image
        if current_video_cap is not None:
            current_video_cap.release()
            current_video_cap = None
            current_video_path = None
        
        try:
            img = cv2.imread(media_path)
            if img is None:
                return
            
            # Resize image for bigger display window
            height, width = img.shape[:2]
            max_size = 600  
            
            if width > height:
                new_width = max_size
                new_height = int((max_size * height) / width)
            else:
                new_height = max_size
                new_width = int((max_size * width) / height)
            
            resized_img = cv2.resize(img, (new_width, new_height))
            
            # Create a bigger square canvas and center the image
            canvas = np.zeros((max_size, max_size, 3), dtype=np.uint8)
            y_offset = (max_size - new_height) // 2
            x_offset = (max_size - new_width) // 2
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
            
            cv2.imshow(window_name, canvas)
            
        except Exception as e:
            print(f"Error displaying media {media_path}: {e}")
        
# Load the gesture recognizer model
gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)

# Load the face landmarker model
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=face_result_callback,
    num_faces=1
)

recognizer = GestureRecognizer.create_from_options(gesture_options)
face_landmarker = FaceLandmarker.create_from_options(face_options)

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize hands detection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


# recognizer.set_result_callback(print_result)

timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    hand_results = hands.process(rgb_frame)
    
    # Draw hand landmarks and connections
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Optionally draw landmark numbers
            for i, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Draw face mesh if landmarks are available
    if face_landmarks_for_drawing:
        h, w, _ = frame.shape
        
        # Draw face mesh points
        for i, landmark in enumerate(face_landmarks_for_drawing):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Draw key facial features with different colors
        # Eyes (draw circles around eyes)
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        for idx in left_eye_indices:
            landmark = face_landmarks_for_drawing[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)  # Blue for left eye
            
        for idx in right_eye_indices:
            landmark = face_landmarks_for_drawing[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)  # Blue for right eye
        
        # Mouth outline
        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        for i in range(len(mouth_indices)):
            start_idx = mouth_indices[i]
            end_idx = mouth_indices[(i + 1) % len(mouth_indices)]
            
            start_landmark = face_landmarks_for_drawing[start_idx]
            end_landmark = face_landmarks_for_drawing[end_idx]
            
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            
            cv2.line(frame, start_point, end_point, (0, 0, 255), 2)  # Red for mouth

    # Process gesture recognition
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    recognizer.recognize_async(mp_image, timestamp)
    
    # Process face landmark detection
    face_landmarker.detect_async(mp_image, timestamp)
    
    timestamp += 33  # roughly 30 FPS

    # Display gesture result on frame
    cv2.putText(frame, f"Gesture: {current_gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display facial expression result
    cv2.putText(frame, f"Expression: {current_expression}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display instructions
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Get and display reaction image/video in separate window
    reaction_media_path = get_reaction_image(current_gesture, current_expression)
    display_reaction_image_window(reaction_media_path, "Reaction Image")
    
    # Get and play reaction sound
    reaction_sound_path = get_reaction_sound(current_gesture, current_expression)
    manage_sound_playback(reaction_sound_path)
        
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
face_landmarker.close()
# Clean up video capture if it exists
if current_video_cap is not None:
    current_video_cap.release()
# Clean up sound
pygame.mixer.quit()
cv2.destroyAllWindows()

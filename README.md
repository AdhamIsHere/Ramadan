# OpenCV Gesture & Expression Detection 🤪

> **Warning**: This project was made as a joke to test OpenCV capabilities. Don't expect production-ready code! 😄

## What's This About? 🤔

A silly little experiment that uses your webcam to detect hand gestures and facial expressions, then plays corresponding images/videos and sounds. It's basically a real-time emotion and gesture recognition system that responds with multimedia reactions.

## Features ✨

- **Hand Gesture Recognition**: Detects gestures like thumbs up, pointing up, open palm, etc.
- **Facial Expression Analysis**: Recognizes happy, sad, surprised, neutral expressions
- **Real-time Reactions**: Shows reaction images/videos and plays sounds based on what it detects
- **Visual Feedback**: Draws hand landmarks and face mesh on the camera feed
- **Multimedia Support**: Handles both images (PNG, JPG) and videos (MP4) for reactions

## What You'll See 👀

When you run this, you'll get:

- A webcam window showing your face with detected landmarks
- A separate "Reaction" window that displays images/videos based on your gestures and expressions
- Sound effects that play in response to what you're doing
- Text overlays showing detected gestures and expressions

## Dependencies 📦

This project uses a bunch of Python packages (see `Pipfile`):

- `opencv-python` - For camera capture and image processing
- `mediapipe` - Google's ML framework for gesture and face detection
- `pygame` - For playing sound effects
- `numpy` - Because everything needs numpy

## Setup 🚀

1. Make sure you have Python 3.11 (or close enough)
2. Install pipenv if you don't have it: `pip install pipenv`
3. Install dependencies: `pipenv install`
4. Run it: `pipenv run python main.py`

## Required Files 📁

You'll need these MediaPipe model files (not included in repo):

- `gesture_recognizer.task`
- `face_landmarker.task`

Download them from [MediaPipe's website](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer) and place them in the project root.

## Assets Included 🎭

### Images

- `happy.png` - Happy reaction
- `sad.jpg` - Sad reaction
- `neutral.png` - Default expression
- `thumbs_up.png` - Thumbs up reaction
- `happy_point_up.png` - Happy pointing reaction
- `open_palm.mp4` - Video reaction for open palm

### Sounds

- Various `.wav` files for different gesture/expression combinations
- Background ambient sounds for neutral states

## How It Works (Kinda) 🔧

1. Captures video from your webcam
2. Uses MediaPipe to detect hand landmarks and facial features
3. Analyzes the landmarks to classify gestures and expressions
4. Maps gesture+expression combinations to reaction media
5. Displays reactions and plays sounds accordingly
6. Rinses and repeats at ~30 FPS

## Controls 🎮

- Press `q` to quit (revolutionary, I know)
- Wave your hands around and make faces for entertainment
- Try pointing up while smiling for maximum effect

## Known Issues 🐛

- Sometimes thinks you're sad when you're just concentrating
- Might play sounds on loop (that's a feature, not a bug!)
- Face detection can be wonky in poor lighting
- Code quality is... let's call it "experimental"

## Disclaimer ⚠️

This was literally just made to mess around with OpenCV and MediaPipe. Don't use this for anything important. It's basically digital silly putty. Have fun with it! 🎉

## License 📄

Do whatever you want with it. It's a joke project anyway! 🤷‍♂️

---



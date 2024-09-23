import cv2

class VideoData:
    def __init__(self, video_path: str):
         # Open the video file
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print(f"Error: Could not open video from {video_path}")
            return None

        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = video.get(cv2.CAP_PROP_FPS)

        frames = []
        frame_count = 0
        while True:
            ret, frame = video.read()

            # If the frame was not read successfully, we're at the end of the video
            if not ret:
                break

            frames.append(frame)
            frame_count += 1


        self.frames = frames
        self.first_frame = frames[0]
        self.length = frame_count / self.fps
        self.frame_count = frame_count

        video.release()

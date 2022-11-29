import cv2
import os, glob
import moviepy.video.io.ImageSequenceClip


if __name__ == "__main__":

    image_folder = '/home/rahim/Desktop/projects/mrob/notebooks/2dplots/s20/*.png'
    video_name = 's20.mp4'

    images = sorted(glob.glob(image_folder))

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=5)
    clip.write_videofile(video_name)
import yt_dlp
import av
import logging
import numpy as np

class YouTubeClient:
    def __init__(self, youtube_url):
        self.youtube_url = youtube_url
        self.video_container = None

    def get_stream_url(self):
        try:
            # yt-dlp configuration for extracting stream URL
            ydl_opts = {
                'format': 'best[ext=mp4]',  # Get the best available mp4 format
                'quiet': True,              # Silence yt-dlp output
                'noplaylist': True           # Ensure we don't fetch a playlist
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.youtube_url, download=False)
                stream_url = info_dict.get('url')
                if stream_url:
                    logging.info(f"Stream URL retrieved: {stream_url}")
                    return stream_url
                else:
                    logging.error("No suitable stream found.")
                    return None
        except Exception as e:
            logging.error(f"Error retrieving YouTube stream URL: {e}")
            return None

    def open_stream(self):
        """
        Open the YouTube stream using PyAV and get the video container.
        """
        stream_url = self.get_stream_url()
        if stream_url:
            try:
                logging.info("Opening stream with PyAV...")
                self.video_container = av.open(stream_url)
            except av.AVError as e:
                logging.error(f"Error opening stream with PyAV: {e}")
        else:
            logging.error("Failed to retrieve a valid stream URL.")

    def get_frame(self):
        """
        Get the next frame from the video container using PyAV.
        """
        if self.video_container is None:
            self.open_stream()

        try:
            for frame in self.video_container.decode(video=0):
                # Convert the PyAV frame to a numpy array
                image = frame.to_ndarray(format='bgr24')
                return image
        except Exception as e:
            logging.error(f"Error getting frame from stream: {e}")
            return None

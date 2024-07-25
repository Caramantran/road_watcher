import numpy as np
import cv2
from hikvisionapi import Client

class HikvisionClient:
    def __init__(self, ip, username, password):
        self.client = Client(ip, username, password)

    def get_frame(self):
        response = self.client.Streaming.channels[101].picture(method='get', type='opaque_data')
        img_data = b''
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                img_data += chunk
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame

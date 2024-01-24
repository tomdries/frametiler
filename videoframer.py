import os
import cv2
import numpy as np
from math import ceil
from string import ascii_lowercase


class VideoSampler:
    """
    Extracts frames from a video file.
    """
    def __init__(self, video_file):
        self.video_file = video_file
        self.video = cv2.VideoCapture(video_file)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        self.video_duration = self.frame_count / self.frame_rate
        self.sample_timestamps = []

    def sample_fixed_rate(self, sample_rate, t_start=0, t_end=None):
        """ 
        Samples frames at a fixed rate between t_start and t_end.
        """
        t_end = t_end if t_end is not None else self.video_duration
        self.sample_timestamps = [t for t in np.arange(t_start, t_end, sample_rate)]

    def sample_fixed_n(self, n_frames, t_start=0, t_end=None):
        """
        Samples a fixed number of frames, evenly spaced, between t_start and t_end.
        """
        t_end = t_end if t_end is not None else self.video_duration
        interval = (t_end - t_start) / (n_frames - 1)
        self.sample_timestamps = [t_start + i * interval for i in range(n_frames)]

    def save_frames_to_folder(self, folder_path):
        """
        Saves sampled frames to the specified folder.
        """
        os.makedirs(folder_path, exist_ok=True)
        for timestamp in self.sample_timestamps:
            self.video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = self.video.read()
            if not success:
                raise ValueError(f'Could not read frame at timestamp {timestamp}')
            frame_path = os.path.join(folder_path, f'frame_{int(timestamp)}.jpg')
            cv2.imwrite(frame_path, frame)

    def to_array(self):
        """
        Returns a list of sampled frames.
        """
        frames = []
        for timestamp in self.sample_timestamps:
            self.video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = self.video.read()
            if not success:
                raise ValueError(f'Could not read frame at timestamp {timestamp}')
            frames.append(frame)
        return frames


class Tiler:
    """
    Creates a mosaic from a set of images.

    Args:
        source: Either a list of images or a folder containing images.
        from_folder: Whether source is a folder or a list of images.
    """
    def __init__(self, source, from_folder=False):
        self.tiles = []
        if from_folder:
            self.tiles = [cv2.imread(os.path.join(source, f)) for f in os.listdir(source) if f.endswith('.jpg')]
        else:
            self.tiles = source
        self.assert_same_dimensions()

    def assert_same_dimensions(self):
        """
        Asserts that all tiles have the same dimensions.
        """
        if not all(t.shape == self.tiles[0].shape for t in self.tiles):
            raise ValueError("All tiles must have the same dimensions")

    def create_mosaic(self, n_rows, border_size=1, annotation_style=None,
                      annotation_position='topleft', annotation_size=0.1,
                      text_color=(255, 255, 255), bg_color=(100, 100, 100)):
        """
        Creates a mosaic from the tiles.

        Args:
            n_rows: Number of rows in the mosaic.
            border_size: Size of the border between tiles.
            annotation_style: Style of the annotation. Can be 'number', 'letter' or None.
            annotation_position: Position of the annotation. Can be 'topleft', 'topright', 'bottomleft', or 'bottomright'.
            annotation_size: Size of the annotation text relative to the tile height. (0.1 = 10% of tile height)
            text_color: Color of the annotation text.
            bg_color: Color of the annotation background.
        """
        n_tiles = len(self.tiles)
        n_cols = ceil(n_tiles / n_rows)
        tile_height, tile_width, _ = self.tiles[0].shape

        mosaic_width = n_cols * tile_width + (n_cols - 1) * border_size
        mosaic_height = n_rows * tile_height + (n_rows - 1) * border_size

        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
        for i, tile in enumerate(self.tiles):
            row, col = divmod(i, n_cols)
            x, y = col * (tile_width + border_size), row * (tile_height + border_size)
            mosaic[y:y + tile_height, x:x + tile_width] = tile

            if annotation_style:  # Add annotation
                if annotation_style == 'number':
                    text = str(i + 1)
                elif annotation_style == 'letter':
                    text = ascii_lowercase[i % 26]
                else:
                    raise ValueError("Invalid annotation style. Must be 'number', 'letter' or None.")

                font_scale = annotation_size * tile_height / 20  # Dynamic font scaling
                text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_width, text_height = text_size

                text_x, text_y = 0, 0
                if annotation_position == 'topleft':
                    text_x, text_y = x + 5, y + text_height + 5
                elif annotation_position == 'topright':
                    text_x, text_y = x + tile_width - text_width - 5, y + text_height + 5
                elif annotation_position == 'bottomleft':
                    text_x, text_y = x + 5, y + tile_height - 5
                elif annotation_position == 'bottomright':
                    text_x, text_y = x + tile_width - text_width - 5, y + tile_height - 5

                bg_x, bg_y = text_x, text_y - text_height

                if annotation_position in ['bottomleft', 'bottomright']:
                    bg_y -= baseline
                # Background rectangle for text
                cv2.rectangle(mosaic,
                              (bg_x, bg_y),
                              (text_x + text_width, text_y + baseline),
                              bg_color,
                              -1)

                # Draw text
                cv2.putText(mosaic,
                            text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, text_color,
                            4,
                            cv2.LINE_AA)
        return mosaic

    def resize_mosaic(self, mosaic, output_width, output_height):
        """
        Resizes the mosaic to a specified output size.
        """
        mosaic_height, mosaic_width = mosaic.shape[:2]
        scale = min(output_width / mosaic_width, output_height / mosaic_height)

        new_width = int(mosaic_width * scale)
        new_height = int(mosaic_height * scale)

        resized = cv2.resize(mosaic, (new_width, new_height))
        background = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        x_offset = (output_width - new_width) // 2
        y_offset = (output_height - new_height) // 2
        background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        return background

    def save_mosaic_to_image(self, mosaic, file_path):
        """
        Saves the mosaic to an image file.
        """
        cv2.imwrite(file_path, mosaic)
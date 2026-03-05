"""
Video Processing

Features:
- Video encoding/decoding with FFmpeg
- Streaming support (RTSP/RTMP/HTTP)
- Hardware acceleration (NVDEC/VAAPI/VideoToolbox)
- Frame extraction and video creation
- Video format conversion
- Real-time video processing
"""

from typing import Optional, Generator, Tuple, Union, Callable
import subprocess
import json
import tempfile
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class VideoReader:
    """Read video files and streams."""

    def __init__(
        self,
        source: Union[str, Path, int],
        backend: str = 'opencv',
        hwaccel: Optional[str] = None
    ):
        """
        Initialize video reader.

        Parameters
        ----------
        source : str, Path, or int
            Video file path, stream URL, or camera index
        backend : str, default='opencv'
            Backend to use: 'opencv', 'ffmpeg'
        hwaccel : str, optional
            Hardware acceleration: 'nvdec', 'vaapi', 'videotoolbox', 'qsv'
        """
        self.source = source
        self.backend = backend
        self.hwaccel = hwaccel
        self.cap = None
        self.process = None

        if backend == 'opencv':
            self._init_opencv()
        elif backend == 'ffmpeg':
            self._init_ffmpeg()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_opencv(self):
        """Initialize OpenCV video capture."""
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for 'opencv' backend")

        if isinstance(self.source, int):
            # Camera
            self.cap = cv2.VideoCapture(self.source)
        else:
            # File or stream
            self.cap = cv2.VideoCapture(str(self.source))

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

    def _init_ffmpeg(self):
        """Initialize FFmpeg video reader."""
        cmd = ['ffmpeg', '-i', str(self.source)]

        if self.hwaccel:
            cmd.extend(['-hwaccel', self.hwaccel])

        cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-'
        ])

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

    def get_properties(self) -> dict:
        """
        Get video properties.

        Returns
        -------
        props : dict
            Video properties (width, height, fps, frame_count, codec)
        """
        if self.backend == 'opencv' and self.cap:
            return {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            }
        elif self.backend == 'ffmpeg':
            # Use ffprobe to get properties
            return self._ffprobe_properties()
        else:
            return {}

    def _ffprobe_properties(self) -> dict:
        """Get video properties using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(self.source)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)

            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            if video_stream:
                fps_parts = video_stream.get('r_frame_rate', '0/1').split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 0

                return {
                    'width': video_stream.get('width', 0),
                    'height': video_stream.get('height', 0),
                    'fps': fps,
                    'frame_count': int(video_stream.get('nb_frames', 0)),
                    'codec': video_stream.get('codec_name', ''),
                    'duration': float(data.get('format', {}).get('duration', 0)),
                }
        except:
            pass

        return {}

    def read(self) -> Tuple[bool, Optional[NDArray]]:
        """
        Read next frame.

        Returns
        -------
        success : bool
            Whether frame was read successfully
        frame : ndarray or None
            Frame data (H, W, 3) in RGB format
        """
        if self.backend == 'opencv' and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ret, frame if ret else None
        elif self.backend == 'ffmpeg' and self.process:
            props = self.get_properties()
            w, h = props['width'], props['height']
            frame_size = w * h * 3

            raw_frame = self.process.stdout.read(frame_size)
            if len(raw_frame) == frame_size:
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(h, w, 3)
                return True, frame
            else:
                return False, None
        else:
            return False, None

    def iter_frames(self) -> Generator[NDArray, None, None]:
        """
        Iterate over all frames.

        Yields
        ------
        frame : ndarray
            Video frame (H, W, 3)
        """
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame

    def seek(self, frame_number: int):
        """
        Seek to specific frame.

        Parameters
        ----------
        frame_number : int
            Target frame number
        """
        if self.backend == 'opencv' and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        else:
            raise NotImplementedError("Seek not supported for this backend")

    def release(self):
        """Release video resources."""
        if self.cap:
            self.cap.release()
        if self.process:
            self.process.kill()
            self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """Write video files."""

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = 'mp4v',
        backend: str = 'opencv',
        hwaccel: Optional[str] = None,
        bitrate: Optional[str] = None
    ):
        """
        Initialize video writer.

        Parameters
        ----------
        output_path : str or Path
            Output video path
        fps : float
            Frames per second
        frame_size : tuple
            Frame size (width, height)
        codec : str, default='mp4v'
            Video codec: 'mp4v', 'h264', 'h265', 'vp9', 'av1'
        backend : str, default='opencv'
            Backend: 'opencv', 'ffmpeg'
        hwaccel : str, optional
            Hardware encoder: 'nvenc', 'vaapi', 'videotoolbox', 'qsv'
        bitrate : str, optional
            Target bitrate (e.g., '5M', '2000k')
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.backend = backend
        self.hwaccel = hwaccel
        self.bitrate = bitrate
        self.writer = None
        self.process = None

        if backend == 'opencv':
            self._init_opencv()
        elif backend == 'ffmpeg':
            self._init_ffmpeg()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_opencv(self):
        """Initialize OpenCV video writer."""
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for 'opencv' backend")

        # Map codec names to FourCC
        codec_map = {
            'mp4v': 'mp4v',
            'h264': 'avc1',
            'h265': 'hev1',
            'xvid': 'XVID',
        }

        fourcc_str = codec_map.get(self.codec, self.codec)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.frame_size
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")

    def _init_ffmpeg(self):
        """Initialize FFmpeg video writer."""
        w, h = self.frame_size

        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{w}x{h}',
            '-r', str(self.fps),
            '-i', '-',  # Input from pipe
        ]

        # Hardware encoder
        if self.hwaccel:
            if self.hwaccel == 'nvenc':
                if self.codec == 'h264':
                    cmd.extend(['-c:v', 'h264_nvenc'])
                elif self.codec == 'h265':
                    cmd.extend(['-c:v', 'hevc_nvenc'])
            elif self.hwaccel == 'vaapi':
                cmd.extend(['-vaapi_device', '/dev/dri/renderD128'])
                cmd.extend(['-c:v', f'{self.codec}_vaapi'])
            elif self.hwaccel == 'videotoolbox':
                cmd.extend(['-c:v', f'{self.codec}_videotoolbox'])
        else:
            # Software encoder
            codec_map = {
                'h264': 'libx264',
                'h265': 'libx265',
                'vp9': 'libvpx-vp9',
                'av1': 'libaom-av1',
            }
            cmd.extend(['-c:v', codec_map.get(self.codec, 'libx264')])

        # Bitrate
        if self.bitrate:
            cmd.extend(['-b:v', self.bitrate])

        # Output
        cmd.append(str(self.output_path))

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def write(self, frame: NDArray):
        """
        Write frame to video.

        Parameters
        ----------
        frame : ndarray
            Frame to write (H, W, 3) in RGB format
        """
        if self.backend == 'opencv' and self.writer:
            # Convert RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(frame_bgr)
        elif self.backend == 'ffmpeg' and self.process:
            self.process.stdin.write(frame.tobytes())
        else:
            raise RuntimeError("Video writer not initialized")

    def release(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()
        if self.process:
            self.process.stdin.close()
            self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoProcessor:
    """Process videos with frame-by-frame operations."""

    @staticmethod
    def process_video(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        process_fn: Callable[[NDArray], NDArray],
        show_progress: bool = True
    ):
        """
        Process video frame-by-frame.

        Parameters
        ----------
        input_path : str or Path
            Input video path
        output_path : str or Path
            Output video path
        process_fn : callable
            Function to apply to each frame: frame -> processed_frame
        show_progress : bool, default=True
            Show progress bar
        """
        with VideoReader(input_path) as reader:
            props = reader.get_properties()

            with VideoWriter(
                output_path,
                fps=props['fps'],
                frame_size=(props['width'], props['height'])
            ) as writer:
                total_frames = props.get('frame_count', 0)

                for i, frame in enumerate(reader.iter_frames()):
                    # Process frame
                    processed = process_fn(frame)

                    # Write
                    writer.write(processed)

                    if show_progress and total_frames > 0:
                        progress = (i + 1) / total_frames * 100
                        print(f"\rProcessing: {progress:.1f}%", end='', flush=True)

                if show_progress:
                    print()  # New line

    @staticmethod
    def extract_frames(
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        frame_interval: int = 1,
        max_frames: Optional[int] = None
    ) -> int:
        """
        Extract frames from video.

        Parameters
        ----------
        video_path : str or Path
            Input video path
        output_dir : str or Path
            Output directory for frames
        frame_interval : int, default=1
            Extract every Nth frame
        max_frames : int, optional
            Maximum number of frames to extract

        Returns
        -------
        count : int
            Number of frames extracted
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        with VideoReader(video_path) as reader:
            for i, frame in enumerate(reader.iter_frames()):
                if i % frame_interval == 0:
                    output_path = output_dir / f"frame_{count:06d}.png"

                    # Save frame
                    if HAS_OPENCV:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), frame_bgr)
                    else:
                        from PIL import Image
                        Image.fromarray(frame).save(output_path)

                    count += 1

                    if max_frames and count >= max_frames:
                        break

        return count

    @staticmethod
    def create_video_from_frames(
        frame_paths: list,
        output_path: Union[str, Path],
        fps: float = 30.0
    ):
        """
        Create video from image frames.

        Parameters
        ----------
        frame_paths : list
            List of frame image paths
        output_path : str or Path
            Output video path
        fps : float, default=30.0
            Frames per second
        """
        if not frame_paths:
            raise ValueError("No frames provided")

        # Read first frame to get size
        if HAS_OPENCV:
            first_frame = cv2.imread(str(frame_paths[0]))
            h, w = first_frame.shape[:2]
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        else:
            from PIL import Image
            first_img = Image.open(frame_paths[0])
            w, h = first_img.size
            first_frame = np.array(first_img)

        with VideoWriter(output_path, fps, (w, h)) as writer:
            writer.write(first_frame)

            for frame_path in frame_paths[1:]:
                if HAS_OPENCV:
                    frame = cv2.imread(str(frame_path))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    from PIL import Image
                    frame = np.array(Image.open(frame_path))

                writer.write(frame)


# Convenience functions
def read_video(video_path: Union[str, Path], max_frames: Optional[int] = None) -> NDArray:
    """
    Read video into memory.

    Parameters
    ----------
    video_path : str or Path
        Video file path
    max_frames : int, optional
        Maximum frames to read

    Returns
    -------
    frames : ndarray
        Video frames (T, H, W, 3)
    """
    frames = []
    with VideoReader(video_path) as reader:
        for i, frame in enumerate(reader.iter_frames()):
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break

    return np.stack(frames) if frames else np.array([])


def write_video(
    frames: NDArray,
    output_path: Union[str, Path],
    fps: float = 30.0
):
    """
    Write frames to video file.

    Parameters
    ----------
    frames : ndarray
        Video frames (T, H, W, 3)
    output_path : str or Path
        Output video path
    fps : float, default=30.0
        Frames per second
    """
    if len(frames) == 0:
        raise ValueError("No frames to write")

    T, H, W, C = frames.shape

    with VideoWriter(output_path, fps, (W, H)) as writer:
        for frame in frames:
            writer.write(frame)


def convert_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    codec: str = 'h264',
    bitrate: Optional[str] = None
):
    """
    Convert video format/codec.

    Parameters
    ----------
    input_path : str or Path
        Input video path
    output_path : str or Path
        Output video path
    codec : str, default='h264'
        Target codec
    bitrate : str, optional
        Target bitrate
    """
    with VideoReader(input_path) as reader:
        props = reader.get_properties()

        with VideoWriter(
            output_path,
            fps=props['fps'],
            frame_size=(props['width'], props['height']),
            codec=codec,
            bitrate=bitrate
        ) as writer:
            for frame in reader.iter_frames():
                writer.write(frame)

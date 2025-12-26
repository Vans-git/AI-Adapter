import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
from typing import Optional
from ..schemas import MatchResponse, MatchBBox


class RecordingModelAdapter:
    """
    Adapter that:
    - accepts a video recording (bytes),
    - extracts a frame at a provided timestamp (milliseconds),
    - matches the provided image against that frame using ORB feature matching,
    - returns a MatchResponse with confidence and optional bbox.

    Notes:
    - This adapter expects the "recording" to be a video (mp4, mov, etc.). If you require audio-only
      recordings, matching an image is not possible and the adapter will raise an error.
    - max_processing_time_seconds can be used by callers to enforce a hard timeout.
    """

    def __init__(self, max_processing_time_seconds: int = 5):
        self.max_processing_time_seconds = max_processing_time_seconds
        # ORB params
        self._orb = cv2.ORB_create(nfeatures=1000)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _save_bytes_to_tempfile(self, b: bytes, suffix: str = "") -> str:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.write(b)
        tf.flush()
        tf.close()
        return tf.name

    def _extract_frame_at_ms(self, video_path: str, timestamp_ms: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None
        # set position in milliseconds
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return frame

    def _decode_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def _match_images(self, frame: np.ndarray, query: np.ndarray) -> (bool, float, Optional[MatchBBox]):
        """
        Perform ORB feature matching between frame (scene) and query (template).
        Returns (matched, confidence, bbox)
        - confidence is in [0,1], computed from number of good matches relative to an expected scale.
        - bbox if homography is found (bounding box of query in frame), otherwise None.
        """
        # convert to grayscale
        scene_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self._orb.detectAndCompute(query_gray, None)  # query
        kp2, des2 = self._orb.detectAndCompute(scene_gray, None)  # scene/frame

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return False, 0.0, None

        matches = self._bf.match(des1, des2)
        if not matches:
            return False, 0.0, None

        # sort by distance (lower is better)
        matches = sorted(matches, key=lambda x: x.distance)
        # pick top matches
        top_matches = matches[:50]  # consider up to first 50 matches
        # heuristics: count "good" matches by distance threshold
        good_matches = [m for m in top_matches if m.distance < 60]  # tuned threshold; may be adjusted
        num_good = len(good_matches)

        # Confidence heuristic: saturate at 1.0 when many good matches found
        confidence = min(1.0, num_good / 20.0)  # 20 good matches -> confidence 1.0

        bbox = None
        if num_good >= 8:
            # Try to estimate homography to get bbox
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            try:
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    hq, wq = query_gray.shape
                    pts = np.float32([[0, 0], [wq, 0], [wq, hq], [0, hq]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, H)
                    xs = dst[:, 0, 0]
                    ys = dst[:, 0, 1]
                    x_min = int(max(0, xs.min()))
                    y_min = int(max(0, ys.min()))
                    x_max = int(min(frame.shape[1] - 1, xs.max()))
                    y_max = int(min(frame.shape[0] - 1, ys.max()))
                    bbox = MatchBBox(x=x_min, y=y_min, w=max(0, x_max - x_min), h=max(0, y_max - y_min))
            except Exception:
                bbox = None

        matched = num_good >= 6  # require a minimum number of good matches
        return matched, float(confidence), bbox

    def match_image_at_time(self, recording_bytes: bytes, image_bytes: bytes, timestamp_ms: int) -> MatchResponse:
        """
        Synchronous method to perform the matching. The caller is responsible for enforcing a timeout.
        """
        video_path = None
        image_path = None
        try:
            video_path = self._save_bytes_to_tempfile(recording_bytes, suffix=".mp4")
            # decode frame directly from video, no need to save image, but keep for debugging
            frame = self._extract_frame_at_ms(video_path, timestamp_ms)
            if frame is None:
                return MatchResponse(
                    matched=False,
                    confidence=0.0,
                    timestamp_ms=timestamp_ms,
                    bbox=None,
                    message="Could not extract frame at the requested timestamp. Is the recording a valid video and is the timestamp within duration?",
                    processed_at=datetime.utcnow(),
                )

            query_img = self._decode_image_bytes(image_bytes)
            if query_img is None:
                return MatchResponse(
                    matched=False,
                    confidence=0.0,
                    timestamp_ms=timestamp_ms,
                    bbox=None,
                    message="Could not decode the provided image. Ensure it is a valid image format (jpg/png).",
                    processed_at=datetime.utcnow(),
                )

            matched, confidence, bbox = self._match_images(frame, query_img)
            msg = "matched" if matched else "no strong match"
            return MatchResponse(
                matched=matched,
                confidence=confidence,
                timestamp_ms=timestamp_ms,
                bbox=bbox,
                message=msg,
                processed_at=datetime.utcnow(),
            )
        finally:
            # cleanup
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception:
                    pass
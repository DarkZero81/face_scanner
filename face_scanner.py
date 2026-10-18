"""
نظام تجريبي للتعرّف على الوجه باستخدام DeepFace وArcFace وRetinaFace.

التثبيت:
    pip install deepface opencv-python pillow

ملاحظة: قاعدة البيانات ملف Pickle محلي. لا تفتح ملفات Pickle من مصادر غير موثوقة.
"""

from __future__ import annotations

import os
import pickle
import queue
import sys
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any


for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")


import cv2
import numpy as np
from PIL import Image, ImageTk


APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "face_database.pkl"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
# عتبة أكثر تحفظًا لأن الصور المستخدمة للتمثيل مطموسة جزئيًا.
COSINE_DISTANCE_THRESHOLD = 0.45
MATCH_SAMPLE_COUNT = 2
CAPTURE_COUNT = 5
CAMERA_RESOLUTIONS = {
    "640 × 480": (640, 480),
    "1280 × 720": (1280, 720),
    "1920 × 1080": (1920, 1080),
}
MIN_FACE_SIZE = 90
ENROLLMENT_STEPS = (
    ("أمام", "انظر مباشرة إلى الكاميرا"),
    ("يمين", "أدر وجهك قليلًا إلى اليمين"),
    ("يسار", "أدر وجهك قليلًا إلى اليسار"),
    ("أعلى", "ارفع ذقنك قليلًا"),
    ("أسفل", "اخفض ذقنك قليلًا"),
)
COLORS = {
    "background": "#0b1220",
    "surface": "#111c2f",
    "surface_light": "#18263b",
    "text": "#e5edf7",
    "muted": "#9aabc2",
    "accent": "#38bdf8",
    "success": "#34d399",
    "warning": "#fbbf24",
}


@dataclass
class FaceDetection:
    """نتيجة وجه واحدة، مع إحداثياتها وبيانات المطابقة إن وُجدت."""

    box: tuple[int, int, int, int]
    username: str | None = None
    distance: float | None = None
    quality_message: str | None = None
    alternate_username: str | None = None
    alternate_distance: float | None = None


class ExpandableHelp:
    """شرح مضمّن قابل للفتح دون إغراق لوحة التحكم بالنص."""

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        text: str,
        background: str = COLORS["surface"],
    ) -> None:
        self.frame = tk.Frame(parent, bg=background)
        self.text = text
        self.title = title
        self.background = background
        self.expanded = False
        self.toggle_button = tk.Button(
            self.frame,
            text=f"ⓘ {title}",
            command=self.toggle,
            bg=background,
            fg=COLORS["accent"],
            activebackground=background,
            activeforeground=COLORS["text"],
            relief="flat",
            cursor="hand2",
            anchor="e",
            font=("Arial", 10, "bold"),
        )
        self.toggle_button.pack(anchor="e")
        self.detail = tk.Label(
            self.frame,
            text=text,
            bg=COLORS["surface_light"],
            fg=COLORS["text"],
            justify="right",
            anchor="e",
            wraplength=290,
            padx=10,
            pady=9,
        )

    def toggle(self) -> None:
        self.expanded = not self.expanded
        if self.expanded:
            self.detail.pack(fill="x", pady=(3, 0))
            self.toggle_button.configure(text="إخفاء الشرح")
        else:
            self.detail.pack_forget()
            self.toggle_button.configure(text=f"ⓘ {self.title}")


def assess_face_quality(
    frame: np.ndarray, box: tuple[int, int, int, int]
) -> str | None:
    """يرفض الوجوه الصغيرة أو الضبابية أو سيئة الإضاءة قبل المطابقة."""
    x, y, width, height = box
    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        return "اقترب من الكاميرا؛ الوجه صغير جدًا للتعرّف."
    frame_height, frame_width = frame.shape[:2]
    left, top = max(x, 0), max(y, 0)
    right, bottom = min(x + width, frame_width), min(y + height, frame_height)
    face = frame[top:bottom, left:right]
    if face.size == 0:
        return "تعذّر قياس جودة الوجه."
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < 35 or float(np.mean(gray)) > 220:
        return "حسّن الإضاءة على الوجه."
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 18:
        return "الصورة غير واضحة؛ ثبّت الكاميرا والوجه."
    return None


class FaceEngine:
    """استخراج تمثيل الوجه المطَمّس أسفله لتقليل الاعتماد على الفم والذقن."""

    def __init__(self) -> None:
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise RuntimeError(
                "مكتبة DeepFace غير مثبتة. ثبّتها بالأمر: "
                "pip install deepface opencv-python pillow"
            ) from exc
        self.deepface = DeepFace
        self.inference_lock = threading.Lock()

    def _extract_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return self.deepface.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
            color_face="rgb",
            normalize_face=False,
        )

    @staticmethod
    def _upper_face_image(face_rgb: np.ndarray) -> np.ndarray:
        """يطمس أسفل الوجه مع إبقاء منطقة العينين والأنف العلوي واضحة."""
        face_bgr = cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        height, width = face_bgr.shape[:2]
        mask_start = int(height * 0.60)
        if mask_start < height:
            # لون رمادي ثابت لتقليل أثر تفاصيل الفم والذقن في التمثيل.
            face_bgr[mask_start:, :] = (127, 127, 127)
        return face_bgr

    def embedding_from_frame(
        self, frame: np.ndarray, mask_lower_face: bool = True
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """يستخرج embedding لوجه واحد من صورة كاميرا خام."""
        upper, full, box = self.embeddings_from_frame(frame)
        return (upper if mask_lower_face else full), box

    def embeddings_from_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        """يستخرج تمثيلي الوجه الكامل والعلوي من كشف واحد."""
        faces = self._extract_faces(frame)
        if len(faces) != 1:
            raise ValueError("يجب أن يظهر وجه واحد فقط في الصورة.")

        face = faces[0]
        face_rgb = face["face"]
        embeddings = []
        for face_image in (
            self._upper_face_image(face_rgb),
            cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
        ):
            representations = self.deepface.represent(
                img_path=face_image,
                model_name=MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
                align=False,
            )
            if not representations:
                raise RuntimeError("لم يتمكن ArcFace من إنشاء تمثيل الوجه.")
            embeddings.append(
                np.asarray(representations[0]["embedding"], dtype=np.float32)
            )

        area = face.get("facial_area", {})
        box = (
            int(area.get("x", 0)),
            int(area.get("y", 0)),
            int(area.get("w", 0)),
            int(area.get("h", 0)),
        )
        return embeddings[0], embeddings[1], box

    def analyze_frame(
        self,
        frame: np.ndarray,
        database: "FaceDatabase",
        representation: str = "upper",
    ) -> list[FaceDetection]:
        """يستخرج الوجوه ويفحصها؛ الوجوه غير المسجلة تعاد بإطار دون اسم."""
        try:
            faces = self._extract_faces(frame)
        except ValueError as exc:
            # عدم وجود وجه أمام الكاميرا حالة عادية أثناء الفحص، أما أخطاء النموذج
            # الأخرى فيتم إظهارها للمستخدم بدل إخفائها.
            if "face could not be detected" in str(exc).lower():
                return []
            raise

        detections: list[FaceDetection] = []
        for face in faces:
            area = face.get("facial_area", {})
            box = (
                int(area.get("x", 0)),
                int(area.get("y", 0)),
                int(area.get("w", 0)),
                int(area.get("h", 0)),
            )
            if not box[2] or not box[3]:
                continue

            quality_message = assess_face_quality(frame, box)
            if quality_message:
                detections.append(
                    FaceDetection(box=box, quality_message=quality_message)
                )
                continue

            face_rgb = face["face"]
            embeddings: dict[str, np.ndarray] = {}
            modes = ("upper", "full") if representation == "compare" else (representation,)
            for mode in modes:
                face_image = (
                    self._upper_face_image(face_rgb)
                    if mode == "upper"
                    else cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                representations = self.deepface.represent(
                    img_path=face_image,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False,
                )
                if representations:
                    embeddings[mode] = np.asarray(
                        representations[0]["embedding"], dtype=np.float32
                    )
            if not embeddings:
                continue

            primary_mode = "upper" if representation == "compare" else representation
            match = (
                database.find_match(embeddings[primary_mode], primary_mode)
                if primary_mode in embeddings
                else None
            )
            alternate_match = (
                database.find_match(embeddings["full"], "full")
                if representation == "compare" and "full" in embeddings
                else None
            )
            if match is None:
                detections.append(
                    FaceDetection(
                        box=box,
                        alternate_username=(
                            alternate_match[0] if alternate_match else None
                        ),
                        alternate_distance=(
                            alternate_match[1] if alternate_match else None
                        ),
                    )
                )
            else:
                username, distance = match
                detections.append(
                    FaceDetection(
                        box=box,
                        username=username,
                        distance=distance,
                        alternate_username=(
                            alternate_match[0] if alternate_match else None
                        ),
                        alternate_distance=(
                            alternate_match[1] if alternate_match else None
                        ),
                    )
                )
        return detections

    def detect_frame(self, frame: np.ndarray) -> list[FaceDetection]:
        """يكتشف مواقع الوجوه دون حساب embeddings خلال التسجيل."""
        try:
            faces = self._extract_faces(frame)
        except ValueError as exc:
            if "face could not be detected" in str(exc).lower():
                return []
            raise

        detections = []
        for face in faces:
            area = face.get("facial_area", {})
            box = (
                int(area.get("x", 0)),
                int(area.get("y", 0)),
                int(area.get("w", 0)),
                int(area.get("h", 0)),
            )
            if box[2] and box[3]:
                detections.append(FaceDetection(box=box))
        return detections


class FaceDatabase:
    """قاعدة بيانات محلية بسيطة تحفظ المستخدمين و embeddings الخاصة بهم."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.users: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            # يجب ألا يكون ملف Pickle مصدره جهة غير موثوقة.
            with self.path.open("rb") as database_file:
                data = pickle.load(database_file)
            if not isinstance(data, dict):
                raise RuntimeError("تنسيق قاعدة بيانات الوجوه غير صالح.")
            self.users = data

    def contains(self, username: str) -> bool:
        with self.lock:
            return username in self.users

    def list_users(self) -> list[tuple[str, str]]:
        with self.lock:
            return [
                (username, str(record.get("details", {}).get("name", "")))
                for username, record in self.users.items()
            ]

    def count_users_with_full_embeddings(self) -> int:
        with self.lock:
            return sum(
                bool(record.get("embeddings_full"))
                for record in self.users.values()
            )

    def has_full_embeddings(self, username: str) -> bool:
        with self.lock:
            return bool(self.users.get(username, {}).get("embeddings_full"))

    def delete_user(self, username: str) -> None:
        with self.lock:
            if username not in self.users:
                raise ValueError("المستخدم غير موجود في قاعدة البيانات.")
            record = self.users.pop(username)
            try:
                self._save_locked()
            except Exception:
                self.users[username] = record
                raise

    def add_user(
        self,
        username: str,
        details: dict[str, str],
        embeddings: list[np.ndarray],
        full_embeddings: list[np.ndarray] | None = None,
    ) -> None:
        with self.lock:
            if username in self.users:
                raise ValueError("اسم المستخدم مسجل مسبقًا.")
            self.users[username] = {
                "details": details,
                "embeddings": [embedding.tolist() for embedding in embeddings],
            }
            if full_embeddings is not None:
                self.users[username]["embeddings_full"] = [
                    embedding.tolist() for embedding in full_embeddings
                ]
            try:
                self._save_locked()
            except Exception:
                del self.users[username]
                raise

    def update_user(
        self,
        username: str,
        details: dict[str, str],
        embeddings: list[np.ndarray],
        full_embeddings: list[np.ndarray],
    ) -> None:
        with self.lock:
            if username not in self.users:
                raise ValueError("المستخدم غير موجود في قاعدة البيانات.")
            previous_record = self.users[username]
            self.users[username] = {
                "details": details,
                "embeddings": [embedding.tolist() for embedding in embeddings],
                "embeddings_full": [
                    embedding.tolist() for embedding in full_embeddings
                ],
            }
            try:
                self._save_locked()
            except Exception:
                self.users[username] = previous_record
                raise

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.path.parent, delete=False
            ) as temporary_file:
                temporary_path = temporary_file.name
                pickle.dump(self.users, temporary_file, protocol=pickle.HIGHEST_PROTOCOL)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def find_match(
        self, embedding: np.ndarray, representation: str = "upper"
    ) -> tuple[str, float] | None:
        if representation not in ("upper", "full"):
            raise ValueError("نوع تمثيل الوجه غير مدعوم.")
        query_norm = float(np.linalg.norm(embedding))
        if query_norm == 0:
            return None

        with self.lock:
            users = list(self.users.items())
        user_distances: list[tuple[str, float]] = []
        for username, record in users:
            distances: list[float] = []
            embeddings_key = "embeddings" if representation == "upper" else "embeddings_full"
            for stored in record.get(embeddings_key, []):
                known = np.asarray(stored, dtype=np.float32)
                known_norm = float(np.linalg.norm(known))
                if known_norm == 0 or known.shape != embedding.shape:
                    continue
                distance = 1.0 - float(
                    np.dot(embedding, known) / (query_norm * known_norm)
                )
                distances.append(distance)
            if distances:
                closest_distances = sorted(distances)[:MATCH_SAMPLE_COUNT]
                user_distances.append(
                    (username, sum(closest_distances) / len(closest_distances))
                )

        if user_distances:
            best_username, best_distance = min(user_distances, key=lambda item: item[1])
            if best_distance <= COSINE_DISTANCE_THRESHOLD:
                return best_username, best_distance
        return None

    def get_details(self, username: str) -> dict[str, str]:
        with self.lock:
            record = self.users.get(username, {})
            return dict(record.get("details", {}))


class CameraStream:
    """التقاط الكاميرا وتحليلها في خيوط خلفية مستقلة عن Tkinter."""

    def __init__(self, database: FaceDatabase) -> None:
        self.database = database
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.detections: list[FaceDetection] = []
        self.detection_revision = 0
        self.error: str | None = None
        self.scan_enabled = False
        self.detection_enabled = False
        self.representation = "upper"
        self.camera_commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.camera_statuses: queue.Queue[str] = queue.Queue()
        self.camera = cv2.VideoCapture(
            0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        )
        if not self.camera.isOpened() and os.name == "nt":
            self.camera.release()
            self.camera = cv2.VideoCapture(0, cv2.CAP_ANY)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError("تعذر فتح كاميرا الجهاز. تحقق من توصيلها وأذوناتها.")

        self.engine: FaceEngine | None = None
        self.engine_lock = threading.Lock()
        self.capture_thread = threading.Thread(
            target=self._capture_loop, name="camera-capture", daemon=True
        )
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop, name="face-analysis", daemon=True
        )
        self.capture_thread.start()
        self.analysis_thread.start()
        self.set_resolution(CAMERA_RESOLUTIONS["1280 × 720"])

    def get_engine(self) -> FaceEngine:
        # تحميل DeepFace يتم عند أول استخدام ومن خيط خلفي، لا عند فتح النافذة.
        with self.engine_lock:
            if self.engine is None:
                self.engine = FaceEngine()
            return self.engine

    def set_scan_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.scan_enabled = enabled
            self.detection_enabled = enabled
            if not enabled:
                self.detections = []

    def set_detection_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.detection_enabled = enabled
            if not enabled:
                self.detections = []
                self.detection_revision += 1

    def set_representation(self, representation: str) -> None:
        if representation not in ("upper", "full", "compare"):
            raise ValueError("نوع تمثيل الوجه غير مدعوم.")
        with self.lock:
            self.representation = representation

    def snapshot(
        self,
    ) -> tuple[np.ndarray | None, list[FaceDetection], str | None, int]:
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            detections = list(self.detections)
            error = self.error
            revision = self.detection_revision
            self.error = None
        return frame, detections, error, revision

    def _capture_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                self._apply_camera_commands()
                success, frame = self.camera.read()
                if not success:
                    with self.lock:
                        self.error = "فشل التقاط إطار من الكاميرا."
                    time.sleep(0.1)
                    continue
                with self.lock:
                    self.frame = frame
                time.sleep(0.01)
        finally:
            self.camera.release()

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        self.camera_commands.put(("resolution", resolution))

    def set_optical_zoom(self, zoom: float) -> None:
        self.camera_commands.put(("zoom", zoom))

    def poll_camera_status(self) -> str | None:
        latest_status = None
        while True:
            try:
                latest_status = self.camera_statuses.get_nowait()
            except queue.Empty:
                return latest_status

    def _apply_camera_commands(self) -> None:
        while True:
            try:
                command, value = self.camera_commands.get_nowait()
            except queue.Empty:
                return

            if command == "resolution":
                width, height = value
                try:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                except cv2.error as exc:
                    self.camera_statuses.put(f"تعذر تغيير الدقة: {exc}")
                    continue
                if (actual_width, actual_height) == (width, height):
                    self.camera_statuses.put(
                        f"الدقة الحالية: {actual_width} × {actual_height}"
                    )
                else:
                    self.camera_statuses.put(
                        f"الكاميرا لم تصل للدقة المطلوبة؛ الحالية "
                        f"{actual_width} × {actual_height}."
                    )
            elif command == "zoom":
                try:
                    self.camera.set(cv2.CAP_PROP_ZOOM, value)
                    actual_zoom = self.camera.get(cv2.CAP_PROP_ZOOM)
                except cv2.error as exc:
                    self.camera_statuses.put(f"تعذر تقريب الكاميرا: {exc}")
                    continue
                if value <= 0:
                    self.camera_statuses.put("تم إيقاف التقريب البصري.")
                else:
                    self.camera_statuses.put(
                        "تم تطبيق تقريب الكاميرا."
                        if actual_zoom > 0
                        else "التقريب البصري غير مدعوم؛ استخدم تقريب العرض الرقمي."
                    )

    def _analysis_loop(self) -> None:
        while not self.stop_event.wait(0.65):
            with self.lock:
                if not self.detection_enabled or self.frame is None:
                    continue
                frame = self.frame.copy()
                scan_enabled = self.scan_enabled
                representation = self.representation
            try:
                engine = self.get_engine()
                with engine.inference_lock:
                    if scan_enabled:
                        detections = engine.analyze_frame(
                            frame, self.database, representation
                        )
                    else:
                        detections = engine.detect_frame(frame)
                with self.lock:
                    self.detections = detections
                    self.detection_revision += 1
            except Exception as exc:
                with self.lock:
                    self.detections = []
                    self.error = f"حدث خطأ أثناء تحليل الوجه: {exc}"
                    self.detection_revision += 1

    def close(self) -> None:
        self.stop_event.set()
        self.capture_thread.join(timeout=1.5)
        self.analysis_thread.join(timeout=1.5)
        if self.camera.isOpened():
            self.camera.release()


class CameraWindow:
    """نافذة كاميرا أساسية مشتركة بين التسجيل والفحص."""

    def __init__(self, parent: tk.Misc, database: FaceDatabase, title: str) -> None:
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1120x720")
        self.window.minsize(900, 600)
        self.window.configure(bg=COLORS["background"])
        self.database = database
        self.stream = CameraStream(database)
        self.image_reference: ImageTk.PhotoImage | None = None
        self.show_face_guide = False
        self.guide_hint = "ضع وجهك داخل الإطار"
        self.guide_color = COLORS["accent"]
        self.last_detection_revision = -1
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.window.columnconfigure(0, weight=3)
        self.window.rowconfigure(1, weight=1)
        self.camera_controls = tk.Frame(
            self.window, bg=COLORS["surface"], padx=12, pady=8
        )
        self.camera_controls.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(12, 0)
        )
        tk.Label(
            self.camera_controls,
            text="الدقة",
            bg=COLORS["surface"],
            fg=COLORS["text"],
        ).pack(side="right", padx=(8, 3))
        self.resolution_var = tk.StringVar(value="1280 × 720")
        self.resolution_selector = ttk.Combobox(
            self.camera_controls,
            textvariable=self.resolution_var,
            values=tuple(CAMERA_RESOLUTIONS),
            state="readonly",
            width=13,
        )
        self.resolution_selector.pack(side="right", padx=4)
        self.resolution_selector.bind(
            "<<ComboboxSelected>>", self._change_resolution
        )
        tk.Label(
            self.camera_controls,
            text="تقريب بصري (إن دعمته الكاميرا)",
            bg=COLORS["surface"],
            fg=COLORS["text"],
        ).pack(side="right", padx=(12, 3))
        self.optical_zoom_var = tk.DoubleVar(value=0)
        tk.Scale(
            self.camera_controls,
            variable=self.optical_zoom_var,
            from_=0,
            to=4,
            resolution=1,
            orient="horizontal",
            length=100,
            showvalue=True,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            highlightthickness=0,
            command=self._change_optical_zoom,
        ).pack(side="right", padx=4)
        tk.Label(
            self.camera_controls,
            text="تقريب العرض فقط",
            bg=COLORS["surface"],
            fg=COLORS["text"],
        ).pack(side="right", padx=(12, 3))
        self.digital_zoom_var = tk.DoubleVar(value=1)
        tk.Scale(
            self.camera_controls,
            variable=self.digital_zoom_var,
            from_=1,
            to=4,
            resolution=0.25,
            orient="horizontal",
            length=100,
            showvalue=True,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            highlightthickness=0,
        ).pack(side="right", padx=4)
        self.camera_status_var = tk.StringVar(
            value="تقريب العرض لا يزيد تفاصيل الوجه؛ اختر دقة تدعمها الكاميرا."
        )
        tk.Label(
            self.camera_controls,
            textvariable=self.camera_status_var,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
        ).pack(side="left", padx=8)
        self.optical_zoom_after_id: str | None = None

        self.preview = tk.Canvas(
            self.window,
            bg=COLORS["background"],
            highlightthickness=0,
            bd=0,
        )
        self.preview.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        self.image_item = self.preview.create_image(0, 0, anchor="center")
        self.guide_item = self.preview.create_oval(
            0, 0, 0, 0, outline=COLORS["accent"], width=3, dash=(8, 5), state="hidden"
        )
        self.guide_text_item = self.preview.create_text(
            0,
            0,
            text=self.guide_hint,
            fill=COLORS["text"],
            font=("Arial", 13, "bold"),
            state="hidden",
        )
        self.eye_line_item = self.preview.create_line(
            0, 0, 0, 0, fill=COLORS["accent"], width=2, dash=(4, 5), state="hidden"
        )
        self.window.after(30, self._refresh)

    def _change_resolution(self, _event: tk.Event | None = None) -> None:
        self.stream.set_resolution(CAMERA_RESOLUTIONS[self.resolution_var.get()])
        self.camera_status_var.set("جارٍ تغيير دقة الكاميرا...")

    def _change_optical_zoom(self, value: str) -> None:
        if self.optical_zoom_after_id is not None:
            self.window.after_cancel(self.optical_zoom_after_id)
        self.optical_zoom_after_id = self.window.after(
            250, lambda: self._apply_optical_zoom(float(value))
        )

    def _apply_optical_zoom(self, value: float) -> None:
        self.optical_zoom_after_id = None
        self.stream.set_optical_zoom(value)

    def _refresh(self) -> None:
        if not self.window.winfo_exists():
            return
        camera_status = self.stream.poll_camera_status()
        if camera_status:
            self.camera_status_var.set(camera_status)
        frame, detections, error, revision = self.stream.snapshot()
        if error:
            self.on_stream_error(error)
        if frame is not None:
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            for detection in detections:
                x, y, width, height = detection.box
                x = frame_width - (x + width)
                recognized = detection.username is not None
                color = (
                    (80, 210, 130)
                    if recognized or self.show_face_guide
                    else (50, 100, 245)
                )
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                label = detection.username or (
                    "جودة منخفضة"
                    if detection.quality_message
                    else ("وجه مكتشف" if self.show_face_guide else "غير معروف")
                )
                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            zoom = float(self.digital_zoom_var.get())
            if zoom > 1:
                frame = self._zoom_preview(frame, detections, zoom)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            canvas_width = max(self.preview.winfo_width() - 20, 400)
            canvas_height = max(self.preview.winfo_height() - 20, 300)
            image.thumbnail((canvas_width, canvas_height))
            self.image_reference = ImageTk.PhotoImage(image=image)
            center_x = max(self.preview.winfo_width(), canvas_width) / 2
            center_y = max(self.preview.winfo_height(), canvas_height) / 2
            self.preview.itemconfigure(self.image_item, image=self.image_reference)
            self.preview.coords(self.image_item, center_x, center_y)
            self._draw_face_guide(center_x, center_y, image.width, image.height)
        if revision != self.last_detection_revision:
            self.last_detection_revision = revision
            self.on_detections(detections, revision)
        self.window.after(30, self._refresh)

    @staticmethod
    def _zoom_preview(
        frame: np.ndarray, detections: list[FaceDetection], zoom: float
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        crop_width = max(1, int(width / zoom))
        crop_height = max(1, int(height / zoom))
        center_x, center_y = width / 2, height / 2
        if detections:
            x, y, face_width, face_height = max(
                detections, key=lambda detection: detection.box[2] * detection.box[3]
            ).box
            center_x = x + face_width / 2
            center_y = y + face_height / 2
            center_x = width - center_x
        left = min(max(int(center_x - crop_width / 2), 0), width - crop_width)
        top = min(max(int(center_y - crop_height / 2), 0), height - crop_height)
        return frame[top : top + crop_height, left : left + crop_width]

    def _draw_face_guide(
        self, center_x: float, center_y: float, image_width: int, image_height: int
    ) -> None:
        if not self.show_face_guide:
            self.preview.itemconfigure(self.guide_item, state="hidden")
            self.preview.itemconfigure(self.guide_text_item, state="hidden")
            self.preview.itemconfigure(self.eye_line_item, state="hidden")
            return
        guide_width = image_width * 0.39
        guide_height = image_height * 0.72
        top = center_y - guide_height * 0.52
        self.preview.coords(
            self.guide_item,
            center_x - guide_width / 2,
            top,
            center_x + guide_width / 2,
            top + guide_height,
        )
        self.preview.coords(
            self.guide_text_item,
            center_x,
            max(center_y - image_height * 0.43, 20),
        )
        eye_line_y = top + guide_height * 0.39
        self.preview.coords(
            self.eye_line_item,
            center_x - guide_width * 0.35,
            eye_line_y,
            center_x + guide_width * 0.35,
            eye_line_y,
        )
        self.preview.itemconfigure(
            self.guide_item, outline=self.guide_color, state="normal"
        )
        self.preview.itemconfigure(self.eye_line_item, fill=self.guide_color, state="normal")
        self.preview.itemconfigure(
            self.guide_text_item, text=self.guide_hint, state="normal"
        )

    def on_detections(
        self, detections: list[FaceDetection], revision: int
    ) -> None:
        pass

    def on_stream_error(self, error: str) -> None:
        messagebox.showerror("خطأ في الكاميرا", error, parent=self.window)

    def close(self) -> None:
        if self.optical_zoom_after_id is not None:
            self.window.after_cancel(self.optical_zoom_after_id)
        self.stream.close()
        self.window.destroy()


class EnrollmentWindow(CameraWindow):
    """تسجيل موجّه يلتقط الصور تلقائيًا عند ثبات الوجه داخل الدليل."""

    def __init__(self, parent: tk.Misc, database: FaceDatabase) -> None:
        super().__init__(parent, database, "إضافة وجه جديد")
        self.window.geometry("1240x900")
        self.window.minsize(1050, 780)
        self.window.columnconfigure(1, weight=1)
        self.captured_frames: list[np.ndarray] = []
        self.result_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.stable_frames = 0
        self.previous_box: tuple[int, int, int, int] | None = None
        self.countdown_value = 0
        self.countdown_after_id: str | None = None
        self.step_after_id: str | None = None
        self.show_face_guide = True
        self.guide_color = COLORS["warning"]
        self.awaiting_next_step = False

        self.side_panel = tk.Frame(
            self.window, bg=COLORS["surface"], padx=18, pady=18
        )
        self.side_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 16), pady=16)
        self.side_panel.columnconfigure(0, weight=1)
        tk.Label(
            self.side_panel,
            text="تسجيل وجه جديد",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Arial", 17, "bold"),
        ).grid(row=0, column=0, sticky="e", pady=(0, 8))
        tk.Label(
            self.side_panel,
            text=(
                "اتبع التعليمات؛ سيتم التقاط الصورة تلقائيًا عند ثبات الوجه. "
                "سجّل في الإضاءة والمسافة المعتادتين للاستخدام."
            ),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            justify="right",
            wraplength=270,
        ).grid(row=1, column=0, sticky="ew", pady=(0, 14))
        self.enrollment_help = ExpandableHelp(
            self.side_panel,
            "شرح التسجيل وجودة الوجه",
            (
                "اتبع وضعيات الوجه الخمس، وانتظر الالتقاط التلقائي. "
                "يجب أن يكون الوجه واضحًا ومضيئًا وبحجم مناسب؛ التقريب الرقمي "
                "لا يضيف تفاصيل. سجّل في ظروف قريبة من المسافة والإضاءة "
                "المتوقعة عند الفحص. تُحفظ بيانات المستخدم وتمثيلات الوجه "
                "محليًا؛ صور اللقطات الخام لا تُضاف لملف قاعدة البيانات. "
                "التسجيل القديم لا يُستبدل إلا بعد تفعيل الخيار وتأكيد العملية."
            ),
        )
        self.enrollment_help.frame.grid(
            row=2, column=0, sticky="ew", pady=(0, 8)
        )
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.side_panel,
            variable=self.progress_var,
            maximum=CAPTURE_COUNT,
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress_bar.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.step_var = tk.StringVar(value="اللقطة 1 من 5 — أمام")
        tk.Label(
            self.side_panel,
            textvariable=self.step_var,
            bg=COLORS["surface"],
            fg=COLORS["accent"],
            font=("Arial", 12, "bold"),
            anchor="e",
        ).grid(row=4, column=0, sticky="ew", pady=(0, 4))
        self.pose_var = tk.StringVar(value=ENROLLMENT_STEPS[0][1])
        tk.Label(
            self.side_panel,
            textvariable=self.pose_var,
            bg=COLORS["surface"],
            fg=COLORS["warning"],
            font=("Arial", 11, "bold"),
            anchor="e",
            justify="right",
            wraplength=270,
        ).grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.status_var = tk.StringVar(value="ضع وجهك داخل الإطار وانتظر.")
        tk.Label(
            self.side_panel,
            textvariable=self.status_var,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            justify="right",
            wraplength=270,
            anchor="e",
        ).grid(row=6, column=0, sticky="ew", pady=(0, 12))
        self.shot_indicators: list[tk.Label] = []
        indicators = tk.Frame(self.side_panel, bg=COLORS["surface"])
        indicators.grid(row=7, column=0, sticky="ew", pady=(0, 18))
        for index, (short_name, _) in enumerate(ENROLLMENT_STEPS):
            indicator = tk.Label(
                indicators,
                text=f"{index + 1}\n{short_name}",
                bg=COLORS["surface_light"],
                fg=COLORS["muted"],
                width=6,
                height=3,
                font=("Arial", 9, "bold"),
            )
            indicator.pack(side="right", padx=2)
            self.shot_indicators.append(indicator)
        tk.Label(
            self.side_panel,
            text="بيانات المستخدم",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Arial", 12, "bold"),
            anchor="e",
        ).grid(row=8, column=0, sticky="ew", pady=(0, 8))

        self.name_var = tk.StringVar()
        self.username_var = tk.StringVar()
        self.phone_var = tk.StringVar()
        self.email_var = tk.StringVar()
        self.notes_var = tk.StringVar()
        self.consent_var = tk.BooleanVar(value=False)
        self.update_existing_var = tk.BooleanVar(value=False)
        fields = [
            ("الاسم الكامل", self.name_var),
            ("اسم المستخدم (فريد)", self.username_var),
            ("رقم الهاتف", self.phone_var),
            ("البريد الإلكتروني", self.email_var),
            ("بيانات إضافية", self.notes_var),
        ]
        for index, (label, variable) in enumerate(fields, start=9):
            tk.Label(
                self.side_panel,
                text=label,
                bg=COLORS["surface"],
                fg=COLORS["muted"],
                anchor="e",
            ).grid(row=index * 2, column=0, sticky="ew", pady=(3, 0))
            ttk.Entry(self.side_panel, textvariable=variable).grid(
                row=index * 2 + 1, column=0, sticky="ew", pady=(0, 4)
            )

        tk.Checkbutton(
            self.side_panel,
            text="أوافق على حفظ بياناتي وتمثيل وجهي محليًا لهذا التطبيق، ويمكنني طلب حذفها.",
            variable=self.consent_var,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            activebackground=COLORS["surface"],
            activeforeground=COLORS["text"],
            selectcolor=COLORS["surface_light"],
            justify="right",
            wraplength=270,
        ).grid(row=28, column=0, sticky="ew", pady=(8, 0))
        tk.Checkbutton(
            self.side_panel,
            text="استبدال التسجيل الحالي بهذا التسجيل (يشمل التمثيلين)",
            variable=self.update_existing_var,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            activebackground=COLORS["surface"],
            activeforeground=COLORS["text"],
            selectcolor=COLORS["surface_light"],
            justify="right",
            wraplength=270,
        ).grid(row=29, column=0, sticky="ew", pady=(4, 0))

        controls = tk.Frame(self.window, bg=COLORS["background"])
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 14))
        self.capture_button = ttk.Button(
            controls, text="التقاط يدوي", command=self.capture_image
        )
        self.capture_button.pack(side="right", padx=(8, 0))
        self.save_button = ttk.Button(
            controls, text="حفظ التسجيل", command=self.save_enrollment, state="disabled"
        )
        self.save_button.pack(side="right", padx=6)
        self.stream.set_detection_enabled(True)
        self.window.after(100, self._check_result)

    def on_stream_error(self, error: str) -> None:
        self.status_var.set(error)

    def on_detections(
        self, detections: list[FaceDetection], revision: int
    ) -> None:
        if (
            self.awaiting_next_step
            or len(self.captured_frames) >= CAPTURE_COUNT
        ):
            return
        if len(detections) != 1:
            self._reset_stability()
            self.status_var.set(
                "لم يتم العثور على وجه واضح." if not detections
                else "يجب أن يظهر شخص واحد فقط أمام الكاميرا."
            )
            self.guide_hint = "ضع وجهًا واحدًا داخل الإطار"
            self.guide_color = COLORS["warning"]
            return

        frame, _, _, _ = self.stream.snapshot()
        if frame is None:
            return
        box = detections[0].box
        quality_message = assess_face_quality(frame, box)
        if quality_message:
            self._reset_stability()
            self.status_var.set(quality_message)
            self.guide_hint = "اقترب وحسّن الإضاءة وثبّت الكاميرا"
            self.guide_color = COLORS["warning"]
            return
        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = box
        center_x = x + width / 2
        center_y = y + height / 2
        aligned = (
            0.28 * frame_width <= center_x <= 0.72 * frame_width
            and 0.20 * frame_height <= center_y <= 0.80 * frame_height
            and 0.20 * frame_height <= height <= 0.78 * frame_height
        )
        if not aligned:
            self._reset_stability()
            self.status_var.set("قرّب وجهك قليلًا واجعله في منتصف الإطار.")
            self.guide_hint = "اقترب واجعل وجهك في المنتصف"
            self.guide_color = COLORS["warning"]
            return

        self.guide_hint = "ممتاز — حافظ على وضعيتك"
        self.guide_color = COLORS["success"]
        if self.previous_box is not None and self._is_stable(box, self.previous_box):
            self.stable_frames += 1
        else:
            self._cancel_countdown()
            self.stable_frames = 1
        self.previous_box = box

        if self.stable_frames >= 3 and self.countdown_after_id is None:
            _, instruction = ENROLLMENT_STEPS[len(self.captured_frames)]
            self.status_var.set(f"{instruction} — ثبّت وجهك، سيبدأ الالتقاط تلقائيًا.")
            self._start_countdown()
        elif self.countdown_after_id is None:
            self.status_var.set("اكتشاف الوجه... ابقَ ثابتًا قليلًا لبدء الالتقاط.")

    @staticmethod
    def _is_stable(
        current: tuple[int, int, int, int], previous: tuple[int, int, int, int]
    ) -> bool:
        x, y, width, height = current
        old_x, old_y, old_width, old_height = previous
        current_center = (x + width / 2, y + height / 2)
        previous_center = (old_x + old_width / 2, old_y + old_height / 2)
        return (
            abs(current_center[0] - previous_center[0]) <= width * 0.06
            and abs(current_center[1] - previous_center[1]) <= height * 0.06
            and abs(height - old_height) <= height * 0.12
        )

    def _reset_stability(self) -> None:
        self.stable_frames = 0
        self.previous_box = None
        self._cancel_countdown()

    def _start_countdown(self) -> None:
        self.countdown_value = 3
        self._countdown_tick()

    def _countdown_tick(self) -> None:
        if self.countdown_value <= 0:
            self.countdown_after_id = None
            self._capture_automatic()
            return
        self.status_var.set(f"التقاط تلقائي بعد {self.countdown_value}...")
        self.countdown_value -= 1
        self.countdown_after_id = self.window.after(800, self._countdown_tick)

    def _cancel_countdown(self) -> None:
        if self.countdown_after_id is not None:
            self.window.after_cancel(self.countdown_after_id)
            self.countdown_after_id = None

    def _capture_automatic(self) -> None:
        self.countdown_after_id = None
        frame, detections, _, _ = self.stream.snapshot()
        if frame is None or len(detections) != 1 or self.stable_frames < 3:
            self._reset_stability()
            self.status_var.set("تحرك الوجه؛ أعد التمركز للمتابعة.")
            return
        quality_message = assess_face_quality(frame, detections[0].box)
        if quality_message:
            self._reset_stability()
            self.status_var.set(quality_message)
            return
        self._store_capture(frame)

    def _store_capture(self, frame: np.ndarray) -> None:
        if len(self.captured_frames) >= CAPTURE_COUNT:
            return
        index = len(self.captured_frames)
        self.captured_frames.append(frame.copy())
        self.progress_var.set(len(self.captured_frames))
        self.shot_indicators[index].configure(
            bg=COLORS["success"], fg=COLORS["background"]
        )
        self.window.bell()
        self._reset_stability()
        if len(self.captured_frames) == CAPTURE_COUNT:
            self.guide_hint = "اكتملت الصور الخمس"
            self.status_var.set("اكتملت اللقطات الخمس. أدخل البيانات ثم احفظ التسجيل.")
            self.capture_button.configure(state="disabled")
            self.save_button.configure(state="normal")
            return

        next_index = len(self.captured_frames)
        self.awaiting_next_step = True
        short_name, instruction = ENROLLMENT_STEPS[next_index]
        self.step_var.set(
            f"اللقطة {next_index + 1} من {CAPTURE_COUNT} — {short_name}"
        )
        self.pose_var.set(instruction)
        self.status_var.set(f"تم التقاط الصورة بنجاح. التالية: {instruction}")
        self.guide_hint = instruction
        self.step_after_id = self.window.after(1500, self._allow_next_capture)

    def _allow_next_capture(self) -> None:
        self.step_after_id = None
        self.awaiting_next_step = False
        self.status_var.set("نفّذ الوضعية المطلوبة ثم اثبت قليلًا للالتقاط التلقائي.")

    def capture_image(self) -> None:
        frame, detections, _, _ = self.stream.snapshot()
        if frame is None:
            self.status_var.set("في انتظار أول إطار من الكاميرا...")
            return
        if len(detections) != 1:
            self.status_var.set("لا يمكن الالتقاط اليدوي قبل اكتشاف وجه واحد.")
            return
        quality_message = assess_face_quality(frame, detections[0].box)
        if quality_message:
            self.status_var.set(quality_message)
            return
        self._cancel_countdown()
        self._store_capture(frame)

    def save_enrollment(self) -> None:
        username = self.username_var.get().strip()
        name = self.name_var.get().strip()
        if not self.consent_var.get():
            messagebox.showwarning(
                "الموافقة مطلوبة",
                "يجب الحصول على موافقة صريحة قبل حفظ بيانات الوجه.",
                parent=self.window,
            )
            return
        if not username or not name:
            messagebox.showwarning(
                "بيانات ناقصة", "أدخل الاسم الكامل واسم المستخدم.", parent=self.window
            )
            return
        if len(self.captured_frames) != CAPTURE_COUNT:
            messagebox.showwarning(
                "لقطات ناقصة", f"التقط {CAPTURE_COUNT} صور أولًا.", parent=self.window
            )
            return
        replacing_existing = self.database.contains(username)
        if replacing_existing and not self.update_existing_var.get():
            messagebox.showerror(
                "اسم مستخدم مكرر",
                "فعّل خيار استبدال التسجيل الحالي إذا كنت تريد تحديثه.",
                parent=self.window,
            )
            return
        if replacing_existing and not messagebox.askyesno(
            "تأكيد تحديث التسجيل",
            f"سيتم استبدال تمثيلات الوجه الحالية لـ {username} بعد اكتمال التحليل. هل تريد المتابعة؟",
            parent=self.window,
        ):
            return

        details = {
            "name": name,
            "username": username,
            "phone": self.phone_var.get().strip(),
            "email": self.email_var.get().strip(),
            "notes": self.notes_var.get().strip(),
        }
        frames = [frame.copy() for frame in self.captured_frames]
        self.capture_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.status_var.set("يتم استخراج تمثيلات الوجه وحفظها...")
        self.stream.set_detection_enabled(False)
        threading.Thread(
            target=self._enroll_worker,
            args=(username, details, frames, replacing_existing),
            name="face-enrollment",
            daemon=True,
        ).start()

    def _enroll_worker(
        self,
        username: str,
        details: dict[str, str],
        frames: list[np.ndarray],
        replacing_existing: bool,
    ) -> None:
        try:
            engine = self.stream.get_engine()
            upper_embeddings = []
            full_embeddings = []
            for index, frame in enumerate(frames, start=1):
                with engine.inference_lock:
                    upper, full, _ = engine.embeddings_from_frame(frame)
                upper_embeddings.append(upper)
                full_embeddings.append(full)
                self.result_queue.put(
                    ("progress", f"تم تحليل الصورة {index} من {CAPTURE_COUNT}...")
                )
            if replacing_existing:
                self.database.update_user(
                    username, details, upper_embeddings, full_embeddings
                )
                success_message = f"تم تحديث تسجيل المستخدم {username} بنجاح."
            else:
                self.database.add_user(
                    username, details, upper_embeddings, full_embeddings
                )
                success_message = f"تم تسجيل المستخدم {username} بنجاح."
            self.result_queue.put(("success", success_message))
        except Exception as exc:
            self.result_queue.put(("error", f"تعذر إكمال التسجيل: {exc}"))

    def _check_result(self) -> None:
        if not self.window.winfo_exists():
            return
        try:
            result_type, text = self.result_queue.get_nowait()
        except queue.Empty:
            self.window.after(100, self._check_result)
            return
        self.status_var.set(text)
        if result_type == "success":
            messagebox.showinfo("اكتمل التسجيل", text, parent=self.window)
            self.close()
            return
        if result_type == "error":
            messagebox.showerror("خطأ في التسجيل", text, parent=self.window)
            self.save_button.configure(state="normal")
            if len(self.captured_frames) < CAPTURE_COUNT:
                self.capture_button.configure(state="normal")
                self.stream.set_detection_enabled(True)
        self.window.after(100, self._check_result)

    def close(self) -> None:
        self._cancel_countdown()
        if self.step_after_id is not None:
            self.window.after_cancel(self.step_after_id)
        super().close()


class RegisteredUsersWindow:
    """عرض المستخدمين المسجلين والسماح بحذف بياناتهم محليًا."""

    def __init__(self, parent: tk.Misc, database: FaceDatabase) -> None:
        self.database = database
        self.window = tk.Toplevel(parent)
        self.window.title("إدارة الوجوه المسجلة")
        self.window.geometry("760x500")
        self.window.minsize(620, 400)
        self.window.configure(bg=COLORS["background"])
        self.window.transient(parent)

        container = ttk.Frame(self.window, padding=16)
        container.pack(fill="both", expand=True)
        ttk.Label(
            container,
            text="البيانات محفوظة محليًا على هذا الجهاز.",
            wraplength=460,
            justify="right",
        ).pack(fill="x", pady=(0, 10))
        self.users_view = ttk.Treeview(
            container,
            columns=("username", "name", "full"),
            show="headings",
            selectmode="browse",
        )
        self.users_view.heading("username", text="اسم المستخدم")
        self.users_view.heading("name", text="الاسم")
        self.users_view.heading("full", text="تمثيل كامل")
        self.users_view.column("username", width=190, anchor="center")
        self.users_view.column("name", width=270, anchor="e")
        self.users_view.column("full", width=150, anchor="center")
        self.users_view.pack(fill="both", expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(12, 0))
        ttk.Button(
            controls, text="حذف المحدد", command=self.delete_selected
        ).pack(side="right", padx=(0, 8))
        ttk.Button(controls, text="إغلاق", command=self.window.destroy).pack(
            side="right"
        )
        self.refresh()

    def refresh(self) -> None:
        for item in self.users_view.get_children():
            self.users_view.delete(item)
        for username, name in self.database.list_users():
            full_status = (
                "جاهز للمقارنة"
                if self.database.has_full_embeddings(username)
                else "يحتاج تحديثًا"
            )
            self.users_view.insert(
                "", "end", iid=username, values=(username, name, full_status)
            )

    def delete_selected(self) -> None:
        selection = self.users_view.selection()
        if not selection:
            messagebox.showwarning(
                "لم يتم التحديد", "حدد مستخدمًا لحذفه.", parent=self.window
            )
            return

        username = selection[0]
        if not messagebox.askyesno(
            "تأكيد الحذف",
            f"سيتم حذف بيانات الوجه ومعلومات {username} نهائيًا. هل تريد المتابعة؟",
            parent=self.window,
        ):
            return
        try:
            self.database.delete_user(username)
        except Exception as exc:
            messagebox.showerror(
                "تعذر الحذف", f"لم يتم حذف بيانات المستخدم: {exc}", parent=self.window
            )
            return
        self.refresh()


class ScanWindow(CameraWindow):
    """نافذة الفحص مع لوحة جانبية تعرض معلومات الشخص عند التعرف عليه."""

    def __init__(self, parent: tk.Misc, database: FaceDatabase) -> None:
        super().__init__(parent, database, "تسجيل الدخول / فحص")
        self.window.geometry("1120x650")
        self.preview.grid_configure(row=1, column=0, sticky="nsew", padx=12, pady=12)
        self.window.columnconfigure(0, weight=3)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(1, weight=1)

        self.side_panel = tk.Frame(
            self.window, bg=COLORS["surface"], padx=20, pady=20
        )
        self.side_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 12), pady=12)
        tk.Label(
            self.side_panel,
            text="نتيجة الفحص",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Arial", 17, "bold"),
        ).pack(anchor="e", fill="x", pady=(0, 12))
        ttk.Label(self.side_panel, text="طريقة مقارنة الوجه").pack(
            anchor="e", fill="x", pady=(0, 4)
        )
        self.representation_var = tk.StringVar(value="قارن الطريقتين")
        self.representation_modes = {
            "الجزء العلوي فقط": "upper",
            "الوجه كاملًا": "full",
            "قارن الطريقتين": "compare",
        }
        self.representation_selector = ttk.Combobox(
            self.side_panel,
            textvariable=self.representation_var,
            values=tuple(self.representation_modes),
            state="readonly",
        )
        self.representation_selector.pack(anchor="e", fill="x", pady=(0, 8))
        self.representation_selector.bind(
            "<<ComboboxSelected>>", self._change_representation
        )
        self.representation_hint_var = tk.StringVar()
        ttk.Label(
            self.side_panel,
            textvariable=self.representation_hint_var,
            wraplength=250,
            justify="right",
        ).pack(anchor="e", fill="x", pady=(0, 3))
        self.comparison_help = ExpandableHelp(
            self.side_panel,
            "ما معنى هذه الخيارات؟",
            (
                "الجزء العلوي فقط: يطمس البرنامج أسفل الوجه قبل المقارنة. "
                "الوجه كاملًا: يستخدم صورة الوجه دون طمس. قارن الطريقتين: "
                "يعرض نتيجة كل طريقة على الوجه نفسه للاستعراض، وليس تسجيل دخول. "
                "الرقم هو مسافة تشابه، "
                "وليس نسبة مئوية؛ الأقل أقرب، ويُقبل التطابق حاليًا عند 0.45 "
                "أو أقل. اختلاف المسافة بين الطريقتين لا يثبت وحده أيهما أدق.\n\n"
                "للمقارنة: حدّث تسجيل الشخص المعروف ليحفظ الطريقتين، ثم اختبره "
                "في إضاءات ومسافات متعددة، وسجّل أيضًا نتائج أشخاص غير مسجلين. "
                "التزم بالشروط نفسها ولا تعتبر اتفاق الطريقتين إثباتًا للهوية.\n\n"
                "الوجوه القديمة يمكن مقارنتها بالطريقة العلوية فقط إلى أن "
                "تُحدّث تسجيلاتها. بيانات التسجيل محفوظة محليًا في ملف قاعدة "
                "البيانات وليست مشفّرة؛ لا تشارك الملف. هذا الفحص تجريبي، "
                "ولا يكفي وحده لتسجيل دخول آمن."
            ),
        )
        self.comparison_help.frame.pack(anchor="e", fill="x", pady=(0, 6))
        self.full_embedding_status_var = tk.StringVar()
        ttk.Label(
            self.side_panel,
            textvariable=self.full_embedding_status_var,
            wraplength=250,
            justify="right",
        ).pack(anchor="e", fill="x", pady=(0, 8))
        self.status_var = tk.StringVar(value="يتم الفحص... انظر إلى الكاميرا.")
        tk.Label(
            self.side_panel,
            textvariable=self.status_var,
            wraplength=250,
            justify="right",
            bg=COLORS["surface"],
            fg=COLORS["accent"],
            font=("Arial", 12, "bold"),
        ).pack(anchor="e", fill="x", pady=(0, 16))
        self.info_var = tk.StringVar(value="لم يتم التعرف على مستخدم بعد.")
        tk.Label(
            self.side_panel,
            textvariable=self.info_var,
            wraplength=250,
            justify="right",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Arial", 11),
        ).pack(anchor="e", fill="x")
        self.stream.set_scan_enabled(True)
        self.stream.set_representation("compare")
        self.current_username: str | None = None
        self.candidate_username: str | None = None
        self.candidate_frames = 0
        self.comparison_stats = {
            "frames": 0,
            "upper_matches": 0,
            "full_matches": 0,
            "agreements": 0,
        }
        self._update_full_embedding_status()
        self._update_representation_hint()

    def _change_representation(self, _event: tk.Event | None = None) -> None:
        mode = self.representation_modes[self.representation_var.get()]
        self.stream.set_representation(mode)
        self.candidate_username = None
        self.candidate_frames = 0
        self.current_username = None
        self.comparison_stats = {
            "frames": 0,
            "upper_matches": 0,
            "full_matches": 0,
            "agreements": 0,
        }
        self.info_var.set("انتظار نتائج الوضع المختار...")
        self._update_representation_hint()

    def _update_representation_hint(self) -> None:
        mode = self.representation_modes[self.representation_var.get()]
        hints = {
            "upper": (
                "يطمس أسفل الوجه قبل المقارنة. مناسب للسجلات القديمة "
                "والتي لم تُحدّث بعد."
            ),
            "full": (
                "يقارن الوجه دون طمس. يلزم تسجيل محدّث يحتوي على تمثيل كامل."
            ),
            "compare": (
                "يعرض نتيجة الطريقتين للاستعراض، ولا يؤكد تسجيل الدخول. "
                "حدّث التسجيلات القديمة أولًا ثم اختبر أشخاصًا معروفين وغير مسجلين."
            ),
        }
        self.representation_hint_var.set(hints[mode])

    def _update_full_embedding_status(self) -> None:
        total = len(self.database.list_users())
        full_count = self.database.count_users_with_full_embeddings()
        self.full_embedding_status_var.set(
            f"سجلات الوجه الكامل الجاهزة للمقارنة: {full_count}/{total}. "
            "يمكن تحديث السجل من شاشة إضافة وجه جديد."
        )

    def on_detections(
        self, detections: list[FaceDetection], revision: int
    ) -> None:
        comparison_mode = (
            self.representation_modes[self.representation_var.get()] == "compare"
        )
        if (
            comparison_mode
            and len(detections) == 1
            and detections[0].quality_message is None
        ):
            detection = detections[0]
            self.comparison_stats["frames"] += 1
            self.comparison_stats["upper_matches"] += int(
                detection.username is not None
            )
            self.comparison_stats["full_matches"] += int(
                detection.alternate_username is not None
            )
            self.comparison_stats["agreements"] += int(
                detection.username is not None
                and detection.username == detection.alternate_username
            )

        recognized = next(
            (detection for detection in detections if detection.username is not None), None
        )
        if recognized is None:
            self.candidate_username = None
            self.candidate_frames = 0
            self.current_username = None
            quality_message = next(
                (
                    detection.quality_message
                    for detection in detections
                    if detection.quality_message
                ),
                None,
            )
            comparison = (
                detections[0]
                if comparison_mode and len(detections) == 1
                else None
            )
            if comparison is not None:
                if comparison.alternate_username:
                    self.status_var.set(
                        "التمثيل العلوي لم يطابق؛ "
                        f"الكامل طابق {comparison.alternate_username}."
                    )
                elif comparison.quality_message:
                    self.status_var.set(comparison.quality_message)
                else:
                    self.status_var.set("لم يطابق أي من التمثيلين وجهًا مسجلًا.")
                self.info_var.set(self._comparison_text(comparison))
                return
            self.status_var.set(
                quality_message
                or (
                    "لم يتم التعرف على وجه مسجل."
                    if detections
                    else "انظر إلى الكاميرا."
                )
            )
            if (
                self.representation_modes[self.representation_var.get()] == "full"
                and self.database.count_users_with_full_embeddings()
                < len(self.database.list_users())
            ):
                self.status_var.set(
                    "بعض التسجيلات تحتاج تحديثًا لتضمين تمثيل الوجه الكامل."
                )
            self.info_var.set("لم يتم التعرف على مستخدم بعد.")
            return

        username = recognized.username
        if username == self.candidate_username:
            self.candidate_frames += 1
        else:
            self.candidate_username = username
            self.candidate_frames = 1
        if self.candidate_frames < 3:
            self.status_var.set(
                f"جارٍ تثبيت المطابقة... {self.candidate_frames}/3"
            )
            return
        self.status_var.set(f"تم التعرف على: {username}")
        if username != self.current_username:
            self.current_username = username
            details = self.database.get_details(username)
            lines = [
                f"الاسم: {details.get('name', '')}",
                f"اسم المستخدم: {username}",
            ]
            for label, key in (
                ("الهاتف", "phone"),
                ("البريد الإلكتروني", "email"),
                ("معلومات إضافية", "notes"),
            ):
                if details.get(key):
                    lines.append(f"{label}: {details[key]}")
            if recognized.distance is not None:
                lines.append(f"مسافة المطابقة: {recognized.distance:.3f}")
            self.info_var.set("\n".join(lines))
        if comparison_mode:
            self.info_var.set(self._comparison_text(recognized))

    def _comparison_text(self, detection: FaceDetection) -> str:
        lines = []
        if detection.username and detection.alternate_username:
            agreement = (
                "متفقتان"
                if detection.username == detection.alternate_username
                else "مختلفتان"
            )
            lines.append(f"النتيجتان {agreement}.")
        else:
            lines.append("توجد مطابقة في تمثيل واحد فقط أو لا توجد.")
        if detection.username is not None:
            distance = (
                f"{detection.distance:.3f}"
                if detection.distance is not None
                else "—"
            )
            lines.append(
                f"التمثيل العلوي: {detection.username} "
                f"({distance})"
            )
        else:
            lines.append("التمثيل العلوي: لا توجد مطابقة")
        if detection.alternate_username is not None:
            alternate_distance = (
                f"{detection.alternate_distance:.3f}"
                if detection.alternate_distance is not None
                else "—"
            )
            lines.append(
                f"الوجه الكامل: {detection.alternate_username} "
                f"({alternate_distance})"
            )
        else:
            lines.append("الوجه الكامل: لا توجد مطابقة/تسجيل كامل")
        stats = self.comparison_stats
        if stats["frames"]:
            lines.append(
                "ملاحظات الجلسة (ليست مقياس دقة): "
                f"{stats['frames']} إطار، "
                f"علوي {stats['upper_matches']}، "
                f"كامل {stats['full_matches']}، "
                f"اتفاق {stats['agreements']}."
            )
        return "\n".join(lines)

    def on_stream_error(self, error: str) -> None:
        self.status_var.set(error)


class FaceRecognitionApp:
    """النافذة الرئيسية للتطبيق."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("نظام التعرف على الوجه")
        self.root.geometry("1120x760")
        self.root.minsize(920, 650)
        self.database = FaceDatabase(DATABASE_PATH)
        self.root.configure(bg=COLORS["background"])
        self.users_count_var = tk.StringVar()
        self.full_embeddings_count_var = tk.StringVar()
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["background"])
        style.configure(
            "TLabel",
            background=COLORS["background"],
            foreground=COLORS["text"],
            font=("Arial", 11),
        )
        style.configure(
            "TButton",
            font=("Arial", 11, "bold"),
            padding=(16, 12),
            background=COLORS["surface_light"],
            foreground=COLORS["text"],
            borderwidth=0,
        )
        style.map(
            "TButton",
            background=[
                ("active", COLORS["accent"]),
                ("disabled", COLORS["surface"]),
            ],
            foreground=[("active", COLORS["background"])],
        )
        style.configure(
            "Primary.TButton",
            background=COLORS["accent"],
            foreground=COLORS["background"],
            font=("Arial", 11, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#7dd3fc"), ("disabled", COLORS["surface"])],
        )
        style.configure(
            "TEntry",
            padding=7,
            fieldbackground=COLORS["surface_light"],
            foreground=COLORS["text"],
        )
        style.configure(
            "Treeview",
            background=COLORS["surface"],
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            rowheight=30,
        )
        style.configure(
            "Treeview.Heading",
            background=COLORS["surface_light"],
            foreground=COLORS["text"],
            font=("Arial", 10, "bold"),
        )
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=COLORS["surface_light"],
            background=COLORS["accent"],
        )

        shell = tk.Frame(root, bg=COLORS["background"])
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        sidebar = tk.Frame(shell, bg=COLORS["surface"], width=230, padx=18, pady=24)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        tk.Label(
            sidebar,
            text="FACE\nSCANNER",
            bg=COLORS["surface"],
            fg=COLORS["accent"],
            font=("Arial", 18, "bold"),
            justify="left",
        ).pack(anchor="w", pady=(4, 28))
        tk.Label(
            sidebar,
            text="مساحة العمل",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Button(
            sidebar,
            text="بدء الفحص",
            style="Primary.TButton",
            command=self.open_scan,
        ).pack(fill="x", pady=5)
        ttk.Button(
            sidebar,
            text="إضافة / تحديث وجه",
            command=self.open_enrollment,
        ).pack(fill="x", pady=5)
        ttk.Button(
            sidebar,
            text="إدارة المسجلين",
            command=self.open_user_management,
        ).pack(fill="x", pady=5)
        tk.Frame(sidebar, bg=COLORS["surface_light"], height=1).pack(
            fill="x", pady=20
        )
        tk.Label(
            sidebar,
            text="ArcFace  •  RetinaFace\nمعالجة محلية على الجهاز",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            justify="left",
            font=("Arial", 9),
        ).pack(anchor="w", side="bottom")

        content = tk.Frame(shell, bg=COLORS["background"], padx=32, pady=26)
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        header = tk.Frame(content, bg=COLORS["background"])
        header.grid(row=0, column=0, sticky="ew", pady=(0, 28))
        tk.Label(
            header,
            text="لوحة التحكم",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Arial", 25, "bold"),
        ).pack(anchor="e")
        tk.Label(
            header,
            text="إدارة التسجيلات ومقارنة أداء التعرّف على الوجه",
            bg=COLORS["background"],
            fg=COLORS["muted"],
            font=("Arial", 11),
        ).pack(anchor="e", pady=(5, 0))

        metrics = tk.Frame(content, bg=COLORS["background"])
        metrics.grid(row=1, column=0, sticky="ew", pady=(0, 24))
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        self._create_metric_card(
            metrics, 0, "المستخدمون المسجلون", self.users_count_var, COLORS["accent"]
        )
        self._create_metric_card(
            metrics,
            1,
            "جاهزون لمقارنة الوجه الكامل",
            self.full_embeddings_count_var,
            COLORS["success"],
        )

        tk.Label(
            content,
            text="اختصارات",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Arial", 15, "bold"),
        ).grid(row=2, column=0, sticky="e", pady=(0, 12))
        actions = tk.Frame(content, bg=COLORS["background"])
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1, uniform="actions")
        actions.columnconfigure(1, weight=1, uniform="actions")
        self._create_action_card(
            actions,
            0,
            "فحص ومقارنة",
            "ابدأ الكاميرا، واختر مقارنة التمثيل العلوي بالوجه الكامل.",
            "فتح الفحص",
            self.open_scan,
        )
        self._create_action_card(
            actions,
            1,
            "تسجيل وجه",
            "التقط عينات جديدة واحفظ تمثيلي الوجه للمقارنة العادلة.",
            "إضافة تسجيل",
            self.open_enrollment,
        )

        notice = tk.Frame(
            content,
            bg="#2a2517",
            padx=16,
            pady=13,
            highlightbackground="#725b1c",
            highlightthickness=1,
        )
        notice.grid(row=4, column=0, sticky="ew", pady=(25, 0))
        tk.Label(
            notice,
            text=(
                "ملاحظة: مقارنة التمثيلين أداة تقييم وليست إثباتًا للدقة أو "
                "وسيلة تحقق آمنة. اختبر النتائج على أشخاص ومسافات متعددة."
            ),
            bg="#2a2517",
            fg=COLORS["warning"],
            justify="right",
            wraplength=760,
        ).pack(fill="x")
        self.refresh_dashboard()

    @staticmethod
    def _create_metric_card(
        parent: tk.Misc,
        column: int,
        title: str,
        value: tk.StringVar,
        accent: str,
    ) -> None:
        card = tk.Frame(
            parent,
            bg=COLORS["surface"],
            padx=18,
            pady=15,
            highlightbackground=COLORS["surface_light"],
            highlightthickness=1,
        )
        card.grid(row=0, column=column, sticky="ew", padx=(0, 12))
        tk.Label(
            card, text=title, bg=COLORS["surface"], fg=COLORS["muted"]
        ).pack(anchor="e")
        tk.Label(
            card,
            textvariable=value,
            bg=COLORS["surface"],
            fg=accent,
            font=("Arial", 22, "bold"),
        ).pack(anchor="e", pady=(5, 0))

    def _create_action_card(
        self,
        parent: tk.Misc,
        column: int,
        title: str,
        description: str,
        button_text: str,
        command: Any,
    ) -> None:
        card = tk.Frame(
            parent,
            bg=COLORS["surface"],
            padx=20,
            pady=20,
            highlightbackground=COLORS["surface_light"],
            highlightthickness=1,
        )
        card.grid(row=0, column=column, sticky="nsew", padx=(0, 12))
        tk.Label(
            card,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Arial", 15, "bold"),
        ).pack(anchor="e")
        tk.Label(
            card,
            text=description,
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            justify="right",
            wraplength=330,
            height=3,
        ).pack(fill="x", pady=(10, 18))
        ttk.Button(card, text=button_text, command=command).pack(
            anchor="e", fill="x"
        )

    def refresh_dashboard(self) -> None:
        self.users_count_var.set(str(len(self.database.list_users())))
        self.full_embeddings_count_var.set(
            str(self.database.count_users_with_full_embeddings())
        )

    def open_scan(self) -> None:
        self._open_camera_window(ScanWindow)

    def open_enrollment(self) -> None:
        self._open_camera_window(EnrollmentWindow)

    def open_user_management(self) -> None:
        window = RegisteredUsersWindow(self.root, self.database)
        window.window.bind(
            "<Destroy>",
            lambda event: self.refresh_dashboard()
            if event.widget is window.window
            else None,
        )
        window.window.grab_set()

    def _open_camera_window(self, window_type: type[CameraWindow]) -> None:
        try:
            child = window_type(self.root, self.database)
        except Exception as exc:
            messagebox.showerror("تعذر فتح الكاميرا", str(exc), parent=self.root)
            return
        child.window.bind(
            "<Destroy>",
            lambda event: self.refresh_dashboard()
            if event.widget is child.window
            else None,
        )
        child.window.transient(self.root)
        child.window.grab_set()


def main() -> None:
    root = tk.Tk()
    try:
        FaceRecognitionApp(root)
    except Exception as exc:
        messagebox.showerror("تعذر بدء التطبيق", str(exc), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
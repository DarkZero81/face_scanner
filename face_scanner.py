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
# عتبة DeepFace الافتراضية التقريبية لـ ArcFace مع مسافة cosine.
COSINE_DISTANCE_THRESHOLD = 0.68
CAPTURE_COUNT = 5
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
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """يستخرج embedding لوجه واحد من صورة كاميرا خام."""
        faces = self._extract_faces(frame)
        if len(faces) != 1:
            raise ValueError("يجب أن يظهر وجه واحد فقط في الصورة.")

        face = faces[0]
        face_image = self._upper_face_image(face["face"])
        representations = self.deepface.represent(
            img_path=face_image,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        if not representations:
            raise RuntimeError("لم يتمكن ArcFace من إنشاء تمثيل الوجه.")

        area = face.get("facial_area", {})
        box = (
            int(area.get("x", 0)),
            int(area.get("y", 0)),
            int(area.get("w", 0)),
            int(area.get("h", 0)),
        )
        return np.asarray(representations[0]["embedding"], dtype=np.float32), box

    def analyze_frame(self, frame: np.ndarray, database: "FaceDatabase") -> list[FaceDetection]:
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

            face_image = self._upper_face_image(face["face"])
            representations = self.deepface.represent(
                img_path=face_image,
                model_name=MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
                align=False,
            )
            if not representations:
                continue

            match = database.find_match(
                np.asarray(representations[0]["embedding"], dtype=np.float32)
            )
            if match is None:
                detections.append(FaceDetection(box=box))
            else:
                username, distance = match
                detections.append(
                    FaceDetection(box=box, username=username, distance=distance)
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

    def add_user(
        self, username: str, details: dict[str, str], embeddings: list[np.ndarray]
    ) -> None:
        with self.lock:
            if username in self.users:
                raise ValueError("اسم المستخدم مسجل مسبقًا.")
            self.users[username] = {
                "details": details,
                "embeddings": [embedding.tolist() for embedding in embeddings],
            }
            try:
                self._save_locked()
            except Exception:
                del self.users[username]
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

    def find_match(self, embedding: np.ndarray) -> tuple[str, float] | None:
        best_username: str | None = None
        best_distance = float("inf")
        query_norm = float(np.linalg.norm(embedding))
        if query_norm == 0:
            return None

        with self.lock:
            users = list(self.users.items())
        for username, record in users:
            for stored in record.get("embeddings", []):
                known = np.asarray(stored, dtype=np.float32)
                known_norm = float(np.linalg.norm(known))
                if known_norm == 0 or known.shape != embedding.shape:
                    continue
                distance = 1.0 - float(np.dot(embedding, known) / (query_norm * known_norm))
                if distance < best_distance:
                    best_username, best_distance = username, distance

        if best_username is not None and best_distance <= COSINE_DISTANCE_THRESHOLD:
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

    def _analysis_loop(self) -> None:
        while not self.stop_event.wait(0.65):
            with self.lock:
                if not self.detection_enabled or self.frame is None:
                    continue
                frame = self.frame.copy()
                scan_enabled = self.scan_enabled
            try:
                engine = self.get_engine()
                with engine.inference_lock:
                    detections = (
                        engine.analyze_frame(frame, self.database)
                        if scan_enabled
                        else engine.detect_frame(frame)
                    )
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
        self.window.rowconfigure(0, weight=1)
        self.preview = tk.Canvas(
            self.window,
            bg=COLORS["background"],
            highlightthickness=0,
            bd=0,
        )
        self.preview.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
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

    def _refresh(self) -> None:
        if not self.window.winfo_exists():
            return
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
                color = (80, 210, 130) if recognized else (50, 100, 245)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                label = detection.username or "غير معروف"
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
        self.stream.close()
        self.window.destroy()


class EnrollmentWindow(CameraWindow):
    """تسجيل موجّه يلتقط الصور تلقائيًا عند ثبات الوجه داخل الدليل."""

    def __init__(self, parent: tk.Misc, database: FaceDatabase) -> None:
        super().__init__(parent, database, "إضافة وجه جديد")
        self.window.geometry("1240x820")
        self.window.minsize(1050, 720)
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
        self.side_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
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
            text="اتبع التعليمات؛ سيتم التقاط الصورة تلقائيًا عند ثبات الوجه.",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            justify="right",
            wraplength=270,
        ).grid(row=1, column=0, sticky="ew", pady=(0, 14))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.side_panel,
            variable=self.progress_var,
            maximum=CAPTURE_COUNT,
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress_bar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.step_var = tk.StringVar(value="اللقطة 1 من 5 — أمام")
        tk.Label(
            self.side_panel,
            textvariable=self.step_var,
            bg=COLORS["surface"],
            fg=COLORS["accent"],
            font=("Arial", 12, "bold"),
            anchor="e",
        ).grid(row=3, column=0, sticky="ew", pady=(0, 4))
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
        ).grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.status_var = tk.StringVar(value="ضع وجهك داخل الإطار وانتظر.")
        tk.Label(
            self.side_panel,
            textvariable=self.status_var,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            justify="right",
            wraplength=270,
            anchor="e",
        ).grid(row=5, column=0, sticky="ew", pady=(0, 12))
        self.shot_indicators: list[tk.Label] = []
        indicators = tk.Frame(self.side_panel, bg=COLORS["surface"])
        indicators.grid(row=6, column=0, sticky="ew", pady=(0, 18))
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
        ).grid(row=7, column=0, sticky="ew", pady=(0, 8))

        self.name_var = tk.StringVar()
        self.username_var = tk.StringVar()
        self.phone_var = tk.StringVar()
        self.email_var = tk.StringVar()
        self.notes_var = tk.StringVar()
        fields = [
            ("الاسم الكامل", self.name_var),
            ("اسم المستخدم (فريد)", self.username_var),
            ("رقم الهاتف", self.phone_var),
            ("البريد الإلكتروني", self.email_var),
            ("بيانات إضافية", self.notes_var),
        ]
        for index, (label, variable) in enumerate(fields, start=8):
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

        controls = tk.Frame(self.window, bg=COLORS["background"])
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 14))
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
        self._cancel_countdown()
        self._store_capture(frame)

    def save_enrollment(self) -> None:
        username = self.username_var.get().strip()
        name = self.name_var.get().strip()
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
        if self.database.contains(username):
            messagebox.showerror(
                "اسم مستخدم مكرر", "اسم المستخدم مسجل مسبقًا.", parent=self.window
            )
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
            args=(username, details, frames),
            name="face-enrollment",
            daemon=True,
        ).start()

    def _enroll_worker(
        self, username: str, details: dict[str, str], frames: list[np.ndarray]
    ) -> None:
        try:
            engine = self.stream.get_engine()
            embeddings = []
            for index, frame in enumerate(frames, start=1):
                with engine.inference_lock:
                    embedding, _ = engine.embedding_from_frame(frame)
                embeddings.append(embedding)
                self.result_queue.put(
                    ("progress", f"تم تحليل الصورة {index} من {CAPTURE_COUNT}...")
                )
            self.database.add_user(username, details, embeddings)
            self.result_queue.put(("success", f"تم تسجيل المستخدم {username} بنجاح."))
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


class ScanWindow(CameraWindow):
    """نافذة الفحص مع لوحة جانبية تعرض معلومات الشخص عند التعرف عليه."""

    def __init__(self, parent: tk.Misc, database: FaceDatabase) -> None:
        super().__init__(parent, database, "تسجيل الدخول / فحص")
        self.window.geometry("1120x650")
        self.preview.grid_configure(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.window.columnconfigure(0, weight=3)
        self.window.columnconfigure(1, weight=1)

        self.side_panel = tk.Frame(
            self.window, bg=COLORS["surface"], padx=20, pady=20
        )
        self.side_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        tk.Label(
            self.side_panel,
            text="نتيجة الفحص",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Arial", 17, "bold"),
        ).pack(anchor="e", fill="x", pady=(0, 12))
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
        self.current_username: str | None = None

    def on_detections(
        self, detections: list[FaceDetection], revision: int
    ) -> None:
        recognized = next(
            (detection for detection in detections if detection.username is not None), None
        )
        if recognized is None:
            self.current_username = None
            self.status_var.set(
                "يتم الفحص... لم يتم التعرف على وجه مسجل."
                if detections
                else "يتم الفحص... انظر إلى الكاميرا."
            )
            self.info_var.set("لم يتم التعرف على مستخدم بعد.")
            return

        username = recognized.username
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

    def on_stream_error(self, error: str) -> None:
        self.status_var.set(error)


class FaceRecognitionApp:
    """النافذة الرئيسية للتطبيق."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("نظام التعرف على الوجه")
        self.root.geometry("640x480")
        self.root.resizable(False, False)
        self.database = FaceDatabase(DATABASE_PATH)
        self.root.configure(bg=COLORS["background"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TFrame", background=COLORS["background"]
        )
        style.configure(
            "TLabel",
            background=COLORS["background"],
            foreground=COLORS["text"],
            font=("Arial", 11),
        )
        style.configure(
            "TButton",
            font=("Arial", 11, "bold"),
            padding=(12, 10),
            background=COLORS["surface_light"],
            foreground=COLORS["text"],
            borderwidth=0,
        )
        style.map(
            "TButton",
            background=[("active", COLORS["accent"]), ("disabled", COLORS["surface"])],
            foreground=[("active", COLORS["background"])],
        )
        style.configure(
            "TEntry",
            padding=7,
            fieldbackground=COLORS["surface_light"],
            foreground=COLORS["text"],
        )
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=COLORS["surface_light"],
            background=COLORS["accent"],
        )

        container = ttk.Frame(root, padding=28)
        container.pack(fill="both", expand=True)
        ttk.Label(
            container,
            text="نظام التعرف على الوجه",
            font=("Arial", 23, "bold"),
        ).pack(pady=(12, 10))
        ttk.Label(
            container,
            text=(
                "يستخدم ArcFace وRetinaFace، مع طمس الجزء السفلي من الوجه "
                "لتركيز التمثيل على المنطقة العلوية."
            ),
            wraplength=520,
            justify="center",
        ).pack(pady=(0, 25))
        ttk.Button(
            container,
            text="تسجيل الدخول / فحص",
            command=self.open_scan,
        ).pack(fill="x", ipady=10, pady=8)
        ttk.Button(
            container,
            text="إضافة وجه جديد",
            command=self.open_enrollment,
        ).pack(fill="x", ipady=10, pady=8)
        ttk.Label(
            container,
            text=(
                "تنبيه: التعرف من الوجه العلوي تقريبي وليس وسيلة تحقق آمنة بمفرده. "
                "لا تستخدمه لاتخاذ قرارات حساسة."
            ),
            foreground=COLORS["warning"],
            wraplength=520,
            justify="center",
        ).pack(pady=(24, 0))

    def open_scan(self) -> None:
        self._open_camera_window(ScanWindow)

    def open_enrollment(self) -> None:
        self._open_camera_window(EnrollmentWindow)

    def _open_camera_window(self, window_type: type[CameraWindow]) -> None:
        try:
            child = window_type(self.root, self.database)
        except Exception as exc:
            messagebox.showerror("تعذر فتح الكاميرا", str(exc), parent=self.root)
            return
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
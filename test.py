import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import threading
import pickle
import os
import numpy as np
from deepface import DeepFace

# دالة مخصصة لحساب مسافة التطابق بدلاً من دالة مكتبة DeepFace لتجنب أخطاء التحديثات
def findCosineDistance(source, test):
    a = np.dot(source, test)
    b = np.linalg.norm(source) * np.linalg.norm(test)
    return 1 - (a / b)

# إعدادات قاعدة البيانات
DB_FILE = 'face_db.pkl'

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("نظام التعرف على الوجه الذكي")
        self.root.geometry("1000x600")
        self.root.configure(bg="#2c3e50")

        # المتغيرات الأساسية
        self.video_capture = cv2.VideoCapture(0)
        self.known_faces = self.load_database()
        
        # متغيرات الخيوط (Threading) لتجنب تجميد الواجهة
        self.current_frame = None
        self.is_scanning = False
        self.is_enrolling = False
        self.detected_name = "مجهول"
        self.detected_bbox = None # إحداثيات الوجه (x, y, w, h)
        self.scan_message = ""

        self.setup_gui()
        self.update_camera()

        # تشغيل خيط (Thread) منفصل لمعالجة الذكاء الاصطناعي في الخلفية
        self.ai_thread = threading.Thread(target=self.process_ai_background, daemon=True)
        self.ai_thread.start()

    def load_database(self):
        """تحميل قاعدة بيانات الوجوه المحفوظة (Embeddings)"""
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_database(self):
        """حفظ التحديثات في قاعدة البيانات"""
        with open(DB_FILE, 'wb') as f:
            pickle.dump(self.known_faces, f)

    def setup_gui(self):
        """بناء واجهة المستخدم (GUI)"""
        # قسم الكاميرا (اليسار)
        self.video_frame = tk.Label(self.root, bg="black")
        self.video_frame.pack(side=tk.LEFT, padx=20, pady=20)

        # اللوحة الجانبية (اليمين)
        self.side_panel = tk.Frame(self.root, bg="#34495e", width=300)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)

        # النصوص والأزرار في اللوحة الجانبية
        self.title_label = tk.Label(self.side_panel, text="لوحة التحكم", font=("Arial", 20, "bold"), bg="#34495e", fg="white")
        self.title_label.pack(pady=20)

        self.info_label = tk.Label(self.side_panel, text="جاهز...", font=("Arial", 14), bg="#34495e", fg="#2ecc71")
        self.info_label.pack(pady=20)

        self.btn_scan = tk.Button(self.side_panel, text="تسجيل الدخول / فحص", font=("Arial", 14), bg="#3498db", fg="white", command=self.start_scan)
        self.btn_scan.pack(fill=tk.X, pady=10, padx=20)

        self.btn_enroll = tk.Button(self.side_panel, text="إضافة وجه جديد", font=("Arial", 14), bg="#e67e22", fg="white", command=self.start_enroll)
        self.btn_enroll.pack(fill=tk.X, pady=10, padx=20)

    def update_camera(self):
        """تحديث إطار الكاميرا في الواجهة الرسومية وعرض الإطارات الملونة"""
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1) # عكس الكاميرا كالمرايا
            self.current_frame = frame.copy()

            # رسم الإطار الملون ومعلومات المستخدم إذا كان هناك فحص نشط
            if self.is_scanning or self.is_enrolling:
                cv2.putText(frame, self.scan_message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                if self.detected_bbox is not None:
                    x, y, w, h = self.detected_bbox
                    color = (0, 255, 0) if self.detected_name != "مجهول" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, self.detected_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # تحويل الصورة لتناسب Tkinter
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        # استدعاء الدالة كل 15 مللي ثانية لتحديث الكاميرا بسلاسة
        self.root.after(15, self.update_camera)

    def start_scan(self):
        """تفعيل وضع الفحص"""
        self.is_scanning = True
        self.is_enrolling = False
        self.scan_message = "يتم الفحص... انظر للكاميرا"
        self.info_label.config(text="جاري التعرف على الوجه...", fg="#f1c40f")

    def start_enroll(self):
        """تفعيل وضع إضافة وجه جديد"""
        user_name = simpledialog.askstring("إضافة مستخدم", "أدخل اسم المستخدم:")
        if not user_name:
            return

        self.is_scanning = False
        self.is_enrolling = True
        self.scan_message = f"يتم تسجيل: {user_name}"
        self.info_label.config(text="الرجاء تحريك رأسك ببطء...", fg="#e67e22")
        
        # تشغيل عملية التقاط الزوايا المتعددة في خيط منفصل
        threading.Thread(target=self.enroll_user_process, args=(user_name,), daemon=True).start()

    def enroll_user_process(self, name):
        """التقاط عدة صور بزوايا مختلفة لاستخراج الملامح (Embeddings)"""
        embeddings_list = []
        angles = ["انظر للأمام", "انظر لليمين", "انظر لليسار", "انظر للأعلى", "انظر للأسفل"]
        
        for angle in angles:
            self.scan_message = f"التسجيل: {angle}"
            import time
            time.sleep(2) # إعطاء المستخدم ثانيتين لتغيير زاوية رأسه

            if self.current_frame is not None:
                try:
                    # استخراج الملامح باستخدام ArcFace
                    result = DeepFace.represent(self.current_frame, model_name="ArcFace", detector_backend="retinaface", enforce_detection=True)
                    if len(result) > 0:
                        embeddings_list.append(result[0]["embedding"])
                except Exception as e:
                    print(f"لم يتم العثور على وجه في الزاوية: {angle}")
        
        if embeddings_list:
            self.known_faces[name] = embeddings_list
            self.save_database()
            self.info_label.config(text=f"تم تسجيل {name} بنجاح!", fg="#2ecc71")
        else:
            self.info_label.config(text="فشل التسجيل، حاول مجدداً.", fg="#e74c3c")
            
        self.is_enrolling = False
        self.scan_message = ""

    def process_ai_background(self):
        """معالجة الذكاء الاصطناعي في الخلفية لكي لا تتجمد الكاميرا"""
        while True:
            if self.is_scanning and self.current_frame is not None:
                try:
                    # اكتشاف واستخراج الملامح من الإطار الحالي
                    result = DeepFace.represent(self.current_frame, model_name="ArcFace", detector_backend="retinaface", enforce_detection=True)
                    
                    if len(result) > 0:
                        face_data = result[0]
                        current_embedding = face_data["embedding"]
                        
                        # إحداثيات مربع الوجه
                        region = face_data["facial_area"]
                        self.detected_bbox = (region['x'], region['y'], region['w'], region['h'])
                        
                        # مقارنة الملامح الحالية مع قاعدة البيانات
                        best_match_name = "مجهول"
                        best_score = float("inf")
                        
                        for name, embeddings in self.known_faces.items():
                            for emb in embeddings:
                                # حساب مسافة التطابق (Cosine Similarity)
                                score = findCosineDistance(current_embedding, emb)
                                if score < 0.68: # الحد المسموح به لنموذج ArcFace
                                    if score < best_score:
                                        best_score = score
                                        best_match_name = name

                        self.detected_name = best_match_name
                        
                        if best_match_name != "مجهول":
                            self.info_label.config(text=f"مرحباً بك: {best_match_name}", fg="#2ecc71")
                        else:
                            self.info_label.config(text="وجه غير معروف!", fg="#e74c3c")
                            
                except Exception as e:
                    # في حالة عدم وجود وجه في الإطار
                    self.detected_bbox = None
                    self.detected_name = "مجهول"
            
            import time
            time.sleep(0.1) # استراحة قصيرة لتخفيف العبء على كرت الشاشة

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    
    # عند إغلاق البرنامج
    def on_closing():
        app.video_capture.release()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
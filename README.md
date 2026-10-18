# Face Scanner

## Overview

Face Scanner is a Windows desktop application for local face enrollment and
experimental face comparison. It uses PySide6, OpenCV, DeepFace, ArcFace, and
RetinaFace. Face records and embeddings are stored locally on the computer.

The application is an experiment, not a secure authentication system. Face
recognition can produce false matches or miss a registered person. Do not use
it by itself for access control, identity verification, or other sensitive
decisions.

## Features

- Enroll and manage face records, with a consent prompt and face-quality checks.
- Compare upper-face and full-face representations.
- Choose camera resolution and optical/preview zoom. Camera choices are saved
  between runs; digital preview zoom does not add image detail.
- Clearly distinguishes a registered match from **Unknown**. A match is
  confirmed only after it remains consistent across multiple frames.
- Create and restore versioned JSON backups from the dashboard.
- Uses the supplied `9806214.png` image in the app, and `FaceScanner.ico` as
  the Windows executable icon.

## Run from source

Install Python and the packages listed in `requirements.txt`, then run:

```powershell
python -m pip install -r requirements.txt
python face_scanner.py
```

The first face analysis may take longer while DeepFace initializes and
downloads model files. The camera window shows an indeterminate progress
indicator during initialization; DeepFace does not expose a reliable download
percentage. Internet access may be needed for the first run on a computer. If
camera access fails, check Windows camera privacy settings and that another
application is not using the camera.

## Build a portable Windows folder

From the project directory, run:

```powershell
python -m PyInstaller --noconfirm --clean --windowed --onedir --name FaceScanner --exclude-module pyarrow --icon=FaceScanner.ico --add-data "9806214.png;." --collect-all deepface --collect-all retinaface --collect-all onnxruntime face_scanner.py
```

Distribute the complete `dist\FaceScanner` folder (or zip the complete folder),
not only `FaceScanner.exe`. Run `dist\FaceScanner\FaceScanner.exe` after
extracting it to a folder the current Windows user can write to. Administrator
privileges are normally not required. A new computer may need internet access
the first time it downloads the face models.

## Face data, consent, and backups

- Ask for and record the person's consent before enrolling their face.
- Face records are stored locally in `face_database.pkl`. They are retained
  until a user is deleted in the app; the app does not currently expire records
  automatically.
- The database file is **not encrypted**. Restrict access to the computer and
  do not share the database or backup files; face embeddings and personal
  details are sensitive.
- Backups created by the app use versioned JSON and can be restored from the
  dashboard. Restoring replaces all current records; make a backup first if
  they need to be preserved.
- The legacy database is a Python Pickle file. Never open or replace it with a
  Pickle file from an untrusted source: loading an untrusted Pickle can execute
  code. For restore, use a JSON backup created by this application.
- The JSON backups contain face embeddings and user details in readable form;
  store them securely.

## Tests

```powershell
python -m unittest test_face_database test_face_scanner_qt
```

---

# ماسح الوجوه

## نظرة عامة

ماسح الوجوه هو تطبيق سطح مكتب لنظام Windows لتسجيل الوجوه ومقارنتها لأغراض
تجريبية. يستخدم PySide6 وOpenCV وDeepFace وArcFace وRetinaFace. تُحفظ سجلات
الوجوه والتمثيلات محليًا على الجهاز.

هذا التطبيق تجريبي وليس نظامًا آمنًا للتحقق من الهوية. قد يتعرف التطبيق على
شخص خطأً أو لا يتعرف على شخص مسجل. لا تستخدمه وحده للتحكم بالدخول أو التحقق
من الهوية أو لاتخاذ قرارات حساسة.

## الميزات

- تسجيل الوجوه وإدارتها، مع طلب الموافقة وفحص جودة الوجه.
- مقارنة تمثيل الجزء العلوي من الوجه بالوجه كاملًا.
- اختيار دقة الكاميرا والتقريب البصري وتقريب المعاينة. تُحفظ إعدادات الكاميرا
  بين مرات التشغيل؛ التقريب الرقمي للمعاينة لا يضيف تفاصيل للصورة.
- تمييز واضح بين المطابقة المسجلة وحالة **غير معروف**. لا تُعرض المطابقة
  المؤكدة إلا بعد ثباتها عبر عدة إطارات.
- إنشاء نسخ احتياطية بصيغة JSON واستعادتها من لوحة التحكم.
- استخدام الصورة `9806214.png` داخل التطبيق و`FaceScanner.ico` كأيقونة لملف
  تشغيل Windows.

## التشغيل من المصدر

ثبّت Python والحزم الموجودة في `requirements.txt`، ثم شغّل:

```powershell
python -m pip install -r requirements.txt
python face_scanner.py
```

قد يستغرق تحليل الوجه الأول وقتًا أطول أثناء تهيئة DeepFace وتنزيل النماذج.
تظهر في نافذة الكاميرا إشارة تقدم غير محددة أثناء التهيئة؛ لا توفر DeepFace
نسبة تنزيل موثوقة. قد يلزم اتصال بالإنترنت عند التشغيل لأول مرة على جهاز
جديد. إذا تعذر الوصول إلى الكاميرا، تحقق من إعدادات خصوصية الكاميرا في
Windows ومن عدم استخدام برنامج آخر لها.

## إنشاء مجلد Windows محمول

من مجلد المشروع شغّل:

```powershell
python -m PyInstaller --noconfirm --clean --windowed --onedir --name FaceScanner --exclude-module pyarrow --icon=FaceScanner.ico --add-data "9806214.png;." --collect-all deepface --collect-all retinaface --collect-all onnxruntime face_scanner.py
```

وزّع مجلد `dist\FaceScanner` كاملًا أو اضغطه كملف ZIP؛ لا توزّع
`FaceScanner.exe` وحده. شغّل `dist\FaceScanner\FaceScanner.exe` بعد فك الضغط
في مجلد يستطيع مستخدم Windows الكتابة فيه. لا تحتاج عادةً إلى صلاحيات مسؤول.
قد يحتاج الجهاز الجديد إلى الإنترنت عند تنزيل نماذج الوجوه أول مرة.

## بيانات الوجه والموافقة والنسخ الاحتياطي

- احصل على موافقة الشخص وسجّلها قبل تسجيل وجهه.
- تُحفظ بيانات الوجه محليًا في `face_database.pkl` وتبقى حتى حذف المستخدم من
  التطبيق؛ لا يحذف التطبيق التسجيلات تلقائيًا بعد مدة.
- قاعدة البيانات **غير مشفرة**. قيّد الوصول إلى الجهاز ولا تشارك قاعدة
  البيانات أو ملفات النسخ الاحتياطي؛ فالتمثيلات الوجهية والبيانات الشخصية
  معلومات حساسة.
- ينشئ التطبيق نسخًا احتياطية بصيغة JSON ذات إصدار، ويمكن استعادتها من لوحة
  التحكم. الاستعادة تستبدل جميع التسجيلات الحالية؛ أنشئ نسخة احتياطية أولًا
  إن أردت الاحتفاظ بها.
- قاعدة البيانات القديمة بصيغة Python Pickle. لا تفتحها ولا تستبدلها بملف
  Pickle من مصدر غير موثوق، إذ قد يؤدي تحميله إلى تنفيذ تعليمات برمجية.
  استخدم للاستعادة نسخة JSON أنشأها هذا التطبيق.
- تحتوي نسخ JSON الاحتياطية على تمثيلات الوجه وبيانات المستخدم بصيغة مقروءة؛
  خزّنها في مكان آمن.

## الاختبارات

```powershell
python -m unittest test_face_database test_face_scanner_qt
```

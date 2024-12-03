import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
import cv2
import os
import firebase_admin
from firebase_admin import credentials, db
import face_recognition
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
import datetime
from kivy.clock import Clock  # Thêm thư viện Clock để cập nhật liên tục

# Kết nối Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-8dc3e-default-rtdb.firebaseio.com/"
})
ref = db.reference('Students')

# Khởi tạo bộ phát hiện khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceRecognitionApp(App):
    def build(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Lỗi", "Không thể mở webcam.")
            return

        self.root = BoxLayout(orientation='horizontal')

        # Giao diện nhập liệu
        self.data_layout = GridLayout(cols=2, spacing=10, padding=10)
        self.data_layout.add_widget(Label(text="Student ID:"))
        self.entry_id = TextInput()
        self.data_layout.add_widget(self.entry_id)

        self.data_layout.add_widget(Label(text="Name:"))
        self.entry_name = TextInput()
        self.data_layout.add_widget(self.entry_name)

        self.data_layout.add_widget(Label(text="Major:"))
        self.entry_major = TextInput()
        self.data_layout.add_widget(self.entry_major)

        self.data_layout.add_widget(Label(text="Starting Year:"))
        self.entry_starting_year = TextInput()
        self.data_layout.add_widget(self.entry_starting_year)

        self.data_layout.add_widget(Label(text="Total Attendance:"))
        self.entry_total_attendance = TextInput()
        self.data_layout.add_widget(self.entry_total_attendance)

        self.data_layout.add_widget(Label(text="Standing:"))
        self.entry_standing = TextInput()
        self.data_layout.add_widget(self.entry_standing)

        self.data_layout.add_widget(Label(text="Year:"))
        self.entry_year = TextInput()
        self.data_layout.add_widget(self.entry_year)

        self.add_button = Button(text="Thêm sinh viên", on_press=self.add_student)
        self.data_layout.add_widget(self.add_button)

        # Giao diện hiển thị camera
        self.camera_layout = BoxLayout(orientation='vertical')
        self.video_widget = Image(size_hint=(1, 1))
        self.camera_layout.add_widget(self.video_widget)
        self.capture_button = Button(text="Chụp ảnh", on_press=self.capture_image)
        self.camera_layout.add_widget(self.capture_button)

        # Chia giao diện
        self.root.add_widget(self.data_layout)
        self.root.add_widget(self.camera_layout)

        # Bắt đầu luồng video
        self.update_frame()

        return self.root

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Chuyển sang màu xám để phát hiện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            # Vẽ hình chữ nhật quanh khuôn mặt trước khi lật
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Lật ảnh theo chiều ngang (để hiển thị đúng)
            frame = cv2.flip(frame, 0)

            # Chuyển đổi ảnh từ BGR sang RGB để hiển thị
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            self.video_widget.texture = texture

        # Lên lịch cập nhật frame sau 10ms
        Clock.schedule_once(lambda dt: self.update_frame(), 0.01)

    def capture_image(self, instance):
        student_id = self.entry_id.text
        if student_id:
            if not os.path.exists('../Images'):
                os.makedirs('../Images')

            ret, frame = self.cap.read()
            if ret:
                # Lật ảnh theo chiều ngang nếu bị ngược
                frame = cv2.flip(frame, 1)  # Lật ảnh

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_frame = frame[y:y + h, x:x + w]
                        resized_face = cv2.resize(face_frame, (216, 216))

                        image_path = f"Images/{student_id}.png"
                        cv2.imwrite(image_path, resized_face)

                        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
                        try:
                            encode = face_recognition.face_encodings(rgb_face)[0]
                            print(f"Encoding cho {student_id}: {encode}")
                        except IndexError:
                            self.show_error("Lỗi", "Không thể mã hóa khuôn mặt, hãy thử lại.")
                else:
                    self.show_error("Lỗi", "Không phát hiện được khuôn mặt trong khung hình.")
        else:
            self.show_error("Lỗi", "Vui lòng nhập Student ID trước khi chụp ảnh.")

    def add_student(self, instance):
        student_id = self.entry_id.text
        name = self.entry_name.text
        major = self.entry_major.text
        starting_year = self.entry_starting_year.text
        total_attendance = self.entry_total_attendance.text
        standing = self.entry_standing.text
        year = self.entry_year.text

        if student_id and name and major and starting_year and total_attendance and standing and year:
            last_attendance_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            student_data = {
                "name": name,
                "major": major,
                "starting_year": starting_year,
                "total_attendance": total_attendance,
                "standing": standing,
                "year": year,
                "last_attendance_time": last_attendance_time
            }

            ref.child(student_id).set(student_data)
            self.show_popup("Thành công", f"Đã thêm sinh viên: {name} vào Firebase.")
        else:
            self.show_error("Lỗi", "Vui lòng nhập đầy đủ thông tin.")

    def show_error(self, title, message):
        popup = Popup(title=title, content=Label(text=message), size_hint=(None, None), size=(400, 200))
        popup.open()

    def show_popup(self, title, message):
        popup = Popup(title=title, content=Label(text=message), size_hint=(None, None), size=(400, 200))
        popup.open()

    def on_stop(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    FaceRecognitionApp().run()

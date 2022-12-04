import os

from PyQt5.QtCore import QUrl, pyqtSignal
from PyQt5.QtGui import QWindow
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QFileDialog
from PyQt5 import uic
import sys
import resource
from prediction.live_audio_testing import denoise_from_wav_file


class Fourth_page(QMainWindow):
    def __init__(self, parent=None):
        super(Fourth_page, self).__init__()
        self.parent = parent
        uic.loadUi("ui/fourth_page.ui", self)

        self.back.mousePressEvent = self.back_btn

    def back_btn(self, event):
        self.parent.show()
        self.hide()


class Third_page(QMainWindow):
    def __init__(self, parent=None):
        super(Third_page, self).__init__()
        uic.loadUi("ui/third_page.ui", self)
        self.parent = parent

        self.denoise_btn.hide()
        self.file_name.hide()
        self.play_btn.hide()
        self.save_btn.hide()

        self.select_file_btn.clicked.connect(self.select_file)
        self.denoise_btn.clicked.connect(self.get_denoised_audio)

        self.back.mousePressEvent = self.back_btn

    def select_file(self):
        fname = QFileDialog.getOpenFileName(self, "select file", "", "Audio Files(*.wav)")

        if fname:
            self.full_path = fname[0]
            self.file = fname[0].split('/')[-1]
            self.file_name.setText(str(self.file))
            self.file_name.show()
            self.denoise_btn.show()
            # self.play_btn.show()

    def get_denoised_audio(self):
        self.denoised_audio = denoise_from_wav_file(self.full_path)
        self.play_btn.show()
        self.play_btn.mousePressEvent = self.playAudioFile

    def playAudioFile(self, event):
        self.player = QMediaPlayer()
        # full_file_path = os.path.join(path)
        url = QUrl.fromLocalFile(self.denoised_audio)
        content = QMediaContent(url)

        print(type(content))

        self.player.setMedia(content)
        self.player.play()
        print('play')

    def back_btn(self, event):
        self.parent.show()
        self.hide()


class Second_page(QMainWindow):
    def __init__(self, parent=None):
        super(Second_page, self).__init__()
        self.parent = parent
        uic.loadUi("ui/second_page.ui", self)

        self.denoise_wav_file.clicked.connect(self.denoise_wav_file_btn)
        self.denoise_recording.clicked.connect(self.denoise_live_audio)
        self.exit.clicked.connect(self.exit_btn)

    def denoise_wav_file_btn(self):
        self.denoise_file = Third_page(self)
        self.hide()
        self.denoise_file.show()

    def denoise_live_audio(self):
        self.denoise_record = Fourth_page(self)
        self.denoise_record.show()
        self.hide()

    def exit_btn(self):
        self.parent.show()
        self.hide()


class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()

        uic.loadUi('ui/home_page.ui', self)
        self.pushButton.clicked.connect(self.second_page)

    def second_page(self):
        self.s = Second_page(self)
        self.hide()
        self.s.show()


if __name__ == '__main__':
    app = QApplication([])
    window = Ui()
    window.show()
    app.exec_()
    # sys.exit(app.exec_())

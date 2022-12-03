import os

from PyQt5.QtGui import QWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QFileDialog
from PyQt5 import uic
import sys
import resource


class Fourth_page(QMainWindow):
    def __init__(self, parent=None):
        super(Fourth_page, self).__init__()
        uic.loadUi("ui/fourth_page.ui", self)


class Third_page(QMainWindow):
    def __init__(self, parent=None):
        super(Third_page, self).__init__()
        uic.loadUi("ui/third_page.ui", self)
        self.denoise_btn.hide()
        self.file_name.hide()
        self.play_btn.hide()
        self.save_btn.hide()

        self.select_file_btn.clicked.connect(self.select_file)

    def select_file(self):
        fname = QFileDialog.getOpenFileName(self, "select file", "", "Audio Files(*.wav)")

        if fname:
            file_name = fname[0].split('/')[-1]
            self.file_name.setText(str(file_name))
            self.file_name.show()
            self.denoise_btn.show()


class Second_page(QMainWindow):
    def __init__(self, parent=None):
        super(Second_page, self).__init__()
        self.parent = parent
        uic.loadUi("ui/second_page.ui", self)

        self.denoise_wav_file.clicked.connect(self.denoise_wav_file_btn)
        self.denoise_recording.clicked.connect(self.denoise_live_audio)
        self.exit.clicked.connect(self.exit_btn)

    def denoise_wav_file_btn(self):
        self.denoise_file = Third_page()
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

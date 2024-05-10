import PyQt5 
from PyQt5.QtWidgets  import QApplication , QMainWindow ,QLabel, QPushButton
import sys

def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(200,200,300,300)

    label = QLabel(win)
    label.setText('my first label')
    label.move(100,50)

    b1 = QPushButton('click me', win)
    b1.move(100,100)


    win.show()
    sys.exit(app.exec_())

    

window()
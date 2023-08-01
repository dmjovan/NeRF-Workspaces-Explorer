import sys

from PyQt5.QtWidgets import QApplication

from application.app import LandingPage

if __name__ == '__main__':
    app = QApplication(sys.argv)
    landing_page = LandingPage()
    landing_page.show()
    sys.exit(app.exec_())

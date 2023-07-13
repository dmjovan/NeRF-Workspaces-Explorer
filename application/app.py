import sys
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QColor, QPainter, QFont, QIcon, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QWidget, QVBoxLayout, QGridLayout, \
    QHBoxLayout

WORKSPACES = {
    0: {"name": "Office 0", "folder_name": "office0"},
    1: {"name": "Office 1", "folder_name": "office1"},
    2: {"name": "Office 2", "folder_name": "office2"},
    3: {"name": "Office 4", "folder_name": "office4"}
}


class LandingPage(QMainWindow):
    """
    Landing Page for Workspaces Explorer Application.
    """

    def __init__(self) -> None:
        super().__init__()

        # Set the application window title
        self.setWindowTitle("Workspaces Explorer")
        # Set the fixed size of the window
        # self.setFixedSize(1600, 700)
        self.setFixedSize(1000, 700)

        # Create a central widget and layout
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Set the font
        font = QFont("Arial", 12)

        # Add description label
        instruction_label = QLabel("Please select the workspace to take a detailed tour", self)
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setFont(font)
        instruction_label.setStyleSheet("color: white;")
        layout.addWidget(instruction_label)

        # Create labels for the images and titles
        grid_layout = QGridLayout()

        # Loop over all available workspaces
        for i in range(len(WORKSPACES.keys())):
            # Create a clickable label for each image
            label = QLabel(self)
            pixmap = QPixmap(f"application/workspaces/{WORKSPACES[i]['folder_name']}/thumbnail.jpg")

            # Resize the image to fit the label width
            # pixmap = pixmap.scaledToWidth(400)
            pixmap = pixmap.scaledToWidth(300)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)

            # Store the index of the label
            label.index = i

            # Assign the click event over the image label
            label.mousePressEvent = lambda event, index=i: self.open_workspace_viewer(event, index)

            # Create a shaded background for the label
            label.setStyleSheet("background-color: rgba(0, 0, 0, 50);")

            # Add the label to the grid layout
            # grid_layout.addWidget(label, ((i // 4) * 2), i % 4)
            grid_layout.addWidget(label, ((i // 2) * 2), i % 2)

            # Create a title label for each image
            title_label = QLabel(WORKSPACES[i]["name"], self)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setFont(font)
            title_label.setStyleSheet("color: white;")

            # Add the title label to the grid layout
            # grid_layout.addWidget(title_label, ((i // 4) * 2 + 1), i % 4)
            grid_layout.addWidget(title_label, ((i // 2) * 2 + 1), i % 2)

        layout.addLayout(grid_layout)

    def paintEvent(self, event: Any) -> None:
        """
        Overriding the paintEvent to set the background color.
        """

        painter = QPainter(self)

        # Set a smooth black color (dark gray)
        painter.setBrush(QColor(80, 80, 90))
        painter.drawRect(0, 0, self.width(), self.height())

        super().paintEvent(event)

    def open_workspace_viewer(self, event: Any, index: int) -> None:
        """
        By clicking on any workspace thumbnail WorkspaceViewer page gets started.
        """

        self.hide()
        workspace_viewer = WorkspaceViewer(self, index)
        workspace_viewer.show()


class WorkspaceViewer(QMainWindow):

    def __init__(self, parent: QMainWindow, index: int) -> None:
        super().__init__(parent)

        # Store the index of the previously selected workspace
        self.workspace_index = index

        # Set the application window title
        self.setWindowTitle("Workspace Details")
        # Set the fixed size of the window
        self.setFixedSize(1200, 1000)

        # Create a central widget and layout
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.main_buttons_style_sheet = "QPushButton { background-color: #4CAF50; color: white; border: none; " \
                                        "padding: 10px; border-radius: 5px; font-weight: bold; } " \
                                        "QPushButton:hover { background-color: #45a049; }"

        self.camera_buttons_style_sheet = "QPushButton { background-color: #EEC10F; color: white; border: none; " \
                                          "padding: 10px; border-radius: 5px; font-weight: bold; } " \
                                          "QPushButton:hover { background-color: #CDA609; }"

        # Button for returning to the landing page
        return_landing_page_button = QPushButton("Explore another workspace", self)
        return_landing_page_button.clicked.connect(self.return_to_landing_page)
        return_landing_page_button.setMaximumWidth(200)
        return_landing_page_button.setMaximumHeight(50)
        return_landing_page_button.setStyleSheet(self.main_buttons_style_sheet)
        layout.addWidget(return_landing_page_button, alignment=Qt.AlignCenter | Qt.AlignTop)

        self.bev_description_label = QLabel(f"Bird's-eye view representation of the workspace "
                                            f"'{WORKSPACES[self.workspace_index]['name']}'", self)
        self.bev_description_label.setAlignment(Qt.AlignCenter)
        self.bev_description_label.setFont(QFont("Arial", 12))
        self.bev_description_label.setStyleSheet("color: white;")
        layout.addWidget(self.bev_description_label)

        # Create the bird's-eye view image and description
        self.bev_label = BirdsEyeViewImageArea(self)
        self.bev_pixmap = QPixmap(f"application/workspaces/{WORKSPACES[self.workspace_index]['folder_name']}/floor_plan.jpg")
        self.bev_pixmap = self.bev_pixmap.scaledToWidth(800)
        self.bev_label.setPixmap(self.bev_pixmap)
        self.bev_label.setAlignment(Qt.AlignCenter)
        self.bev_label.left_click.connect(self.render_detailed_workspace_view)
        layout.addWidget(self.bev_label)

        self.bev_instruction_label = QLabel(f"Click on the image for detailed in-place workspace view", self)
        self.bev_instruction_label.setAlignment(Qt.AlignCenter)
        self.bev_instruction_label.setFont(QFont("Arial", 10))
        self.bev_instruction_label.setStyleSheet("color: white;")
        layout.addWidget(self.bev_instruction_label)

        # Placeholder for render image label
        self.nerf_image = None

        # Placeholder for text label for camera turn buttons
        self.camera_buttons_text_label = None

        # Placeholders for left-right buttons
        self.left_button = None
        self.right_button = None

        # Placeholders for up-down buttons
        self.up_button = None
        self.down_button = None

        # Placeholder for button for returning to bird's-eye view
        self.return_bev_button = None

        # Camera coordinates - relative x, y
        # Relative coordinates x and y have range [0, 1], where starting point on
        # BEV image is top left corner
        # Relative coordinate x goes along left-to-right axis, and
        # Relative coordinate y goes along top-to-bottom axis
        self.rel_x = 0.0
        self.rel_y = 0.0

        # Camera angles [deg]
        self.yaw_angle = 0
        self.pitch_angle = 0
        self.angle_step = 15

    def reset_coordinates(self):
        """
        Resetting coordinates of camera view.
        """

        self.rel_x = 0.0
        self.rel_y = 0.0

    def reset_angles(self):
        """
        Resetting angles for camera view.
        """

        self.yaw_angle = 0
        self.pitch_angle = 0

    def paintEvent(self, event: Any) -> None:
        """
        Overriding the paintEvent to set the background color.
        """

        painter = QPainter(self)

        # Set a smooth black color (dark gray)
        painter.setBrush(QColor(80, 80, 90))
        painter.drawRect(0, 0, self.width(), self.height())

        super().paintEvent(event)

    def return_to_landing_page(self) -> None:
        """
        By clicking the 'Explore another workspace' button landing page shows up.
        """

        self.parent().show()
        self.close()

    def render_detailed_workspace_view(self, x: float, y: float) -> None:
        """
        Rendering detailed workspace view - in-place viewing of workspaces interior.
        """

        self.rel_x = x
        self.rel_y = y

        # Gather the central widget
        layout = self.centralWidget().layout()

        # Remove the bird's-eye view image, description and instruction text
        layout.removeWidget(self.bev_label)
        layout.removeWidget(self.bev_description_label)
        layout.removeWidget(self.bev_instruction_label)

        self.bev_label.setParent(None)
        self.bev_description_label.setParent(None)
        self.bev_instruction_label.setParent(None)

        # Create a NeRF image label and display the new image
        self.nerf_image = QLabel(self)
        self.nerf_image.setAlignment(Qt.AlignCenter)
        self.render_nerf_image()

        layout.addWidget(self.nerf_image)

        self.camera_buttons_text_label = QLabel(f"Turn camera by clicking buttons bellow", self)
        self.camera_buttons_text_label.setAlignment(Qt.AlignCenter)
        self.camera_buttons_text_label.setFont(QFont("Arial", 10))
        self.camera_buttons_text_label.setStyleSheet("color: white;")
        layout.addWidget(self.camera_buttons_text_label)

        buttons_layout = QHBoxLayout(self)

        # Button for turning the 'camera' to the left
        self.left_button = QPushButton(self)
        self.left_button.setMaximumWidth(200)
        self.left_button.setMaximumHeight(40)
        self.left_button.setStyleSheet(self.camera_buttons_style_sheet)
        self.left_button.setIcon(QIcon("application/imgs/left_arrow.png"))
        self.left_button.setIconSize(self.left_button.size())
        self.left_button.clicked.connect(self.left_button_clicked)
        buttons_layout.addWidget(self.left_button)

        # Button for turning the 'camera' to the right
        self.right_button = QPushButton(self)
        self.right_button.setMaximumWidth(200)
        self.right_button.setMaximumHeight(40)
        self.right_button.setStyleSheet(self.camera_buttons_style_sheet)
        self.right_button.setIcon(QIcon("application/imgs/rigth_arrow.png"))
        self.right_button.setIconSize(self.right_button.size())
        self.right_button.clicked.connect(self.right_button_clicked)
        buttons_layout.addWidget(self.right_button)

        # Button for turning the 'camera' up
        self.up_button = QPushButton(self)
        self.up_button.setMaximumWidth(200)
        self.up_button.setMaximumHeight(40)
        self.up_button.setStyleSheet(self.camera_buttons_style_sheet)
        self.up_button.setIcon(QIcon("application/imgs/up_arrow.png"))
        self.up_button.setIconSize(self.up_button.size())
        self.up_button.clicked.connect(self.up_button_clicked)
        buttons_layout.addWidget(self.up_button)

        # Button for turning the 'camera' down
        self.down_button = QPushButton(self)
        self.down_button.setMaximumWidth(200)
        self.down_button.setMaximumHeight(40)
        self.down_button.setStyleSheet(self.camera_buttons_style_sheet)
        self.down_button.setIcon(QIcon("application/imgs/down_arrow.png"))
        self.down_button.setIconSize(self.down_button.size())
        self.down_button.clicked.connect(self.down_button_clicked)
        buttons_layout.addWidget(self.down_button)

        layout.addLayout(buttons_layout)

        # Button for returning to the bird's-eye view image
        self.return_bev_button = QPushButton("Back to Bird's-Eye View", self)
        self.return_bev_button.clicked.connect(self.return_to_bev_view)
        self.return_bev_button.setMaximumWidth(200)
        self.return_bev_button.setMaximumHeight(50)
        self.return_bev_button.setStyleSheet(self.main_buttons_style_sheet)
        layout.addWidget(self.return_bev_button, alignment=Qt.AlignCenter | Qt.AlignBottom)

    def render_nerf_image(self):
        """
        Rendering image for each particular view point using NeRF model pretrained for each workspace.
        """

        print(f"Rendering new NeRF view from following coordinates: \n"
              f"\trelative_x: {self.rel_x:3f},\n"
              f"\trelative_y: {self.rel_y:3f},\n"
              f"\tyaw_angle: {self.yaw_angle},\n"
              f"\tpitch_angle: {self.pitch_angle}")

        # TODO: image_array should be an np.array from NeRF model
        # TODO: this method should call NeRF model for corresponding workspace

        image_array = np.random.randint(low=0, high=255, size=(600, 800, 3), dtype=np.uint8)

        # Convert the NumPy array into a QImage
        height, width, channels = image_array.shape
        qimage = QImage(image_array.data, width, height, width * channels, QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaledToWidth(700)

        # Setting the new rendered image onto the image label
        self.nerf_image.setPixmap(pixmap)

    def return_to_bev_view(self) -> None:
        """
        Removing all widgets from detailed workspace view and returning to the bird's eye view.
        """

        # Resetting coordinates and camera angles
        self.reset_coordinates()
        self.reset_angles()

        layout = self.centralWidget().layout()

        # Removing text labels and return to bird's eye view button
        layout.removeWidget(self.nerf_image)
        layout.removeWidget(self.camera_buttons_text_label)
        layout.removeWidget(self.return_bev_button)

        self.nerf_image.deleteLater()
        self.camera_buttons_text_label.deleteLater()
        self.return_bev_button.deleteLater()

        # Removing left, right, up and down buttons
        buttons_hbox_layout = layout.itemAt(layout.count() - 1).layout()

        while buttons_hbox_layout.count():
            button_widget = buttons_hbox_layout.takeAt(0).widget()
            button_widget.setParent(None)

        # Removing buttons layout
        layout.itemAt(layout.count() - 1).layout().deleteLater()

        # Restore the bird's-eye view image, description and instruction text labels
        layout.addWidget(self.bev_description_label)
        layout.addWidget(self.bev_label)
        layout.addWidget(self.bev_instruction_label)

    def left_button_clicked(self, event: Any) -> None:
        """
        Rendering new image on camera yaw angle change for -15 degrees.
        """

        self.yaw_angle -= self.angle_step if self.yaw_angle > -180 else 0
        self.render_nerf_image()

    def right_button_clicked(self, event: Any) -> None:
        """
        Rendering new image on camera yaw angle change for +15 degrees.
        """

        self.yaw_angle += self.angle_step if self.yaw_angle < 180 else 0
        self.render_nerf_image()

    def up_button_clicked(self, event: Any) -> None:
        """
        Rendering new image on camera pitch angle change for +15 degrees.
        """

        self.pitch_angle += self.angle_step if self.pitch_angle < 180 else 0
        self.render_nerf_image()

    def down_button_clicked(self, event: Any) -> None:
        """
        Rendering new image on camera pitch angle change for -15 degrees.
        """

        self.pitch_angle -= self.angle_step if self.pitch_angle > -180 else 0
        self.render_nerf_image()


class BirdsEyeViewImageArea(QLabel):
    left_click = pyqtSignal(float, float)

    def __init__(self, parent: QMainWindow) -> None:
        super().__init__(parent)

    def mousePressEvent(self, event: Any) -> None:

        x, y, = event.x(), event.y()

        if self.pixmap():
            label_size = self.size()
            pixmap_size = self.pixmap().size()
            width = pixmap_size.width()
            height = pixmap_size.height()

            x0 = int((label_size.width() - width) / 2)
            y0 = int((label_size.height() - height) / 2)

            # Check we clicked on the pixmap
            if (x >= x0 and x < (x0 + width) and
                    y >= y0 and y < (y0 + height)):
                # emit position relative to pixmap top-left corner
                x_relative = (x - x0) / width
                y_relative = (y - y0) / height
                self.left_click.emit(x_relative, y_relative)

        super().mousePressEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    landing_page = LandingPage()
    landing_page.show()
    sys.exit(app.exec_())

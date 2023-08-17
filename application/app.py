import os.path
from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QColor, QPainter, QFont, QIcon, QImage
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QWidget, QVBoxLayout, QGridLayout, \
    QHBoxLayout

from application.workspace import OfficeTokyoWorkspace, OfficeNewYorkWorkspace, OfficeGeneveWorkspace, \
    OfficeBelgradeWorkspace

WORKSPACES = [OfficeTokyoWorkspace(),
              OfficeNewYorkWorkspace(),
              OfficeGeneveWorkspace(),
              OfficeBelgradeWorkspace()]


class LandingPage(QMainWindow):
    """
    Landing Page for Workspaces Explorer Application.
    """

    def __init__(self) -> None:
        super().__init__()

        # Set the application window title
        self.setWindowTitle("Workspaces Explorer")
        # Set the fixed size of the window
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
        for i in range(len(WORKSPACES)):
            # Create a clickable label for each image
            label = QLabel(self)
            pixmap = QPixmap(os.path.join(WORKSPACES[i].folder_path, "thumbnail.jpg"))

            # Resize the image to fit the label width
            # pixmap = pixmap.scaledToWidth(400)
            pixmap = pixmap.scaledToWidth(300)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)

            # Store the index of the label
            label.index = i

            # Assign the click event over the image label
            label.mousePressEvent = lambda event, index=i: self._open_workspace_viewer(event, index)

            # Create a shaded background for the label
            label.setStyleSheet("background-color: rgba(0, 0, 0, 50);")

            # Add the label to the grid layout
            # grid_layout.addWidget(label, ((i // 4) * 2), i % 4)
            grid_layout.addWidget(label, ((i // 2) * 2), i % 2)

            # Create a title label for each image
            title_label = QLabel(WORKSPACES[i].name, self)
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

    def _open_workspace_viewer(self, event: Any, index: int) -> None:
        """
        By clicking on any workspace thumbnail WorkspaceExplorer page gets started.
        """

        self.hide()
        workspace_viewer = WorkspaceExplorer(self, workspace_index=index)
        workspace_viewer.show()


class WorkspaceExplorer(QMainWindow):

    def __init__(self, parent: QMainWindow, workspace_index: int) -> None:
        super().__init__(parent)

        # Workspace instance
        self._workspace = WORKSPACES[workspace_index]
        self._workspace.initialize_models()

        # Application window title
        self.setWindowTitle("Workspace Details")

        # Application window size
        self.setFixedSize(1000, 800)

        # Central widget and layout
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Buttons style sheet
        self._main_buttons_style_sheet = "QPushButton { background-color: #4CAF50; color: white; border: none; " \
                                         "padding: 10px; border-radius: 5px; font-weight: bold; } " \
                                         "QPushButton:hover { background-color: #45a049; }"

        self._camera_buttons_style_sheet = "QPushButton { background-color: #EEC10F; color: white; border: none; " \
                                           "padding: 10px; border-radius: 5px; font-weight: bold; } " \
                                           "QPushButton:hover { background-color: #CDA609; }"

        # Button for returning to the landing page
        return_landing_page_button = QPushButton("Explore another workspace", self)
        return_landing_page_button.clicked.connect(self._return_to_landing_page)
        return_landing_page_button.setMaximumWidth(300)
        return_landing_page_button.setMaximumHeight(50)
        return_landing_page_button.setStyleSheet(self._main_buttons_style_sheet)
        layout.addWidget(return_landing_page_button, alignment=Qt.AlignCenter | Qt.AlignTop)

        # Floor plan image description
        self._floor_plan_description_label = QLabel(f"Floor plan of the workspace '{self._workspace.name}'", self)
        self._floor_plan_description_label.setAlignment(Qt.AlignCenter)
        self._floor_plan_description_label.setFont(QFont("Arial", 12))
        self._floor_plan_description_label.setStyleSheet("color: white;")
        layout.addWidget(self._floor_plan_description_label)

        # Floor plan image
        self._floor_plan_label = FloorPlanImageArea(self)
        self._floor_plan_pixmap = QPixmap(os.path.join(self._workspace.folder_path, "floor_plan.jpg"))
        self._floor_plan_pixmap = self._floor_plan_pixmap.scaled(self._workspace.floor_plan_scale[1],
                                                                 self._workspace.floor_plan_scale[0])
        self._floor_plan_label.setPixmap(self._floor_plan_pixmap)
        self._floor_plan_label.setAlignment(Qt.AlignCenter)
        self._floor_plan_label.left_click.connect(self._render_detailed_workspace_view)
        layout.addWidget(self._floor_plan_label)

        # Floor plan instruction text
        self._floor_plan_instruction_label = QLabel(f"Click on the image for detailed in-place workspace view", self)
        self._floor_plan_instruction_label.setAlignment(Qt.AlignCenter)
        self._floor_plan_instruction_label.setFont(QFont("Arial", 10))
        self._floor_plan_instruction_label.setStyleSheet("color: white;")
        layout.addWidget(self._floor_plan_instruction_label)

        # Placeholder for render image label
        self._nerf_image = None

        # Placeholder for text label for camera turn buttons
        self._camera_buttons_text_label = None

        # Placeholders for left-right buttons
        self._left_button = None
        self._right_button = None

        # Placeholders for up-down buttons
        self._up_button = None
        self._down_button = None

        # Placeholder for button for returning to floor plan view
        self._return_to_floor_plan_button = None

        # Camera coordinates - relative x, y
        # Relative coordinates x and y have range [0, 1], where starting point on
        # floor plan image is top left corner
        # Relative coordinate x goes along left-to-right axis, and
        # Relative coordinate y goes along top-to-bottom axis
        self._rel_x = 0.0
        self._rel_y = 0.0

        # Camera angles [deg]
        self._horizontal_angle = 0
        self._vertical_angle = 0
        self._angle_step = 15

    def _reset_coordinates(self):
        """
        Resetting coordinates of camera view.
        """

        self._rel_x = 0.0
        self._rel_y = 0.0

    def _reset_angles(self):
        """
        Resetting angles for camera view.
        """

        self._horizontal_angle = 0
        self._vertical_angle = 0

    def paintEvent(self, event: Any) -> None:
        """
        Overriding the paintEvent to set the background color.
        """

        painter = QPainter(self)

        # Set a smooth black color (dark gray)
        painter.setBrush(QColor(80, 80, 90))
        painter.drawRect(0, 0, self.width(), self.height())

        super().paintEvent(event)

    def _return_to_landing_page(self) -> None:
        """
        By clicking the 'Explore another workspace' button landing page shows up.
        """

        self.parent().show()
        self.close()

    def _render_detailed_workspace_view(self, x: float, y: float) -> None:
        """
        Rendering detailed workspace view - in-place viewing of workspaces interior.
        """

        self._rel_x = x
        self._rel_y = y

        # Gathering the central widget
        layout = self.centralWidget().layout()

        # Removing the floor plan image, description and instruction text
        layout.removeWidget(self._floor_plan_label)
        layout.removeWidget(self._floor_plan_description_label)
        layout.removeWidget(self._floor_plan_instruction_label)

        self._floor_plan_label.setParent(None)
        self._floor_plan_description_label.setParent(None)
        self._floor_plan_instruction_label.setParent(None)

        # Creating a NeRF image label and displaying the new image
        self._nerf_image = QLabel(self)
        self._nerf_image.setAlignment(Qt.AlignCenter)
        self._render_nerf_image()

        layout.addWidget(self._nerf_image)

        # Description of camera buttons
        self._camera_buttons_text_label = QLabel(f"Turn camera by clicking buttons bellow", self)
        self._camera_buttons_text_label.setAlignment(Qt.AlignCenter)
        self._camera_buttons_text_label.setFont(QFont("Arial", 10))
        self._camera_buttons_text_label.setStyleSheet("color: white;")
        layout.addWidget(self._camera_buttons_text_label)

        buttons_layout = QHBoxLayout(self)

        # Button for turning the 'camera' to the left
        self._left_button = QPushButton(self)
        self._left_button.setMaximumWidth(200)
        self._left_button.setMaximumHeight(40)
        self._left_button.setStyleSheet(self._camera_buttons_style_sheet)
        self._left_button.setIcon(QIcon("application/imgs/left_arrow.png"))
        self._left_button.setIconSize(self._left_button.size())
        self._left_button.clicked.connect(self._left_button_clicked)
        buttons_layout.addWidget(self._left_button)

        # Button for turning the 'camera' to the right
        self._right_button = QPushButton(self)
        self._right_button.setMaximumWidth(200)
        self._right_button.setMaximumHeight(40)
        self._right_button.setStyleSheet(self._camera_buttons_style_sheet)
        self._right_button.setIcon(QIcon("application/imgs/rigth_arrow.png"))
        self._right_button.setIconSize(self._right_button.size())
        self._right_button.clicked.connect(self._right_button_clicked)
        buttons_layout.addWidget(self._right_button)

        # Button for turning the 'camera' up
        self._up_button = QPushButton(self)
        self._up_button.setMaximumWidth(200)
        self._up_button.setMaximumHeight(40)
        self._up_button.setStyleSheet(self._camera_buttons_style_sheet)
        self._up_button.setIcon(QIcon("application/imgs/up_arrow.png"))
        self._up_button.setIconSize(self._up_button.size())
        self._up_button.clicked.connect(self._up_button_clicked)
        buttons_layout.addWidget(self._up_button)

        # Button for turning the 'camera' down
        self._down_button = QPushButton(self)
        self._down_button.setMaximumWidth(200)
        self._down_button.setMaximumHeight(40)
        self._down_button.setStyleSheet(self._camera_buttons_style_sheet)
        self._down_button.setIcon(QIcon("application/imgs/down_arrow.png"))
        self._down_button.setIconSize(self._down_button.size())
        self._down_button.clicked.connect(self._down_button_clicked)
        buttons_layout.addWidget(self._down_button)

        layout.addLayout(buttons_layout)

        # Button for returning to the floor plan image
        self._return_to_floor_plan_button = QPushButton("Back to Floor Plan", self)
        self._return_to_floor_plan_button.clicked.connect(self._return_to_floor_plan_view)
        self._return_to_floor_plan_button.setMaximumWidth(200)
        self._return_to_floor_plan_button.setMaximumHeight(50)
        self._return_to_floor_plan_button.setStyleSheet(self._main_buttons_style_sheet)
        layout.addWidget(self._return_to_floor_plan_button, alignment=Qt.AlignCenter | Qt.AlignBottom)

    def _render_nerf_image(self):
        """
        Rendering image for each particular view point using NeRF model pretrained for each workspace.
        """

        print(f"-----------------------------------------------------------------------------------------\n"
              f"Rendering new NeRF view from following relative coordinates: \n"
              f"\trelative X: {self._rel_x:3f},\n"
              f"\trelative Y: {self._rel_y:3f},\n"
              f"\thorizontal angle: {self._horizontal_angle},\n"
              f"\tvertical angle: {self._vertical_angle}\n"
              f"-----------------------------------------------------------------------------------------")

        image = self._workspace.render_image(self._rel_x, self._rel_y, self._horizontal_angle, self._vertical_angle)

        # Convert the NumPy array into a QImage
        height, width, channels = image.shape
        qimage = QImage(image.data, width, height, width * channels, QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaledToWidth(700)

        # Setting the new rendered image onto the image label
        self._nerf_image.setPixmap(pixmap)

    def _return_to_floor_plan_view(self) -> None:
        """
        Removing all widgets from detailed workspace view and returning to the floor plan view.
        """

        # Resetting coordinates and camera angles
        self._reset_coordinates()
        self._reset_angles()

        layout = self.centralWidget().layout()

        # Removing text labels and return to floor plan view button
        layout.removeWidget(self._nerf_image)
        layout.removeWidget(self._camera_buttons_text_label)
        layout.removeWidget(self._return_to_floor_plan_button)

        self._nerf_image.deleteLater()
        self._camera_buttons_text_label.deleteLater()
        self._return_to_floor_plan_button.deleteLater()

        # Removing left, right, up and down buttons
        buttons_hbox_layout = layout.itemAt(layout.count() - 1).layout()

        while buttons_hbox_layout.count():
            button_widget = buttons_hbox_layout.takeAt(0).widget()
            button_widget.setParent(None)

        # Removing buttons layout
        layout.itemAt(layout.count() - 1).layout().deleteLater()

        # Restore the floor plan image, description and instruction text labels
        layout.addWidget(self._floor_plan_description_label)
        layout.addWidget(self._floor_plan_label)
        layout.addWidget(self._floor_plan_instruction_label)

    def _left_button_clicked(self, event: Any) -> None:
        """
        Rendering new image on camera horizontal movement for -15 degrees.
        """

        self._horizontal_angle -= self._angle_step if self._horizontal_angle > -180 else 0
        self._render_nerf_image()

    def _right_button_clicked(self, event: Any) -> None:
        """
        Rendering new image on camera horizontal movement for +15 degrees.
        """

        self._horizontal_angle += self._angle_step if self._horizontal_angle < 180 else 0
        self._render_nerf_image()

    def _up_button_clicked(self, event: Any) -> None:
        """
        Rendering new image on camera vertical movement for +15 degrees.
        """

        self._vertical_angle += self._angle_step if self._vertical_angle < 180 else 0
        self._render_nerf_image()

    def _down_button_clicked(self, event: Any) -> None:
        """
        Rendering new image on camera vertical movement for -15 degrees.
        """

        self._vertical_angle -= self._angle_step if self._vertical_angle > -180 else 0
        self._render_nerf_image()


class FloorPlanImageArea(QLabel):
    """
    Clickable area over floor plan image.
    """

    left_click = pyqtSignal(float, float)

    def __init__(self, parent: QMainWindow) -> None:
        super().__init__(parent)

    def mousePressEvent(self, event: Any) -> None:
        """
        Catching the mouse-press event over floor-plan image area.
        """

        x, y, = event.x(), event.y()

        if self.pixmap():
            label_size = self.size()
            pixmap_size = self.pixmap().size()
            width = pixmap_size.width()
            height = pixmap_size.height()

            x0 = int((label_size.width() - width) / 2)
            y0 = int((label_size.height() - height) / 2)

            if (x >= x0 and x < (x0 + width) and y >= y0 and y < (y0 + height)):
                x_relative = (x - x0) / width
                y_relative = (y - y0) / height
                self.left_click.emit(x_relative, y_relative)

        super().mousePressEvent(event)

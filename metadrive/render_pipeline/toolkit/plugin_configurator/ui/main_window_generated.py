# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 680)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 680))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 680))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/res/icon.png"), QtGui.QIcon.Normal,
                       QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet(
            "QMainWindow { background: #fff;}\n"
            "*, QLabel { font-family: Roboto; font-weight: 300; }\n"
            "\n"
            "QScrollBar {\n"
            "    background: #eee;\n"
            "}\n"
            "\n"
            "\n"
            "QScrollBar:vertical {\n"
            "                        width: 9px;\n"
            "    margin: 0;\n"
            "                      }\n"
            "\n"
            "                      QScrollBar::handle:vertical {\n"
            "                        min-height: 15px;\n"
            "                        background: #aaa;\n"
            "\n"
            "                      }\n"
            "\n"
            "\n"
            "                      QScrollBar::handle:vertical:hover {\n"
            "                        \n"
            "                        background: #999;\n"
            "\n"
            "                      }\n"
            "                      QScrollBar::add-line:vertical {\n"
            "                      }\n"
            "\n"
            "                      QScrollBar::sub-line:vertical {\n"
            "                      }\n"
            "\n"
            "\n"
            "                      QScrollBar::add-page:vertical {\n"
            "                        background: #ddd;\n"
            "\n"
            "                      }\n"
            "                      QScrollBar::sub-page:vertical {\n"
            "                        background: #ddd;\n"
            "                    }\n"
            "")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lst_plugins = QtWidgets.QListWidget(self.centralwidget)
        self.lst_plugins.setGeometry(QtCore.QRect(20, 80, 251, 531))
        self.lst_plugins.setStyleSheet("QListWidget::item {\n"
                                       "padding: 8px 7px 5px;\n"
                                       "background: #ddd;\n"
                                       "outline: 0 !important;\n"
                                       "margin-bottom: 1px;\n"
                                       "margin-right: 10px;\n"
                                       "color: #777;\n"
                                       "font-weight: 500;\n"
                                       "border: 0;\n"
                                       "border-radius: 0;\n"
                                       "}\n"
                                       "\n"
                                       "QListWidget::item:hover {\n"
                                       "background: #ccc;\n"
                                       "}\n"
                                       "\n"
                                       "QListWidget::item:selected {\n"
                                       "background: #555;\n"
                                       "color: #eee;\n"
                                       "}\n"
                                       "\n"
                                       "QListWidget {\n"
                                       "padding: 0;\n"
                                       "color: #eee;\n"
                                       "background: transparent;\n"
                                       "/*\n"
                                       "background: #222;\n"
                                       "padding: 5px 8px;\n"
                                       "border: 1px solid #444;\n"
                                       "*/\n"
                                       "}\n"
                                       "\n"
                                       "* {\n"
                                       "outline: 0;\n"
                                       "}\n"
                                       "\n"
                                       "")
        self.lst_plugins.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lst_plugins.setLineWidth(0)
        self.lst_plugins.setObjectName("lst_plugins")
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.lst_plugins.addItem(item)
        self.frame_details = QtWidgets.QFrame(self.centralwidget)
        self.frame_details.setGeometry(QtCore.QRect(281, 60, 721, 621))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(229, 229, 229))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(52, 52, 52))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(28, 28, 28))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(229, 229, 229))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(229, 229, 229))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(229, 229, 229))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(52, 52, 52))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(28, 28, 28))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(229, 229, 229))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(229, 229, 229))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(229, 229, 229))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(52, 52, 52))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(28, 28, 28))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(229, 229, 229))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(229, 229, 229))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 42))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText,
                         brush)
        self.frame_details.setPalette(palette)
        self.frame_details.setAutoFillBackground(False)
        self.frame_details.setStyleSheet(
            "#frame_details { background: #e5e5e5; }")
        self.frame_details.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_details.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_details.setLineWidth(1)
        self.frame_details.setObjectName("frame_details")
        self.lbl_plugin_version = QtWidgets.QLabel(self.frame_details)
        self.lbl_plugin_version.setGeometry(QtCore.QRect(20, 40, 391, 21))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 151, 25))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(236, 236, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 151, 25))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 151, 25))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(236, 236, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(236, 236, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 151, 25))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(236, 236, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 151, 25))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 151, 25))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(236, 236, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(236, 236, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 151, 25))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(236, 236, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 151, 25))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 151, 25))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(236, 236, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(236, 236, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.lbl_plugin_version.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setBold(True)
        font.setWeight(75)
        self.lbl_plugin_version.setFont(font)
        self.lbl_plugin_version.setStyleSheet(
            "color: #ff9719; border: 0; border-radius: 4px; font-weight: bold;")
        self.lbl_plugin_version.setTextFormat(QtCore.Qt.PlainText)
        self.lbl_plugin_version.setAlignment(QtCore.Qt.AlignLeading |
                                             QtCore.Qt.AlignLeft |
                                             QtCore.Qt.AlignVCenter)
        self.lbl_plugin_version.setObjectName("lbl_plugin_version")
        self.lbl_plugin_desc = QtWidgets.QLabel(self.frame_details)
        self.lbl_plugin_desc.setGeometry(QtCore.QRect(20, 60, 441, 51))
        self.lbl_plugin_desc.setStyleSheet(
            "color: #777; border: 0; font-size: 11px;")
        self.lbl_plugin_desc.setTextFormat(QtCore.Qt.PlainText)
        self.lbl_plugin_desc.setAlignment(QtCore.Qt.AlignLeading |
                                          QtCore.Qt.AlignLeft |
                                          QtCore.Qt.AlignTop)
        self.lbl_plugin_desc.setWordWrap(True)
        self.lbl_plugin_desc.setObjectName("lbl_plugin_desc")
        self.table_plugin_settings = QtWidgets.QTableWidget(self.frame_details)
        self.table_plugin_settings.setGeometry(QtCore.QRect(20, 110, 681, 491))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(247, 247, 247))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 160))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(247, 247, 247))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(247, 247, 247))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 160))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(247, 247, 247))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(247, 247, 247))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(160, 160, 160))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText,
                         brush)
        self.table_plugin_settings.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(-1)
        font.setBold(False)
        font.setWeight(37)
        self.table_plugin_settings.setFont(font)
        self.table_plugin_settings.setStyleSheet(
            "QTableWidget { border: 1px solid #ccc; font-family: Roboto; font-size: 11px; }"
        )
        self.table_plugin_settings.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.table_plugin_settings.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded)
        self.table_plugin_settings.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.table_plugin_settings.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_plugin_settings.setProperty("showDropIndicator", False)
        self.table_plugin_settings.setDragDropOverwriteMode(False)
        self.table_plugin_settings.setAlternatingRowColors(True)
        self.table_plugin_settings.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection)
        self.table_plugin_settings.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.table_plugin_settings.setTextElideMode(QtCore.Qt.ElideNone)
        self.table_plugin_settings.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table_plugin_settings.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table_plugin_settings.setShowGrid(False)
        self.table_plugin_settings.setGridStyle(QtCore.Qt.NoPen)
        self.table_plugin_settings.setWordWrap(True)
        self.table_plugin_settings.setCornerButtonEnabled(True)
        self.table_plugin_settings.setObjectName("table_plugin_settings")
        self.table_plugin_settings.setColumnCount(4)
        self.table_plugin_settings.setRowCount(2)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter |
                              QtCore.Qt.AlignCenter)
        self.table_plugin_settings.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter |
                              QtCore.Qt.AlignCenter)
        self.table_plugin_settings.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setItem(1, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_plugin_settings.setItem(1, 3, item)
        self.table_plugin_settings.horizontalHeader().setVisible(True)
        self.table_plugin_settings.horizontalHeader(
        ).setCascadingSectionResizes(False)
        self.table_plugin_settings.horizontalHeader().setHighlightSections(True)
        self.table_plugin_settings.horizontalHeader().setSortIndicatorShown(
            False)
        self.table_plugin_settings.horizontalHeader().setStretchLastSection(
            True)
        self.table_plugin_settings.verticalHeader().setVisible(False)
        self.table_plugin_settings.verticalHeader().setCascadingSectionResizes(
            False)
        self.table_plugin_settings.verticalHeader().setDefaultSectionSize(45)
        self.table_plugin_settings.verticalHeader().setSortIndicatorShown(False)
        self.table_plugin_settings.verticalHeader().setStretchLastSection(False)
        self.lbl_plugin_name = QtWidgets.QLabel(self.frame_details)
        self.lbl_plugin_name.setGeometry(QtCore.QRect(20, 20, 391, 21))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(-1)
        font.setBold(False)
        font.setWeight(37)
        self.lbl_plugin_name.setFont(font)
        self.lbl_plugin_name.setStyleSheet(
            "color: #555; border: 0;  font-size: 15px;")
        self.lbl_plugin_name.setTextFormat(QtCore.Qt.RichText)
        self.lbl_plugin_name.setObjectName("lbl_plugin_name")
        self.btn_reset_plugin_settings = QtWidgets.QPushButton(
            self.frame_details)
        self.btn_reset_plugin_settings.setGeometry(
            QtCore.QRect(520, 60, 180, 31))
        self.btn_reset_plugin_settings.setStyleSheet("QPushButton {\n"
                                                     "color: #eee;\n"
                                                     "background: #666;\n"
                                                     "border: 0; \n"
                                                     "}\n"
                                                     "\n"
                                                     "QPushButton:hover {\n"
                                                     "    background: #555;\n"
                                                     "}\n"
                                                     "\n"
                                                     "QPushButton:pressed {\n"
                                                     "    background: #111;\n"
                                                     "}")
        self.btn_reset_plugin_settings.setObjectName(
            "btn_reset_plugin_settings")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 60, 281, 621))
        self.frame.setAutoFillBackground(False)
        self.frame.setStyleSheet("QFrame {background: #eee;}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.lbl_hint_restart = QtWidgets.QLabel(self.frame)
        self.lbl_hint_restart.setGeometry(QtCore.QRect(20, 560, 251, 51))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(153, 153, 153))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 238))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(153, 153, 153))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(153, 153, 153))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 238))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 238))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(153, 153, 153))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 238))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(153, 153, 153))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(153, 153, 153))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 238))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 238))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(153, 153, 153))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 238))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(153, 153, 153))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(153, 153, 153))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText,
                         brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 238))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 238))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.lbl_hint_restart.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setBold(False)
        font.setWeight(37)
        self.lbl_hint_restart.setFont(font)
        self.lbl_hint_restart.setStyleSheet(
            "border-top: 1px dotted #aaa; color: #999; padding: 0;")
        self.lbl_hint_restart.setTextFormat(QtCore.Qt.PlainText)
        self.lbl_hint_restart.setScaledContents(False)
        self.lbl_hint_restart.setAlignment(QtCore.Qt.AlignLeading |
                                           QtCore.Qt.AlignLeft |
                                           QtCore.Qt.AlignVCenter)
        self.lbl_hint_restart.setWordWrap(True)
        self.lbl_hint_restart.setObjectName("lbl_hint_restart")
        self.lbl_restart_pipeline = QtWidgets.QLabel(self.centralwidget)
        self.lbl_restart_pipeline.setGeometry(QtCore.QRect(687, 18, 294, 28))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(-1)
        font.setBold(False)
        font.setWeight(37)
        self.lbl_restart_pipeline.setFont(font)
        self.lbl_restart_pipeline.setAutoFillBackground(False)
        self.lbl_restart_pipeline.setStyleSheet(
            "background: #ff9719; color: #eee; padding: 5px; border: 0; font-size: 12px;"
        )
        self.lbl_restart_pipeline.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_restart_pipeline.setObjectName("lbl_restart_pipeline")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 25, 261, 31))
        self.label.setStyleSheet("font-size: 20px; color: #ff9719;\n"
                                 "font-weight: 100;")
        self.label.setObjectName("label")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(20, 6, 261, 31))
        self.label_9.setStyleSheet("font-size: 13px;\n"
                                   "color: #aaa;\n"
                                   "font-weight: 600;")
        self.label_9.setObjectName("label_9")
        self.frame.raise_()
        self.lst_plugins.raise_()
        self.lbl_restart_pipeline.raise_()
        self.frame_details.raise_()
        self.label.raise_()
        self.label_9.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(
            _translate("MainWindow", "Plugin Configurator"))
        __sortingEnabled = self.lst_plugins.isSortingEnabled()
        self.lst_plugins.setSortingEnabled(False)
        item = self.lst_plugins.item(0)
        item.setText(_translate("MainWindow", "Ambient Occlusion"))
        item = self.lst_plugins.item(1)
        item.setText(_translate("MainWindow", "Another Ite"))
        item = self.lst_plugins.item(2)
        item.setText(_translate("MainWindow", "Bloom"))
        item = self.lst_plugins.item(3)
        item.setText(_translate("MainWindow", "Volumetric Clouds"))
        item = self.lst_plugins.item(4)
        item.setText(_translate("MainWindow", "Depth of Field"))
        item = self.lst_plugins.item(5)
        item.setText(_translate("MainWindow", "Environment Probes"))
        item = self.lst_plugins.item(6)
        item.setText(_translate("MainWindow", "FXAA (Antialiasing)"))
        item = self.lst_plugins.item(7)
        item.setText(_translate("MainWindow", "SKAA"))
        item = self.lst_plugins.item(8)
        item.setText(_translate("MainWindow", "Motion Blur"))
        item = self.lst_plugins.item(9)
        item.setText(_translate("MainWindow", "PSSM Shadows"))
        item = self.lst_plugins.item(10)
        item.setText(_translate("MainWindow", "Atmospheric Scattering"))
        item = self.lst_plugins.item(11)
        item.setText(_translate("MainWindow", "Skin Shading"))
        item = self.lst_plugins.item(12)
        item.setText(_translate("MainWindow", "SMAA (Antialiasing)"))
        item = self.lst_plugins.item(13)
        item.setText(_translate("MainWindow", "Screen Space Reflections"))
        item = self.lst_plugins.item(14)
        item.setText(_translate("MainWindow", "Volumetric Lighting"))
        item = self.lst_plugins.item(15)
        item.setText(_translate("MainWindow", "VXGI"))
        item = self.lst_plugins.item(16)
        item.setText(_translate("MainWindow", "Item 346"))
        item = self.lst_plugins.item(17)
        item.setText(_translate("MainWindow", "Other Item"))
        item = self.lst_plugins.item(18)
        item.setText(_translate("MainWindow", "Other Item to force scrollbars"))
        self.lst_plugins.setSortingEnabled(__sortingEnabled)
        self.lbl_plugin_version.setText(
            _translate("MainWindow", "version 0.1 by some author"))
        self.lbl_plugin_desc.setText(
            _translate(
                "MainWindow", "This is a fancy description\n"
                "It shows information about the plugin and maybe\n"
                "a website or so."))
        item = self.table_plugin_settings.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Row"))
        item = self.table_plugin_settings.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Row2"))
        item = self.table_plugin_settings.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Setting"))
        item = self.table_plugin_settings.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Default"))
        item = self.table_plugin_settings.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Current Value"))
        item = self.table_plugin_settings.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Description"))
        __sortingEnabled = self.table_plugin_settings.isSortingEnabled()
        self.table_plugin_settings.setSortingEnabled(False)
        item = self.table_plugin_settings.item(0, 0)
        item.setText(
            _translate("MainWindow", "Long setting name which requires"))
        item = self.table_plugin_settings.item(0, 1)
        item.setText(_translate("MainWindow", "DefaultRow0"))
        item = self.table_plugin_settings.item(0, 2)
        item.setText(_translate("MainWindow", "CurrentRow0"))
        item = self.table_plugin_settings.item(0, 3)
        item.setText(_translate("MainWindow", "DescRow0"))
        item = self.table_plugin_settings.item(1, 1)
        item.setText(_translate("MainWindow", "DefaultRow1"))
        item = self.table_plugin_settings.item(1, 2)
        item.setText(_translate("MainWindow", "CurrentRow1"))
        item = self.table_plugin_settings.item(1, 3)
        item.setText(
            _translate(
                "MainWindow",
                "aasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfgaasd\\n\\nsdf\\n\\ndfdfG\\n\\nfg"
            ))
        self.table_plugin_settings.setSortingEnabled(__sortingEnabled)
        self.lbl_plugin_name.setText(
            _translate("MainWindow", "Ambient Occlusion"))
        self.btn_reset_plugin_settings.setText(
            _translate("MainWindow", "Reset Settings of this Plugin"))
        self.lbl_hint_restart.setText(
            _translate(
                "MainWindow",
                "Hint: Settings with a gray color require a pipeline restart when changed."
            ))
        self.lbl_restart_pipeline.setText(
            _translate("MainWindow",
                       "Pipeline needs to be restarted to apply all changes!"))
        self.label.setText(_translate("MainWindow", "PLUGIN CONFIGURATOR"))
        self.label_9.setText(_translate("MainWindow", "RENDER PIPELINE"))


from . import resources_rc

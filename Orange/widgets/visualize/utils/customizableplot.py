import sys
from typing import Tuple, List, Dict, Callable, Iterable

from AnyQt.QtGui import QFont, QFontDatabase
from AnyQt.QtWidgets import QApplication

import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import ItemSample

from orangewidget.utils.visual_settings_dlg import KeyType, ValueType, \
    SettingsType

_SettingType = Dict[str, ValueType]
_LegendItemType = Tuple[ItemSample, pg.LabelItem]


def available_font_families() -> List:
    """
    Function returns list of available font families.
    Can be used to instantiate font combo boxes.

    Returns
    -------
    fonts: list
        List of available font families.
    """
    if not QApplication.instance():
        _ = QApplication(sys.argv)
    return QFontDatabase().families()


def default_font_family() -> str:
    """
    Function returns default font family used in Qt application.
    Can be used to instantiate initial dialog state.

    Returns
    -------
    font: str
        Default font family.
    """
    if not QApplication.instance():
        _ = QApplication(sys.argv)
    return QFont().family()


def default_font_size() -> int:
    """
    Function returns default font size in points used in Qt application.
    Can be used to instantiate initial dialog state.

    Returns
    -------
    size: int
        Default font size in points.
    """
    if not QApplication.instance():
        _ = QApplication(sys.argv)
    return QFont().pointSize()


class Updater:
    """ Class with helper functions and constants. """
    FONT_FAMILY_LABEL, SIZE_LABEL, IS_ITALIC_LABEL = \
        "Font family", "Font size", "Italic"
    FONT_FAMILY_SETTING: SettingsType = {
        FONT_FAMILY_LABEL: (available_font_families(), default_font_family()),
    }
    FONT_SETTING: SettingsType = {
        SIZE_LABEL: (range(4, 50), default_font_size()),
        IS_ITALIC_LABEL: (None, False)
    }

    @staticmethod
    def update_plot_title_text(title_item: pg.LabelItem, text: str):
        title_item.text = text
        title_item.setVisible(bool(text))
        title_item.item.setPlainText(text)
        Updater.plot_title_resize(title_item)

    @staticmethod
    def update_plot_title_font(title_item: pg.LabelItem,
                               **settings: _SettingType):
        font = Updater.change_font(title_item.item.font(), settings)
        title_item.item.setFont(font)
        title_item.item.setPlainText(title_item.text)
        Updater.plot_title_resize(title_item)

    @staticmethod
    def plot_title_resize(title_item):
        height = title_item.item.boundingRect().height() + 6 \
            if title_item.text else 0
        title_item.setMaximumHeight(height)
        title_item.parentItem().layout.setRowFixedHeight(0, height)
        title_item.resizeEvent(None)

    @staticmethod
    def update_axis_title_text(item: pg.AxisItem, text: str):
        item.setLabel(text)
        item.resizeEvent(None)

    @staticmethod
    def update_axes_titles_font(items: List[pg.AxisItem],
                                **settings: _SettingType):
        for item in items:
            font = Updater.change_font(item.label.font(), settings)
            item.label.setFont(font)
            fstyle = ["normal", "italic"][font.italic()]
            style = {"font-size": f"{font.pointSize()}pt",
                     "font-family": f"{font.family()}",
                     "font-style": f"{fstyle}"}
            item.setLabel(None, None, None, **style)

    @staticmethod
    def update_axes_ticks_font(items: List[pg.AxisItem],
                               **settings: _SettingType):
        for item in items:
            font = item.style["tickFont"] or QFont()
            # remove when contained in setTickFont() - version 0.11.0
            item.style['tickFont'] = font
            item.setTickFont(Updater.change_font(font, settings))

    @staticmethod
    def update_legend_font(items: Iterable[_LegendItemType],
                           **settings: _SettingType):
        for sample, label in items:
            sample.setFixedHeight(sample.height())
            sample.setFixedWidth(sample.width())
            label.item.setFont(Updater.change_font(label.item.font(), settings))
            bounds = label.itemRect()
            label.setMaximumWidth(bounds.width())
            label.setMaximumHeight(bounds.height())
            label.updateMin()
            label.resizeEvent(None)
            label.updateGeometry()

    @staticmethod
    def update_num_legend_font(legend: pg.LegendItem,
                               **settings: _SettingType):
        if not legend:
            return
        for sample, label in legend.items:
            sample.set_font(Updater.change_font(sample.font, settings))
            legend.setGeometry(sample.boundingRect())

    @staticmethod
    def update_label_font(items: List[pg.TextItem], font: QFont):
        for item in items:
            item.setFont(font)

    @staticmethod
    def change_font(font: QFont, settings: _SettingType) -> QFont:
        assert all(s in (Updater.FONT_FAMILY_LABEL, Updater.SIZE_LABEL,
                         Updater.IS_ITALIC_LABEL) for s in settings), settings

        family = settings.get(Updater.FONT_FAMILY_LABEL)
        if family is not None:
            font.setFamily(family)
        size = settings.get(Updater.SIZE_LABEL)
        if size is not None:
            font.setPointSize(size)
        italic = settings.get(Updater.IS_ITALIC_LABEL)
        if italic is not None:
            font.setItalic(italic)
        return font


class BaseParameterSetter:
    """ Subclass to add 'setter' functionality to a plot. """
    LABELS_BOX = "Fonts"
    ANNOT_BOX = "Annotations"

    FONT_FAMILY_LABEL = "Font family"
    AXIS_TITLE_LABEL = "Axis title"
    AXIS_TICKS_LABEL = "Axis ticks"
    LEGEND_LABEL = "Legend"
    LABEL_LABEL = "Label"
    X_AXIS_LABEL = "x-axis title"
    Y_AXIS_LABEL = "y-axis title"
    TITLE_LABEL = "Title"

    initial_settings: Dict[str, Dict[str, SettingsType]] = NotImplemented

    def __init__(self):
        self._setters: Dict[str, Dict[str, Callable]] = NotImplemented

    @property
    def setters(self) -> Dict:
        return self._setters

    @setters.setter
    def setters(self, setters: Dict[str, Dict[str, Callable]]):
        assert setters.keys() == self.initial_settings.keys()
        assert all(setters[key].keys() == self.initial_settings[key].keys()
                   for key in setters.keys())

        self._setters = setters

    def set_parameter(self, key: KeyType, value: ValueType):
        self.setters[key[0]][key[1]](**{key[2]: value})
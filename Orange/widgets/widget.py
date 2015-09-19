import sys
import time
import os
import warnings
import types
from functools import reduce

from PyQt4.QtCore import QByteArray, Qt, pyqtSignal as Signal, pyqtProperty,\
    QEventLoop
from PyQt4.QtGui import QDialog, QPixmap, QLabel, QVBoxLayout, QSizePolicy, \
    qApp, QFrame, QStatusBar, QHBoxLayout, QStyle, QApplication, QGraphicsScene
from pyqtgraph import PlotWidget, PlotItem

from Orange.data import Table
from Orange.widgets import settings, gui
from Orange.canvas.registry import description as widget_description
from Orange.widgets.gui import ControlledAttributesDict, notify_changed
from Orange.widgets.settings import SettingsHandler


def _asmappingproxy(mapping):
    if isinstance(mapping, types.MappingProxyType):
        return mapping
    else:
        return types.MappingProxyType(mapping)


class WidgetMetaClass(type(QDialog)):
    """Meta class for widgets. If the class definition does not have a
       specific settings handler, the meta class provides a default one
       that does not handle contexts. Then it scans for any attributes
       of class settings.Setting: the setting is stored in the handler and
       the value of the attribute is replaced with the default."""

    #noinspection PyMethodParameters
    def __new__(mcs, name, bases, dict):
        from Orange.canvas.registry.description import (
            input_channel_from_args, output_channel_from_args)

        cls = type.__new__(mcs, name, bases, dict)
        if not cls.name: # not a widget
            return cls

        cls.inputs = [input_channel_from_args(inp) for inp in cls.inputs]
        cls.outputs = [output_channel_from_args(outp) for outp in cls.outputs]

        for inp in cls.inputs:
            if not hasattr(cls, inp.handler):
                raise AttributeError("missing input signal handler '{}' in {}".
                                     format(inp.handler, cls.name))

        # TODO Remove this when all widgets are migrated to Orange 3.0
        if (hasattr(cls, "settingsToWidgetCallback") or
                hasattr(cls, "settingsFromWidgetCallback")):
            raise TypeError("Reimplement settingsToWidgetCallback and "
                            "settingsFromWidgetCallback")

        cls.settingsHandler = SettingsHandler.create(cls, template=cls.settingsHandler)

        return cls


class OWWidget(QDialog, metaclass=WidgetMetaClass):
    # Global widget count
    widget_id = 0

    # Widget description
    name = None
    id = None
    category = None
    version = None
    description = None
    long_description = None
    icon = "icons/Unknown.png"
    priority = sys.maxsize
    author = None
    author_email = None
    maintainer = None
    maintainer_email = None
    help = None
    help_ref = None
    url = None
    keywords = []
    background = None
    replaces = None
    inputs = []
    outputs = []

    # Default widget layout settings
    want_basic_layout = True
    want_main_area = True
    want_control_area = True
    want_graph = False
    show_save_graph = True
    want_status_bar = False
    want_report = False
    report_html = ""

    save_position = True
    resizing_enabled = True

    widgetStateChanged = Signal(str, int, str)
    blockingStateChanged = Signal(bool)
    progressBarValueChanged = Signal(float)
    processingStateChanged = Signal(int)

    settingsHandler = None
    """:type: SettingsHandler"""

    savedWidgetGeometry = settings.Setting(None)

    def __new__(cls, parent=None, *args, **kwargs):
        self = super().__new__(cls, None, cls.get_flags())
        QDialog.__init__(self, None, self.get_flags())

        stored_settings = kwargs.get('stored_settings', None)
        if self.settingsHandler:
            self.settingsHandler.initialize(self, stored_settings)

        self.signalManager = kwargs.get('signal_manager', None)
        self.__env = _asmappingproxy(kwargs.get("env", {}))

        setattr(self, gui.CONTROLLED_ATTRIBUTES, ControlledAttributesDict(self))
        self.__reportData = None

        OWWidget.widget_id += 1
        self.widget_id = OWWidget.widget_id

        if self.name:
            self.setCaption(self.name)

        self.setFocusPolicy(Qt.StrongFocus)

        self.startTime = time.time()    # used in progressbar

        self.widgetState = {"Info": {}, "Warning": {}, "Error": {}}

        self.__blocking = False
        # flag indicating if the widget's position was already restored
        self.__was_restored = False

        self.__progressBarValue = -1
        self.__progressState = 0
        self.__statusMessage = ""

        if self.want_basic_layout:
            self.insertLayout()

        return self

    def __init__(self, *args, **kwargs):
        """QDialog __init__ was already called in __new__,
        please do not call it here."""

    @classmethod
    def get_flags(cls):
        return (Qt.Window if cls.resizing_enabled
                else Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

    # noinspection PyAttributeOutsideInit
    def insertLayout(self):
        def createPixmapWidget(self, parent, iconName):
            w = QLabel(parent)
            parent.layout().addWidget(w)
            w.setFixedSize(16, 16)
            w.hide()
            if os.path.exists(iconName):
                w.setPixmap(QPixmap(iconName))
            return w

        self.setLayout(QVBoxLayout())
        self.layout().setMargin(2)

        self.warning_bar = gui.widgetBox(self, orientation="horizontal",
                                         margin=0, spacing=0)
        self.warning_icon = gui.widgetLabel(self.warning_bar, "")
        self.warning_label = gui.widgetLabel(self.warning_bar, "")
        self.warning_label.setStyleSheet("padding-top: 5px")
        self.warning_bar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        gui.rubber(self.warning_bar)
        self.warning_bar.setVisible(False)

        self.topWidgetPart = gui.widgetBox(self,
                                           orientation="horizontal", margin=0)
        self.leftWidgetPart = gui.widgetBox(self.topWidgetPart,
                                            orientation="vertical", margin=0)
        if self.want_main_area:
            self.leftWidgetPart.setSizePolicy(
                QSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding))
            self.leftWidgetPart.updateGeometry()
            self.mainArea = gui.widgetBox(self.topWidgetPart,
                                          orientation="vertical",
                                          sizePolicy=QSizePolicy(QSizePolicy.Expanding,
                                                                 QSizePolicy.Expanding),
                                          margin=0)
            self.mainArea.layout().setMargin(4)
            self.mainArea.updateGeometry()

        if self.want_control_area:
            self.controlArea = gui.widgetBox(self.leftWidgetPart,
                                             orientation="vertical", margin=4)

        if self.want_graph and self.show_save_graph:
            graphButtonBackground = gui.widgetBox(self.leftWidgetPart,
                                                  orientation="horizontal", margin=4)
            self.graphButton = gui.button(graphButtonBackground,
                                          self, "&Save Graph")
            self.graphButton.setAutoDefault(0)

        if self.want_status_bar:
            self.widgetStatusArea = QFrame(self)
            self.statusBarIconArea = QFrame(self)
            self.widgetStatusBar = QStatusBar(self)

            self.layout().addWidget(self.widgetStatusArea)

            self.widgetStatusArea.setLayout(QHBoxLayout(self.widgetStatusArea))
            self.widgetStatusArea.layout().addWidget(self.statusBarIconArea)
            self.widgetStatusArea.layout().addWidget(self.widgetStatusBar)
            self.widgetStatusArea.layout().setMargin(0)
            self.widgetStatusArea.setFrameShape(QFrame.StyledPanel)

            self.statusBarIconArea.setLayout(QHBoxLayout())
            self.widgetStatusBar.setSizeGripEnabled(0)

            self.statusBarIconArea.hide()

            self._warningWidget = createPixmapWidget(
                self.statusBarIconArea,
                gui.resource_filename("icons/triangle-orange.png"))
            self._errorWidget = createPixmapWidget(
                self.statusBarIconArea,
                gui.resource_filename("icons/triangle-red.png"))

        if not self.resizing_enabled:
            self.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

        if self.want_report:
            self.report_button_background = gui.widgetBox(
                self.leftWidgetPart,
                orientation="horizontal", margin=4
            )
            self.report_button = gui.button(
                self.report_button_background,
                self, "&Report", callback=self.show_report)
            self.report_button.setAutoDefault(0)

    def show_report(self):
        from Orange.canvas.report.owreport import OWReport

        report = OWReport.get_instance()
        self.create_report_html()
        report.make_report(self)
        report.show()

    def create_report_html(self):
        from Orange.canvas.report.owreport import OWReport

        self.report_html = OWReport.get_html_section(self.name)
        self.send_report()

    def send_report(self):
        if hasattr(self, "data") and isinstance(self.data, Table):
            self.report_data("Data", self.data)
        if hasattr(self, "canvas"):
            self.report_plot("", self.canvas)
        if hasattr(self, "box_scene"):
            self.report_plot("", self.box_scene)
        if hasattr(self, "plot"):
            self.report_plot("", self.plot)

    def report_settings(self, name, items):
        from Orange.canvas.report.owreport import OWReport

        if name != "":
            self.report_html += OWReport.get_html_subsection(name)
        self.report_html += OWReport.get_html_paragraph(items)

    def report_data(self, name, data):
        def clipped_list(items, s):
            from Orange.canvas.report.owreport import OWReport
            r = OWReport.clipped_list(a.name for a in items)
            if len(items) > 10:
                r += " (total: {} {})".format(len(items), s)
            return r

        if data is None:
            self.report_raw("No data.")
        else:
            items = [
                ("Data instances", len(data)),
                ("Features", clipped_list(data.domain.attributes, "features"))]
            if data.domain.metas:
                items.append(
                    ("Meta attributes",
                     clipped_list(data.domain.metas, "meta attributes")))
            if data.domain.class_vars:
                items.append(
                    ("Target",
                     clipped_list(data.domain.class_vars, "targets variables")))
            self.report_settings(name, items)

    def report_plot(self, name, plot):
        from Orange.canvas.report.owreport import OWReport

        if name != "":
            self.report_html += OWReport.get_html_subsection(name)
        if isinstance(plot, QGraphicsScene) or isinstance(plot, PlotItem):
            self.report_html += OWReport.get_html_img(plot)
        elif isinstance(plot, PlotWidget):
            self.report_html += OWReport.get_html_img(plot.plotItem)

    def report_table(self, name, table, head_column=True):
        from Orange.canvas.report.owreport import OWReport
        from PyQt4.QtGui import QTreeView

        t = ""
        if isinstance(table, QTreeView):
            model = table.model()
            rows = model.rowCount()
            columns = model.columnCount()
            t = "<table>\n"
            if not table.isHeaderHidden():
                t += "  <tr>\n"
                t += "".join("    <th>{}</th>\n".format(
                    model.horizontalHeaderItem(col).data(Qt.DisplayRole))
                    for col in range(columns))
                t += "  </tr>\n"

            for row in range(rows):
                t += "  <tr>\n"
                if head_column and columns:
                    t += "    <th>{}</th>".format(
                        model.item(row, 0).data(Qt.DisplayRole))
                t += "".join("    <td>{}</td>\n".format(
                    model.item(row, col).data(Qt.DisplayRole))
                             for col in range(head_column, columns))
                t += "  </tr>\n"
            t += "</table>"
        if name != "":
            self.report_html += OWReport.get_html_subsection(name)
        if t:
            self.report_html += t

    def report_raw(self, name, html):
        from Orange.canvas.report.owreport import OWReport

        if name != "":
            self.report_html += OWReport.get_html_subsection(name)
        self.report_html += html

    def updateStatusBarState(self):
        if not hasattr(self, "widgetStatusArea"):
            return
        if self.widgetState["Warning"] or self.widgetState["Error"]:
            self.widgetStatusArea.show()
        else:
            self.widgetStatusArea.hide()

    def setStatusBarText(self, text, timeout=5000):
        if hasattr(self, "widgetStatusBar"):
            self.widgetStatusBar.showMessage(" " + text, timeout)

    def __restoreWidgetGeometry(self):
        restored = False
        if self.save_position:
            geometry = self.savedWidgetGeometry
            if geometry is not None:
                restored = self.restoreGeometry(QByteArray(geometry))

            if restored and not self.windowState() & \
                    (Qt.WindowMaximized | Qt.WindowFullScreen):
                space = qApp.desktop().availableGeometry(self)
                frame, geometry = self.frameGeometry(), self.geometry()

                #Fix the widget size to fit inside the available space
                width = space.width() - (frame.width() - geometry.width())
                width = min(width, geometry.width())
                height = space.height() - (frame.height() - geometry.height())
                height = min(height, geometry.height())
                self.resize(width, height)

                # Move the widget to the center of available space if it is
                # currently outside it
                if not space.contains(self.frameGeometry()):
                    x = max(0, space.width() / 2 - width / 2)
                    y = max(0, space.height() / 2 - height / 2)

                    self.move(x, y)
        return restored

    def __updateSavedGeometry(self):
        if self.__was_restored and self.isVisible():
            # Update the saved geometry only between explicit show/hide
            # events (i.e. changes initiated by the user not by Qt's default
            # window management).
            self.savedWidgetGeometry = self.saveGeometry()

    # when widget is resized, save the new width and height
    def resizeEvent(self, ev):
        QDialog.resizeEvent(self, ev)
        # Don't store geometry if the widget is not visible
        # (the widget receives a resizeEvent (with the default sizeHint)
        # before first showEvent and we must not overwrite the the
        # savedGeometry with it)
        if self.save_position and self.isVisible():
            self.__updateSavedGeometry()

    def moveEvent(self, ev):
        QDialog.moveEvent(self, ev)
        if self.save_position and self.isVisible():
            self.__updateSavedGeometry()

    # set widget state to hidden
    def hideEvent(self, ev):
        if self.save_position:
            self.__updateSavedGeometry()
        QDialog.hideEvent(self, ev)

    def closeEvent(self, ev):
        if self.save_position and self.isVisible():
            self.__updateSavedGeometry()
        QDialog.closeEvent(self, ev)

    def showEvent(self, ev):
        QDialog.showEvent(self, ev)
        if self.save_position and not self.__was_restored:
            # Restore saved geometry on show
            self.__restoreWidgetGeometry()
            self.__was_restored = True

    def wheelEvent(self, event):
        """ Silently accept the wheel event. This is to ensure combo boxes
        and other controls that have focus don't receive this event unless
        the cursor is over them.
        """
        event.accept()

    def setCaption(self, caption):
        # we have to save caption title in case progressbar will change it
        self.captionTitle = str(caption)
        self.setWindowTitle(caption)

    # put this widget on top of all windows
    def reshow(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def send(self, signalName, value, id=None):
        if not any(s.name == signalName for s in self.outputs):
            raise ValueError('{} is not a valid output signal for widget {}'.format(
                signalName, self.name))
        if self.signalManager is not None:
            self.signalManager.send(self, signalName, value, id)

    def __setattr__(self, name, value):
        """Set value to members of this instance or any of its members.

        If member is used in a gui control, notify the control about the change.

        name: name of the member, dot is used for nesting ("graph.point.size").
        value: value to set to the member.

        """

        names = name.rsplit(".")
        field_name = names.pop()
        obj = reduce(lambda o, n: getattr(o, n, None), names, self)
        if obj is None:
            raise AttributeError("Cannot set '{}' to {} ".format(name, value))

        if obj is self:
            super().__setattr__(field_name, value)
        else:
            setattr(obj, field_name, value)

        notify_changed(obj, field_name, value)

        if self.settingsHandler:
            self.settingsHandler.fast_save(self, name, value)

    def openContext(self, *a):
        self.settingsHandler.open_context(self, *a)

    def closeContext(self):
        self.settingsHandler.close_context(self)

    def retrieveSpecificSettings(self):
        pass

    def storeSpecificSettings(self):
        pass

    def saveSettings(self):
        self.settingsHandler.update_defaults(self)

    def onDeleteWidget(self):
        """
        Invoked by the canvas to notify the widget it has been deleted
        from the workflow.

        If possible, subclasses should gracefully cancel any currently
        executing tasks.
        """
        pass

    def handleNewSignals(self):
        """
        Invoked by the workflow signal propagation manager after all
        signals handlers have been called.

        Reimplement this method in order to coalesce updates from
        multiple updated inputs.
        """
        pass

    # ############################################
    # PROGRESS BAR FUNCTIONS

    def progressBarInit(self, processEvents=QEventLoop.AllEvents):
        """
        Initialize the widget's progress (i.e show and set progress to 0%).

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        :param processEvents: Process events flag
        :type processEvents: `QEventLoop.ProcessEventsFlags` or `None`
        """
        self.startTime = time.time()
        self.setWindowTitle(self.captionTitle + " (0% complete)")

        if self.__progressState != 1:
            self.__progressState = 1
            self.processingStateChanged.emit(1)

        self.progressBarSet(0, processEvents)

    def progressBarSet(self, value, processEvents=QEventLoop.AllEvents):
        """
        Set the current progress bar to `value`.

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        :param float value: Progress value
        :param processEvents: Process events flag
        :type processEvents: `QEventLoop.ProcessEventsFlags` or `None`
        """
        old = self.__progressBarValue
        self.__progressBarValue = value

        if value > 0:
            if self.__progressState != 1:
                warnings.warn("progressBarSet() called without a "
                              "preceding progressBarInit()",
                              stacklevel=2)
                self.__progressState = 1
                self.processingStateChanged.emit(1)

            usedTime = max(1, time.time() - self.startTime)
            totalTime = (100.0 * usedTime) / float(value)
            remainingTime = max(0, totalTime - usedTime)
            h = int(remainingTime / 3600)
            min = int((remainingTime - h * 3600) / 60)
            sec = int(remainingTime - h * 3600 - min * 60)
            if h > 0:
                text = "%(h)d:%(min)02d:%(sec)02d" % vars()
            else:
                text = "%(min)d:%(sec)02d" % vars()
            self.setWindowTitle(self.captionTitle +
                                " (%(value).2f%% complete, remaining time: %(text)s)" % vars())
        else:
            self.setWindowTitle(self.captionTitle + " (0% complete)")

        if old != value:
            self.progressBarValueChanged.emit(value)

        if processEvents is not None and processEvents is not False:
            qApp.processEvents(processEvents)

    def progressBarValue(self):
        return self.__progressBarValue

    progressBarValue = pyqtProperty(float, fset=progressBarSet,
                                    fget=progressBarValue)

    processingState = pyqtProperty(int, fget=lambda self: self.__progressState)

    def progressBarAdvance(self, value, processEvents=QEventLoop.AllEvents):
        self.progressBarSet(self.progressBarValue + value, processEvents)

    def progressBarFinished(self, processEvents=QEventLoop.AllEvents):
        """
        Stop the widget's progress (i.e hide the progress bar).

        .. note::
            This method will by default call `QApplication.processEvents`
            with `processEvents`. To suppress this behavior pass
            ``processEvents=None``.

        :param processEvents: Process events flag
        :type processEvents: `QEventLoop.ProcessEventsFlags` or `None`
        """
        self.setWindowTitle(self.captionTitle)
        if self.__progressState != 0:
            self.__progressState = 0
            self.processingStateChanged.emit(0)

        if processEvents is not None and processEvents is not False:
            qApp.processEvents(processEvents)

    #: Widget's status message has changed.
    statusMessageChanged = Signal(str)

    def setStatusMessage(self, text):
        if self.__statusMessage != text:
            self.__statusMessage = text
            self.statusMessageChanged.emit(text)

    def statusMessage(self):
        return self.__statusMessage

    def keyPressEvent(self, e):
        if (int(e.modifiers()), e.key()) in OWWidget.defaultKeyActions:
            OWWidget.defaultKeyActions[int(e.modifiers()), e.key()](self)
        else:
            QDialog.keyPressEvent(self, e)

    def information(self, id=0, text=None):
        self.setState("Info", id, text)

    def warning(self, id=0, text=""):
        self.setState("Warning", id, text)

    def error(self, id=0, text=""):
        self.setState("Error", id, text)

    def setState(self, state_type, id, text):
        changed = 0
        if type(id) == list:
            for val in id:
                if val in self.widgetState[state_type]:
                    self.widgetState[state_type].pop(val)
                    changed = 1
        else:
            if type(id) == str:
                text = id
                id = 0
            if not text:
                if id in self.widgetState[state_type]:
                    self.widgetState[state_type].pop(id)
                    changed = 1
            else:
                self.widgetState[state_type][id] = text
                changed = 1

        if changed:
            if type(id) == list:
                for i in id:
                    self.widgetStateChanged.emit(state_type, i, "")
            else:
                self.widgetStateChanged.emit(state_type, id, text or "")

        tooltip_lines = []
        highest_type = None
        for a_type in ("Error", "Warning", "Info"):
            msgs_for_ids = self.widgetState.get(a_type)
            if not msgs_for_ids:
                continue
            msgs_for_ids = list(msgs_for_ids.values())
            if not msgs_for_ids:
                continue
            tooltip_lines += msgs_for_ids
            if highest_type is None:
                highest_type = a_type

        if highest_type is None:
            self.set_warning_bar(None)
        elif len(tooltip_lines) == 1:
            msg = tooltip_lines[0]
            if "\n" in msg:
                self.set_warning_bar(
                    highest_type, msg[:msg.index("\n")] + " (...)", msg)
            else:
                self.set_warning_bar(
                    highest_type, tooltip_lines[0], tooltip_lines[0])
        else:
            self.set_warning_bar(
                highest_type,
                "{} problems during execution".format(len(tooltip_lines)),
                "\n".join(tooltip_lines))

        return changed

    def set_warning_bar(self, state_type, text=None, tooltip=None):
        colors = {"Error": ("#ffc6c6", "black", QStyle.SP_MessageBoxCritical),
                  "Warning": ("#ffffc9", "black", QStyle.SP_MessageBoxWarning),
                  "Info": ("#ceceff", "black", QStyle.SP_MessageBoxInformation)}
        current_height = self.height()
        if state_type is None:
            if not self.warning_bar.isHidden():
                new_height = current_height - self.warning_bar.height()
                self.warning_bar.setVisible(False)
                self.resize(self.width(), new_height)
            return
        background, foreground, icon = colors[state_type]
        style = QApplication.instance().style()
        self.warning_icon.setPixmap(style.standardIcon(icon).pixmap(14, 14))

        self.warning_bar.setStyleSheet(
            "background-color: {}; color: {};"
            "padding: 3px; padding-left: 6px; vertical-align: center".
            format(background, foreground))
        self.warning_label.setText(text)
        self.warning_bar.setToolTip(tooltip)
        if self.warning_bar.isHidden():
            self.warning_bar.setVisible(True)
            new_height = current_height + self.warning_bar.height()
            self.resize(self.width(), new_height)

    def widgetStateToHtml(self, info=True, warning=True, error=True):
        iconpaths = {
            "Info": gui.resource_filename("icons/information.png"),
            "Warning": gui.resource_filename("icons/warning.png"),
            "Error": gui.resource_filename("icons/error.png")
        }
        items = []

        for show, what in [(info, "Info"), (warning, "Warning"),
                           (error, "Error")]:
            if show and self.widgetState[what]:
                items.append('<img src="%s" style="float: left;"> %s' %
                             (iconpaths[what],
                              "\n".join(self.widgetState[what].values())))
        return "<br>".join(items)

    @classmethod
    def getWidgetStateIcons(cls):
        if not hasattr(cls, "_cached__widget_state_icons"):
            info = QPixmap(gui.resource_filename("icons/information.png"))
            warning = QPixmap(gui.resource_filename("icons/warning.png"))
            error = QPixmap(gui.resource_filename("icons/error.png"))
            cls._cached__widget_state_icons = \
                {"Info": info, "Warning": warning, "Error": error}
        return cls._cached__widget_state_icons

    defaultKeyActions = {}

    if sys.platform == "darwin":
        defaultKeyActions = {
            (Qt.ControlModifier, Qt.Key_M):
                lambda self: self.showMaximized
                if self.isMinimized() else self.showMinimized(),
            (Qt.ControlModifier, Qt.Key_W):
                lambda self: self.setVisible(not self.isVisible())}

    def setBlocking(self, state=True):
        """
        Set blocking flag for this widget.

        While this flag is set this widget and all its descendants
        will not receive any new signals from the workflow signal manager.

        This is useful for instance if the widget does it's work in a
        separate thread or schedules processing from the event queue.
        In this case it can set the blocking flag in it's processNewSignals
        method schedule the task and return immediately. After the task
        has completed the widget can clear the flag and send the updated
        outputs.

        .. note::
            Failure to clear this flag will block dependent nodes forever.
        """
        if self.__blocking != state:
            self.__blocking = state
            self.blockingStateChanged.emit(state)

    def isBlocking(self):
        """
        Is this widget blocking signal processing.
        """
        return self.__blocking

    def resetSettings(self):
        self.settingsHandler.reset_settings(self)

    def workflowEnv(self):
        """
        Return (a view to) the workflow runtime environment.

        Returns
        -------
        env : types.MappingProxyType
        """
        return self.__env

    def workflowEnvChanged(self, key, value, oldvalue):
        """
        A workflow environment variable `key` has changed to value.

        Called by the canvas framework to notify widget of a change
        in the workflow runtime environment.

        The default implementation does nothing.
        """
        pass


# Pull signal constants from canvas to widget namespace
Default = widget_description.Default
NonDefault = widget_description.NonDefault
Single = widget_description.Single
Multiple = widget_description.Multiple
Explicit = widget_description.Explicit
Dynamic = widget_description.Dynamic

InputSignal = widget_description.InputSignal
OutputSignal = widget_description.OutputSignal


class AttributeList(list):
    pass


class ExampleList(list):
    pass

import os
import sys
import glob
import numpy as np
import urllib
import tarfile
import tensorflow as tf
from PyQt4.QtGui import QApplication, QSizePolicy, QStyle, QGridLayout, \
    QFileDialog
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable, \
    StringVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget

MODEL_DIR = "/tmp/imagenet/"
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


def maybe_download_and_extract():
    dest_directory = MODEL_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write("\r>> Downloading %s %.1f%%" % (
                filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    tarfile.open(filepath, ":gz").extractall(dest_directory)


def inference_on_images(images, layer):
    with tf.gfile.FastGFile(
                    MODEL_DIR + "classify_image_graph_def.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")

    with tf.Session() as sess:
        def get_profile(image):
            img_data = tf.gfile.FastGFile(image, "rb").read()
            tensor = sess.graph.get_tensor_by_name("{}:0".format(layer))
            return sess.run(tensor, {"DecodeJpeg/contents:0": img_data})

        return np.vstack((get_profile(image) for image in images))


class OWImages(OWWidget):
    name = "Images"
    description = "Computes profiles of images."
    icon = "icons/Images.svg"
    outputs = [("Data", Table)]

    want_main_area = False

    data_info_default = "No data on the output."
    layer = "pool_3"  # softmax
    dir_combo_items = Setting([])

    def __init__(self):
        super().__init__()
        self.domain = None
        self.data = None
        self.dir_name = ""

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)

        self.dir_combo = gui.comboBox(
            None, self, "", items=tuple(self.dir_combo_items))
        #self.dir_combo.setSizePolicy(QSizePolicy.Maximum)
        self.dir_combo.setMinimumWidth(240)
        self.dir_combo.activated.connect(self.reload)
        layout.addWidget(self.dir_combo, 0, 1)

        file_button = gui.button(
            None, self, '...', callback=self.dir_browse, autoDefault=False)
        file_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        file_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        layout.addWidget(file_button, 0, 2)

        reload_button = gui.button(
            None, self, "Reload", callback=self.reload, autoDefault=False)
        reload_button.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(reload_button, 0, 3)

        box = gui.vBox(self.controlArea, "Info")
        self.info = gui.widgetLabel(box, self.data_info_default)

        maybe_download_and_extract()
        self.load_data(self.dir_combo.currentText())

    def dir_browse(self):
        start_dir = os.path.expanduser("~/")
        dir_name = QFileDialog().getExistingDirectory(
            self, "Open a folder", start_dir, QFileDialog.ShowDirsOnly)
        if not dir_name:
            return
        index = self.dir_combo.findText(dir_name)
        if index < 0:
            index = 0
            self.dir_combo.insertItem(0, dir_name)
        self.dir_combo.setCurrentIndex(index)
        self.load_data(dir_name)

    def reload(self):
        self.load_data(self.dir_combo.currentText())

    def load_data(self, dir_name):
        if not dir_name:
            return

        self.data = self.run(self.layer, dir_name)
        if self.data is None:
            return

        if dir_name not in self.dir_combo_items:
            self.dir_combo_items.insert(0, dir_name)

        domain = self.data.domain
        text = "{} instance(s), {} feature(s), {} meta attribute(s)".format(
            len(self.data), len(domain.attributes), len(domain.metas))
        if domain.has_discrete_class:
            text += "\nClassification; discrete class with {} values.".format(
                len(domain.class_var.values))
        elif not len(domain.class_vars):
            text += "\nData has no target variable."
        self.info.setText(text)
        self.send("Data", self.data)

    def run(self, layer, photo_dir):
        images = []
        for dir_path, _, _ in os.walk(photo_dir):
            images.extend(glob.glob(os.path.join(dir_path, "*.jpg")))
        self.warning(0)
        if not len(images):
            self.warning("Chosen folder does not contain any photos.")
            return None
        scores = inference_on_images(images, layer)
        return self.create_table(images, scores)

    @staticmethod
    def create_table(images, scores):
        if scores.ndim > 2:
            m, _, _, n = scores.shape
            scores = scores.reshape((m, n))
        attrs = [ContinuousVariable("p" + str(i + 1))
                 for i in range(scores.shape[1])]
        values = list(set(os.path.split(os.path.split(image)[0])[1]
                          for image in images))
        domain = Domain(attrs, DiscreteVariable("label", values),
                        metas=[StringVariable("path")])
        labels = np.array([values.index(
            os.path.split(os.path.split(image)[0])[1]) for image in images])
        metas = np.array([image for image in images])[:, None]
        return Table(domain, np.round(scores, 3), labels, metas)


if __name__ == "__main__":
    a = QApplication([])
    ow = OWImages()
    ow.show()
    ow.raise_()
    a.exec_()
    ow.saveSettings()

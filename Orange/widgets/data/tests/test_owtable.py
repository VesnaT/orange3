from Orange.data import Table
from Orange.widgets.data.owtable import OWDataTable
from Orange.widgets.tests.base import WidgetTest


class TestOWDataTable(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDataTable)
        self.iris = Table("iris")

    def test_input_data(self):
        """Check number of tabs with data on the input"""
        self.send_signal("Data", self.iris, 1)
        self.assertEqual(self.widget.tabs.count(), 1)
        self.send_signal("Data", self.iris, 2)
        self.assertEqual(self.widget.tabs.count(), 2)

    def test_input_data_disconnect(self):
        """Check number of tabs after disconnecting data from the input"""
        self.send_signal("Data", self.iris, 1)
        self.assertEqual(self.widget.tabs.count(), 1)
        self.send_signal("Data", None, 1)
        self.assertEqual(self.widget.tabs.count(), 0)

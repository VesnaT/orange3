# pylint: disable=missing-docstring
from unittest.mock import Mock

import numpy as np

from AnyQt.QtCore import QDateTime, QDate, QTime
from AnyQt.QtWidgets import QWidget

from orangewidget.tests.base import GuiTest
from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable, \
    TimeVariable
from Orange.widgets.data.owcreateinstance import OWCreateInstance, \
    DiscreteVariableEditor, ContinuousVariableEditor, StringVariableEditor, \
    TimeVariableEditor
from Orange.widgets.tests.base import WidgetTest, datasets
from Orange.widgets.utils.state_summary import format_summary_details, \
    format_multiple_summaries


class TestOWCreateInstance(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCreateInstance)
        self.data = Table("iris")

    def test_output(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 1)
        self.assertEqual(output.domain, self.data.domain)
        array = np.round(np.median(self.data.X, axis=0), 1).reshape(1, 4)
        np.testing.assert_array_equal(output.X, array)

    def test_summary(self):
        info = self.widget.info
        reference = self.data[:1]
        no_input, no_output = "No data on input", "No data on output"

        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__input_summary.details, no_input)
        self.assertEqual(info._StateInfo__output_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)

        self.send_signal(self.widget.Inputs.data, self.data)
        data_list = [("Data", self.data), ("Reference", None)]
        summary, details = "150, 0", format_multiple_summaries(data_list)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)

        output = self.get_output(self.widget.Outputs.data)
        summary, details = f"{len(output)}", format_summary_details(output)
        self.assertEqual(info._StateInfo__output_summary.brief, summary)
        self.assertEqual(info._StateInfo__output_summary.details, details)

        self.send_signal(self.widget.Inputs.reference, reference)
        data_list = [("Data", self.data), ("Reference", reference)]
        summary, details = "150, 1", format_multiple_summaries(data_list)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)

        self.send_signal(self.widget.Inputs.data, None)
        data_list = [("Data", None), ("Reference", reference)]
        summary, details = "0, 1", format_multiple_summaries(data_list)
        self.assertEqual(info._StateInfo__input_summary.brief, summary)
        self.assertEqual(info._StateInfo__input_summary.details, details)
        self.assertEqual(info._StateInfo__output_summary.brief, "")
        self.assertEqual(info._StateInfo__output_summary.details, no_output)

        self.send_signal(self.widget.Inputs.reference, None)
        self.assertEqual(info._StateInfo__input_summary.brief, "")
        self.assertEqual(info._StateInfo__input_summary.details, no_input)

    def _get_init_buttons(self, widget=None):
        if not widget:
            widget = self.widget
        box = widget.controlArea.layout().itemAt(0).widget().children()[3]
        return box.children()[1:]

    def test_initialize_buttons(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.reference, self.data[:1])
        output = self.get_output(self.widget.Outputs.data)

        buttons = self._get_init_buttons()

        buttons[3].click()  # Input
        output_input = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output_input, self.data[:1])

        buttons[0].click()  # Median
        output_median = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output_median, output)

        buttons[1].click()  # Mean
        output_mean = self.get_output(self.widget.Outputs.data)
        output.X = np.round(np.mean(self.data.X, axis=0), 1).reshape(1, 4)
        self.assert_table_equal(output_mean, output)

        buttons[2].click()  # Random
        output_random = self.get_output(self.widget.Outputs.data)
        self.assertTrue((self.data.X.max(axis=0) >= output_random.X).all())
        self.assertTrue((self.data.X.min(axis=0) <= output_random.X).all())

        self.send_signal(self.widget.Inputs.data, self.data[9:10])
        buttons[2].click()  # Random
        output_random = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output_random, self.data[9:10])

    def test_initialize_buttons_commit_once(self):
        self.widget.commit = self.widget.unconditional_commit = Mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.reference, self.data[:1])
        self.widget.unconditional_commit.assert_called_once()

        self.widget.commit.reset_mock()
        buttons = self._get_init_buttons()
        buttons[3].click()  # Input
        self.widget.commit.assert_called_once()

    def test_table(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(self.widget.view.model().rowCount(), 5)
        self.assertEqual(self.widget.view.horizontalHeader().count(), 2)

        data = Table("zoo")
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(self.widget.view.model().rowCount(), 18)
        self.assertEqual(self.widget.view.horizontalHeader().count(), 2)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.view.model().rowCount(), 0)
        self.assertEqual(self.widget.view.horizontalHeader().count(), 2)

    def test_datasets(self):
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)

    def test_missing_values(self):
        domain = Domain([ContinuousVariable("c")],
                        class_vars=[DiscreteVariable("m", ("a", "b"))])
        data = Table(domain, np.array([[np.nan], [np.nan]]),
                     np.array([np.nan, np.nan]))
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output, data[:1])
        self.assertTrue(self.widget.Information.nans_removed.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.nans_removed.is_shown())

    def test_missing_values_reference(self):
        reference = self.data[:1].copy()
        reference[:] = np.nan
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.reference, reference)
        output1 = self.get_output(self.widget.Outputs.data)
        buttons = self._get_init_buttons()
        buttons[3].click()  # Input
        output2 = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output1, output2)

    def test_saved_workflow(self):
        data = self.data
        data.X[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        buttons = self._get_init_buttons()
        buttons[2].click()  # Random
        output1 = self.get_output(self.widget.Outputs.data)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWCreateInstance, stored_settings=settings)
        self.send_signal(widget.Inputs.data, data, widget=widget)
        output2 = self.get_output(widget.Outputs.data)
        self.assert_table_equal(output1, output2)

    def test_commit_once(self):
        self.widget.commit = self.widget.unconditional_commit = Mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.unconditional_commit.assert_called_once()

        self.widget.commit.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.commit.assert_called_once()

        self.widget.commit.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.commit.assert_called_once()

    def test_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()

    def test_sparse(self):
        data = self.data.to_sparse()
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.reference, data)


class TestDiscreteVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        self.editor = DiscreteVariableEditor(
            self.parent, ["Foo", "Bar"], self.callback
        )

    def test_init(self):
        self.assertEqual(self.editor.value, 0)
        self.assertEqual(self.editor._combo.currentText(), "Foo")
        self.callback.assert_not_called()

    def test_edit(self):
        """ Edit combo by user. """
        self.editor._combo.setCurrentText("Bar")
        self.assertEqual(self.editor.value, 1)
        self.assertEqual(self.editor._combo.currentText(), "Bar")
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set combo box value. """
        self.editor.value = 1
        self.assertEqual(self.editor.value, 1)
        self.assertEqual(self.editor._combo.currentText(), "Bar")
        self.callback.assert_called_once()


class TestContinuousVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        data = Table("iris")
        values = data.get_column_view(data.domain[0])[0]
        self.min_value = np.min(values)
        self.max_value = np.max(values)
        self.editor = ContinuousVariableEditor(
            self.parent, data.domain[0], self.min_value,
            self.max_value, self.callback
        )

    def test_init(self):
        self.assertEqual(self.editor.value, self.min_value)
        self.assertEqual(self.editor._slider.value(), self.min_value * 10)
        self.assertEqual(self.editor._spin.value(), self.min_value)
        self.callback.assert_not_called()

    def test_edit_slider(self):
        """ Edit slider by user. """
        self.editor._slider.setValue(int(self.max_value * 10))
        self.assertEqual(self.editor.value, self.max_value)
        self.assertEqual(self.editor._slider.value(), self.max_value * 10)
        self.assertEqual(self.editor._spin.value(), self.max_value)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        value = self.min_value + (self.max_value - self.min_value) / 2
        self.editor._slider.setValue(int(value * 10))
        self.assertEqual(self.editor.value, value)
        self.assertEqual(self.editor._slider.value(), value * 10)
        self.assertEqual(self.editor._spin.value(), value)
        self.callback.assert_called_once()

    def test_edit_spin(self):
        """ Edit spin by user. """
        self.editor._spin.setValue(self.max_value)
        self.assertEqual(self.editor.value, self.max_value)
        self.assertEqual(self.editor._slider.value(), self.max_value * 10)
        self.assertEqual(self.editor._spin.value(), self.max_value)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        self.editor._spin.setValue(self.max_value + 1)
        self.assertEqual(self.editor.value, self.max_value + 1)
        self.assertEqual(self.editor._slider.value(), self.max_value * 10)
        self.assertEqual(self.editor._spin.value(), self.max_value + 1)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        value = self.min_value + (self.max_value - self.min_value) / 2
        self.editor._spin.setValue(value)
        self.assertEqual(self.editor.value, value)
        self.assertEqual(self.editor._slider.value(), value * 10)
        self.assertEqual(self.editor._spin.value(), value)
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set slider/spin value. """
        self.editor.value = -2
        self.assertEqual(self.editor._slider.value(), self.min_value * 10)
        self.assertEqual(self.editor._spin.value(), -2)
        self.assertEqual(self.editor.value, -2)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        value = self.min_value + (self.max_value - self.min_value) / 4
        self.editor.value = value
        self.assertEqual(self.editor._slider.value(), value * 10)
        self.assertEqual(self.editor._spin.value(), value)
        self.assertEqual(self.editor.value, value)
        self.callback.assert_called_once()

    def test_missing_values(self):
        domain = Domain([ContinuousVariable("var")])
        data = Table(domain, np.array([[np.nan], [np.nan]]))
        self.assertRaises(ValueError, ContinuousVariableEditor, self.parent,
                          data.domain[0], np.nan, np.nan, Mock())


class TestStringVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        self.editor = StringVariableEditor(self.parent, self.callback)

    def test_init(self):
        self.assertEqual(self.editor.value, "")
        self.assertEqual(self.editor._edit.text(), "")
        self.callback.assert_not_called()

    def test_edit(self):
        """ Set lineedit by user. """
        self.editor._edit.setText("Foo")
        self.assertEqual(self.editor.value, "Foo")
        self.assertEqual(self.editor._edit.text(), "Foo")
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set lineedit value. """
        self.editor.value = "Foo"
        self.assertEqual(self.editor.value, "Foo")
        self.assertEqual(self.editor._edit.text(), "Foo")
        self.callback.assert_called_once()


class TestTimeVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        self.editor = TimeVariableEditor(
            self.parent, TimeVariable("var", have_date=1), self.callback
        )

    def test_init(self):
        self.assertEqual(self.editor.value, 0)
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(1970, 1, 1)))
        self.callback.assert_not_called()

    def test_edit(self):
        """ Edit datetimeedit by user. """
        datetime = QDateTime(QDate(2001, 9, 9))
        self.editor._edit.setDateTime(datetime)
        self.assertEqual(self.editor.value, 999993600)
        self.assertEqual(self.editor._edit.dateTime(), datetime)
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set datetimeedit value. """
        value = 999993600
        self.editor.value = value
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(2001, 9, 9)))
        self.assertEqual(self.editor.value, value)
        self.callback.assert_called_once()

    def test_have_date_have_time(self):
        callback = Mock()
        editor = TimeVariableEditor(
            self.parent, TimeVariable("var", have_date=1, have_time=1),
            callback
        )
        self.assertEqual(editor.value, 0)
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(1970, 1, 1), QTime(0, 0, 0)))
        self.callback.assert_not_called()

        datetime = QDateTime(QDate(2001, 9, 9), QTime(1, 2, 3))
        editor._edit.setDateTime(datetime)
        self.assertEqual(editor._edit.dateTime(), datetime)
        self.assertEqual(editor.value, 999993600 + 3723)
        callback.assert_called_once()

    def test_have_time(self):
        callback = Mock()
        editor = TimeVariableEditor(
            self.parent, TimeVariable("var", have_time=1), callback
        )
        self.assertEqual(editor.value, 0)
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(1970, 1, 1), QTime(0, 0, 0)))
        self.callback.assert_not_called()

        datetime = QDateTime(QDate(1900, 1, 1), QTime(1, 2, 3))
        editor._edit.setDateTime(datetime)
        self.assertEqual(editor._edit.dateTime(), datetime)
        self.assertEqual(editor.value, 3723)
        callback.assert_called_once()

    def test_no_date_no_time(self):
        callback = Mock()
        editor = TimeVariableEditor(self.parent, TimeVariable("var"), callback)
        self.assertEqual(editor.value, 0)
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(1970, 1, 1), QTime(0, 0, 0)))
        self.callback.assert_not_called()

        datetime = QDateTime(QDate(2001, 9, 9), QTime(1, 2, 3))
        editor._edit.setDateTime(datetime)
        self.assertEqual(editor._edit.dateTime(), datetime)
        self.assertEqual(editor.value, 999993600 + 3723)
        callback.assert_called_once()


if __name__ == "__main__":
    import unittest
    unittest.main()
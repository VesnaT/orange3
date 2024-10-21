# pylint: disable=missing-docstring,protected-access
import unittest

from Orange.classification import NaiveBayesLearner
from Orange.data import Table
from Orange.regression import PLSRegressionLearner
from Orange.widgets.evaluate.owparameterfitter import OWParameterFitter
from Orange.widgets.model.owrandomforest import OWRandomForest
from Orange.widgets.tests.base import WidgetTest


class TestOWParameterFitter(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._heart = Table("heart_disease")
        cls._housing = Table("housing")
        cls._naive_bayes = NaiveBayesLearner()
        cls._pls = PLSRegressionLearner()

    def setUp(self):
        self.widget = self.create_widget(OWParameterFitter)

    def test_input(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()

        self.send_signal(self.widget.Inputs.data, self._heart)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.incompatible_learner.is_shown())

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

    def test_input_no_params(self):
        self.send_signal(self.widget.Inputs.data, self._heart)
        self.send_signal(self.widget.Inputs.learner, self._naive_bayes)
        self.wait_until_finished()
        self.assertTrue(self.widget.Warning.no_parameters.is_shown())

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertFalse(self.widget.Warning.no_parameters.is_shown())

    def test_random_forest(self):
        rf_widget = self.create_widget(OWRandomForest)
        learner = self.get_output(rf_widget.Outputs.learner)

        self.send_signal(self.widget.Inputs.learner, learner)
        self.assertFalse(self.widget.Warning.no_parameters.is_shown())
        self.assertFalse(self.widget.Error.domain_transform_err.is_shown())
        self.assertFalse(self.widget.Error.unknown_err.is_shown())
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

        self.send_signal(self.widget.Inputs.data, self._heart)
        self.assertFalse(self.widget.Warning.no_parameters.is_shown())
        self.assertFalse(self.widget.Error.domain_transform_err.is_shown())
        self.assertFalse(self.widget.Error.unknown_err.is_shown())
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

        self.send_signal(self.widget.Inputs.data, self._housing)
        self.assertFalse(self.widget.Warning.no_parameters.is_shown())
        self.assertFalse(self.widget.Error.domain_transform_err.is_shown())
        self.assertFalse(self.widget.Error.unknown_err.is_shown())
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())
        self.assertFalse(self.widget.Error.incompatible_learner.is_shown())

    def test_plot(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()

        x = self.widget.graph._FitterPlot__bar_item_tr.opts["x"]
        self.assertEqual(list(x), [-0.2, 0.8])
        x = self.widget.graph._FitterPlot__bar_item_cv.opts["x"]
        self.assertEqual(list(x), [0.2, 1.2])

    def test_manual_steps(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()

        self.widget.controls.manual_steps.setText("1, 2, 3")
        self.widget.controls.type.buttons[1].click()
        self.wait_until_finished()

        x = self.widget.graph._FitterPlot__bar_item_tr.opts["x"]
        self.assertEqual(list(x), [-0.2, 0.8, 1.8])
        x = self.widget.graph._FitterPlot__bar_item_cv.opts["x"]
        self.assertEqual(list(x), [0.2, 1.2, 2.2])

    def test_steps_preview(self):
        self.send_signal(self.widget.Inputs.data, self._housing)
        self.send_signal(self.widget.Inputs.learner, self._pls)
        self.wait_until_finished()
        self.assertEqual(self.widget.preview, "[1, 2]")

        self.widget.controls.type.buttons[1].click()
        self.wait_until_finished()
        self.assertEqual(self.widget.preview, "[]")

        self.widget.controls.manual_steps.setText("10, 15, 20, 25")
        self.widget.controls.type.buttons[1].click()
        self.wait_until_finished()
        self.assertEqual(self.widget.preview, "[10, 15, 20, 25]")


if __name__ == "__main__":
    unittest.main()

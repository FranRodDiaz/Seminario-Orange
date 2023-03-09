import numpy

import Orange.data
from orangewidget.widget import OWBaseWidget, Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget.settings import Setting
from orangewidget import gui

class OWDataSamplerA(OWBaseWidget):
    name = "Data Sampler"
    description = "Randomly selects a subset of instances from the dataset"
    icon = "DataSamplerA.svg"
    priority = 10

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        sample = Output("Sampled Data", Orange.data.Table)
        other = Output("Other Data", Orange.data.Table)

    proportion = Setting(50)
    commitOnChange = Setting(0)

    want_main_area = True

    def __init__(self):
        super().__init__()

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No data on input yet, waiting to get something.')
        self.infob = gui.widgetLabel(box, '')
        self.infoc = gui.widgetLabel(box, '')

        gui.separator(self.controlArea)
        self.optionsBox = gui.widgetBox(self.controlArea, "Options")
        gui.spin(self.optionsBox, self, 'proportion',
                 minv=10, maxv=90, step=10, label='Sample Size [%]:',
                 callback=[self.selection, self.checkCommit])
        gui.checkBox(self.optionsBox, self, 'commitOnChange',
                     'Commit data on selection change')
        gui.button(self.optionsBox, self, "Commit", callback=self.commit)
        self.optionsBox.setDisabled(True)


    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.dataset = dataset
            self.infoa.setText('%d instances in input dataset' % len(dataset))
            self.optionsBox.setDisabled(False)
            self.clear_messages()
            self.info.set_input_summary("Original dataset: " + str(len(self.dataset)))      
            self.selection()
        else:
            self.dataset = None
            self.sample = None
            self.other = None
            self.optionsBox.setDisabled(False)
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.infob.setText('')
            self.warning("No data")
        self.commit()

    def selection(self):
        if self.dataset is None:
            return

        n_selected = int(numpy.ceil(len(self.dataset) * self.proportion / 100.))
        indices = numpy.random.permutation(len(self.dataset))
        indices_sample = indices[:n_selected]
        indices_other = indices[n_selected:]
        self.sample = self.dataset[indices_sample]
        self.other = self.dataset[indices_other]
        self.info.set_output_summary("Dataset sample:" + str(len(self.sample)) + "\nDataset other:" + str(len(self.other)))
        self.infob.setText('%d sampled instances' % len(self.sample))
        self.infoc.setText('%d other instances' % len(self.other))

    def commit(self):
        self.Outputs.sample.send(self.sample)
        self.Outputs.other.send(self.other)
        
    def checkCommit(self):
        if self.commitOnChange:
            self.commit()
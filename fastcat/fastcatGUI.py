#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""xpecgenGUI.py: A GUI for the xpecgen module"""

from __future__ import print_function

import tigre
# import fastcat as fc
import fastcat.fastcat as fc
import re
from glob import glob
import os
import numpy as np
from traceback import print_exc
import queue
import threading
from tkinter import *
from tkinter.ttk import *
import tkinter.filedialog
from tkinter import messagebox

from tigre.demos.Test_data import data_loader
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt


__author__ = "Dih5"

_elements = ['Nihil', 'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon, Graphite', 'Nitrogen', 'Oxygen',
             'Fluorine', 'Neon', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus', 'Sulfur', 'Chlorine',
             'Argon', 'Potassium', 'Calcium', 'Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron',
             'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine', 'Krypton',
             'Rubidium', 'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium',
             'Rhodium', 'Palladium', 'Silver', 'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium', 'Iodine', 'Xenon',
             'Cesium', 'Barium', 'Lanthanum', 'Cerium', 'Praseodymium', 'Neodymium', 'Promethium', 'Samarium',
             'Europium', 'Gadolinium', 'Terbium', 'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium', 'Lutetium',
             'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium', 'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury',
             'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine', 'Radon', 'Francium', 'Radium', 'Actinium',
             'Thorium', 'Protactinium', 'Uranium']


def _add_element_name(material):
    """Check if the string is a integer. If so, convert to element name, e.g., 13-> 13: Aluminum"""
    try:
        int(material)
        return "%d: %s" % (int(material), _elements[int(material)])
    except ValueError:
        return material


def _remove_element_name(material):
    """Revert _add_element_name"""
    if re.match("[0-9]+: .*", material):
        return material.split(":")[0]
    else:
        return material


class CreateToolTip(object):
    """
    A tooltip for a given widget.
    """

    # Based on the content from this post:
    # http://stackoverflow.com/questions/3221956/what-is-the-simplest-way-to-make-tooltips-in-tkinter

    def __init__(self, widget, text, color="#ffe14c"):
        """
        Create a tooltip for an existent widget.

        Args:
            widget: The widget the tooltip is applied to.
            text (str): The text of the tooltip.
            color: The color of the tooltip.
        """
        self.waittime = 500  # miliseconds
        self.wraplength = 180  # pixels
        self.widget = widget
        self.text = text
        self.color = color
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                      background=self.color, relief='solid', borderwidth=1,
                      wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


class ParBox:
    """A parameter entry with labels preceding and succeeding it and an optional tooltip"""

    def __init__(self, master=None, textvariable=0, lblText="", unitsTxt="", helpTxt="", row=0, read_only=False):
        """
        Create the parameter box.

        Args:
            master: the master widget.
            textvariable (:obj:`tkinter.Variable`): The variable associated with the parameter.
            lblText (str): The text preceding the text entry.
            unitsTxt (str): The text succeeding the text entry, typically the units.
            helpTxt (str): The help text to show in the tooltip. If "", no tooltip is shown.
            row (int): The row where the widgets are set in the grid.
            read_only (bool): Whether the entry is read_only.

        """
        self.lbl = Label(master, text=lblText)
        self.lbl.grid(row=row, column=0, sticky=W)

        self.txt = Entry(master, textvariable=textvariable)
        self.txt.grid(row=row, column=1, sticky=W + E)

        self.units = Label(master, text=unitsTxt, anchor=W)
        self.units.grid(row=row, column=2, sticky=W)

        if helpTxt != "":
            self.lblTT = CreateToolTip(self.lbl, helpTxt)
            self.txtTT = CreateToolTip(self.txt, helpTxt)
        if read_only:
            self.txt["state"] = "readonly"


def _human_order_key(text):
    """
    Key function to sort in human order.

    """
    # This is based in http://nedbatchelder.com/blog/200712/human_sorting.html
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', text)]


class XpecgenGUI(Notebook):
    """Tk-based GUI for the xpecgen package."""

    def __init__(self, master=None):
        """
        Create the GUI.

        Args:
            master: the tk master of the GUI.

        """
        Notebook.__init__(self, master)
        Grid.rowconfigure(master, 0, weight=1)
        Grid.columnconfigure(master, 0, weight=1)
        self.grid(row=0, column=0, sticky=N + S + E + W)
        self.master.title("".join(('fastcat v', fc.__version__, " GUI")))
        self.master.minsize(800, 600)

        self.spectra = []
        self.spectra_labels = []
        self.active_spec = 0  # The active spectrum from the list

        self.kernel = []
        # self.kernels = []
        self.phantom = []
        self.img = []
        self.tracker = []
        self.tracker2 = []

        # Interpolations used to calculate HVL
        self.fluence_to_dose = fc.get_fluence_to_dose()
        self.mu_Al = fc.get_mu(13)
        self.mu_Cu = fc.get_mu(29)

        self.initVariables()
        self.createWidgets()

        self.history_poll()

    def history_poll(self):
        """
        Polling method to update changes in spectrum list.

        """
        # Note the Tk manual advised for polling instead of binding all methods that are able to change a listbox
        try:
            now = self.lstHistory.curselection()[0]
            if now != self.active_spec:
                self.active_spec = now
                self.update_plot()

        except IndexError:
            pass
        self.after(150, self.history_poll)

    def initVariables(self):
        """Create and initialize interface variables"""

        # Calculation-related variables
        self.E0 = DoubleVar()
        self.E0.set(100.0)

        self.Theta = DoubleVar()
        self.Theta.set(12.0)

        self.Phi = DoubleVar()
        self.Phi.set(0.0)

        self.Z = StringVar()
        self.Z.set(_add_element_name("74"))

        self.load = StringVar()
        self.load.set("W_spectrum_6")

        self.det = StringVar()
        self.det.set("CuGOS-784-micrometer")

        self.det2 = StringVar()
        self.det2.set("CuGOS-784-micrometer")

        self.geo = StringVar()
        # self.geo.set("Head Phantom")
        self.geo.set("Catphan_404")#"catphan_low_contrast_512_8cm")

        self.EMin = DoubleVar()
        self.EMin.set(3.0)

        self.NumE = IntVar()
        self.NumE.set(50)

        self.Eps = DoubleVar()
        self.Eps.set(0.5)

        self.algorithm = StringVar()
        self.algorithm.set('Filter Back Projection')

        self.filt = StringVar()
        self.filt.set('hamming')

        self.filetype = StringVar()
        self.filetype.set('*.npy')

        self.niter = IntVar()
        self.niter.set(0)

        self.fs_size = DoubleVar()
        self.fs_size.set(0.)

        # Operation-related variables

        self.AttenMaterial = StringVar()
        self.AttenMaterial.set(_add_element_name("13"))

        self.AttenThick = DoubleVar()
        self.AttenThick.set(0.1)

        self.current = DoubleVar()
        self.current.set(0)#6.200432949299284e-07)

        self.noise = None
        self.nphoton = None

        self.NormCriterion = StringVar()
        self.NormCriterion.set("Number")

        self.NormValue = DoubleVar()
        self.NormValue.set(1.0)

        # Output HVLs
        self.HVL1 = StringVar()
        self.HVL1.set("0")
        self.HVL2 = StringVar()
        self.HVL2.set("0")
        self.HVL3 = StringVar()
        self.HVL3.set("0")
        self.HVL4 = StringVar()
        self.HVL4.set("0")
        self.doseperproj = DoubleVar()
        self.doseperproj.set(0)
        self.fluenceperproj = StringVar()
        self.fluenceperproj.set("0")

        self.AttenMaterial2 = StringVar()
        self.AttenMaterial2.set(_add_element_name("13"))

        self.AttenThick2 = DoubleVar()
        self.AttenThick2.set(0.1)

        self.NormCriterion2 = StringVar()
        self.NormCriterion2.set("None")

        self.filename = StringVar()
        self.filename.set("filename")

        self.label = StringVar()
        self.label.set("")

        self.NormValue2 = DoubleVar()
        self.NormValue2.set(1.0)

        # Output HVLs
        self.HVL12 = StringVar()
        self.HVL12.set("0")
        self.HVL22 = StringVar()
        self.HVL22.set(0.0)
        self.HVL32 = DoubleVar()
        self.HVL32.set(30)
        self.HVL42 = DoubleVar()
        self.HVL42.set(0.0)
        self.HVL52 = DoubleVar()
        self.HVL52.set(360)

        # Output Norms
        self.number = StringVar()
        self.number.set("0")
        self.energy = StringVar()
        self.energy.set("0")
        self.dose = StringVar()
        self.dose.set("0")

        # kV energy spectrum
        self.kV = BooleanVar()
        self.kV.set(True)

        self.scatter_on = IntVar()
        self.scatter_on.set(1)

        self.det_on = IntVar()
        self.det_on.set(1)

    def createWidgets(self):
        """Create the widgets in the GUI"""

        self.frmCalc = Frame(self)
        self.frmSpec = Frame(self)
        self.frmKern = Frame(self)
        self.frmGeom = Frame(self)
        self.frmSino = Frame(self)
        self.frmReso = Frame(self)
        self.frmOutp = Frame(self)
        self.add(self.frmCalc, text='Calculate')
        self.add(self.frmSpec, text='Spectrum')
        self.add(self.frmKern, text='Kernel')
        self.add(self.frmGeom, text='Geometry')
        self.add(self.frmSino, text='Projections')
        self.add(self.frmReso, text='Results')
        self.add(self.frmOutp, text='Analysis')

        self.matplotlib_embedded = True

        self.init_calc()
        self.init_spec()
        self.init_geom()
        self.init_sino()
        self.init_reso()
        self.init_kern()
        self.init_outp()

        # Calculate Tab
    def init_calc(self):
        """Initialize the first tab"""

        # Physical Parameters
        self.frmPhysPar = LabelFrame(
            self.frmCalc, text="Physical parameters")
        self.frmPhysPar.grid(row=0, column=0, sticky=N + S + E + W)
        self.ParE0 = ParBox(self.frmPhysPar, self.E0, lblText="Electron Energy (E0)",
                            unitsTxt="keV", helpTxt="Electron kinetic energy in keV.", row=0)
        self.ParTheta = ParBox(self.frmPhysPar, self.Theta, lblText=u"Angle (\u03b8)",
                                unitsTxt="º", helpTxt="X-rays emission angle. The anode's normal is at 90º.", row=1)
        self.ParPhi = ParBox(self.frmPhysPar, self.Phi, lblText=u"Elevation angle (\u03c6)",
                                unitsTxt="º", helpTxt="X-rays emission altitude. The anode's normal is at 0º.", row=2)
        self.lblZ = Label(self.frmPhysPar, text="Target atomic number")
        self.lblZTT = CreateToolTip(self.lblZ,
                                    "Atomic number of the target. IMPORTANT: Only used in the cross-section and distance scaling. Fluence uses a tugsten model, but the range is increased in lower Z materials. Besides, characteristic radiation is only calculated for tugsten.")
        self.lblZ.grid(row=3, column=0, sticky=W)
        self.cmbZ = Combobox(self.frmPhysPar, textvariable=self.Z)
        self.cmbZ.grid(row=3, column=1, sticky=W + E)
        self.cmbZTT = CreateToolTip(self.cmbZ,
                                    "Atomic number of the target. IMPORTANT: Only used in the cross-section and distance scaling. Fluence uses a tugsten model, but the range is increased in lower Z materials. Besides, characteristic radiation is only calculated for tugsten.")
        # Available cross-section data
        target_list = list(map(lambda x: (os.path.split(x)[1]).split(
            ".csv")[0], glob(os.path.join(fc.data_path, "cs", "*.csv"))))
        target_list.remove("grid")
        # Available csda-data
        csda_list = list(map(lambda x: (os.path.split(x)[1]).split(
            ".csv")[0], glob(os.path.join(fc.data_path, "csda", "*.csv"))))
        # Available attenuation data
        mu_list = list(map(lambda x: (os.path.split(x)[1]).split(
            ".csv")[0], glob(os.path.join(fc.data_path, "mu", "*.csv"))))
        mu_list.sort(key=_human_order_key)  # Used later

        available_list = list(
            set(target_list) & set(csda_list) & set(mu_list))
        available_list.sort(key=_human_order_key)

        self.cmbZ["values"] = list(map(_add_element_name, available_list))

        self.loadlbl = Label(self.frmPhysPar, text="Load MV spectra")
        self.loadTT = CreateToolTip(self.lblZ,
                                    "Load known MV spectra from file")
        self.loadlbl.grid(row=5, column=0, sticky=W)
        self.cmbload = Combobox(self.frmPhysPar, textvariable=self.load)
        self.cmbload.grid(row=5, column=1, sticky=W + E)
        self.cmbloadTT = CreateToolTip(self.cmbZ,
                                        "Load known MV spectra from file")

        # Available cross-section data
        target_list_load = list(map(lambda x: (os.path.split(x)[1]).split(
            ".txt")[0], glob(os.path.join(fc.data_path, "MV_spectra", "*.txt"))))
        # list(map(_add_element_name, available_list))
        print(target_list_load)
        self.cmbload["values"] = target_list_load

        Grid.columnconfigure(self.frmPhysPar, 0, weight=0)
        Grid.columnconfigure(self.frmPhysPar, 1, weight=1)
        Grid.columnconfigure(self.frmPhysPar, 2, weight=1)
        Grid.columnconfigure(self.frmPhysPar, 3, weight=0)

        # -Numerical Parameters
        self.frmNumPar = LabelFrame(
            self.frmCalc, text="Numerical parameters")
        self.frmNumPar.grid(row=0, column=1, sticky=N + S + E + W)
        self.ParEMin = ParBox(self.frmNumPar, self.EMin,
                                lblText="Min energy", unitsTxt="keV",
                                helpTxt="Minimum kinetic energy in the bremsstrahlung calculation. Note this might influence the characteristic peaks prediction.",
                                row=0)
        self.ParNumE = ParBox(self.frmNumPar, self.NumE,
                                lblText="Number of points", unitsTxt="",
                                helpTxt="Amount of points for the mesh were the bremsstrahlung spectrum is calculated.\nBremsstrahlung component is extended by interpolation.",
                                row=1)
        self.ParEps = ParBox(self.frmNumPar, self.Eps, lblText="Integrating tolerance", unitsTxt="",
                                helpTxt="A numerical tolerance parameter used in numerical integration. Values around 0.5 provide fast and accurate calculations. If you want insanely accurate (and physically irrelevant) numerical integration you can reduce this value, increasing computation time.",
                                row=2)

        Grid.columnconfigure(self.frmNumPar, 0, weight=0)
        Grid.columnconfigure(self.frmNumPar, 1, weight=1)
        Grid.columnconfigure(self.frmNumPar, 2, weight=0)

        # -Buttons, status bar...
        self.cmdCalculate = Button(self.frmCalc, text="Calculate")
        self.cmdCalculate["command"] = self.calculate
        self.cmdCalculate.bind('<Return>', lambda event: self.calculate())
        self.cmdCalculate.bind(
            '<KP_Enter>', lambda event: self.calculate())  # Enter (num. kb)
        self.cmdCalculate.grid(row=1, column=0, sticky=E + W)

        self.cmdload = Button(self.frmCalc, text="Load")
        self.cmdload["command"] = self.loadfile
        self.cmdload.bind('<Return>', lambda event: self.loadfile())
        self.cmdload.bind(
            '<KP_Enter>', lambda event: self.loadfile())  # Enter (num. kb)
        self.cmdload.grid(row=2, column=0, sticky=E + W)

        self.barProgress = Progressbar(
            self.frmCalc, orient="horizontal", length=100, mode="determinate")
        self.barProgress.grid(row=1, column=1, columnspan=1, sticky=E + W)

        self.barProgressload = Progressbar(
            self.frmCalc, orient="horizontal", length=100, mode="determinate")
        self.barProgressload.grid(
            row=2, column=1, columnspan=1, sticky=E + W)

        Grid.columnconfigure(self.frmCalc, 0, weight=1)
        Grid.columnconfigure(self.frmCalc, 1, weight=1)

        Grid.rowconfigure(self.frmCalc, 0, weight=1)
        Grid.rowconfigure(self.frmCalc, 1, weight=0)

    def init_spec(self):

        """Initialize history tab"""

        # -History frame
        self.frmHist = LabelFrame(self.frmSpec, text="History")
        self.frmHist.grid(row=0, column=0, sticky=N + S + E + W)
        self.lstHistory = Listbox(self.frmHist, selectmode=BROWSE)
        self.lstHistory.grid(row=0, column=0, sticky=N + S + E + W)
        self.scrollHistory = Scrollbar(self.frmHist, orient=VERTICAL)
        self.scrollHistory.grid(row=0, column=1, sticky=N + S)
        self.lstHistory.config(yscrollcommand=self.scrollHistory.set)
        self.scrollHistory.config(command=self.lstHistory.yview)
        self.cmdCleanHistory = Button(
            self.frmHist, text="Revert to selected", state=DISABLED)
        self.cmdCleanHistory["command"] = self.clean_history
        self.cmdCleanHistory.grid(row=1, column=0, columnspan=2, sticky=E + W)
        self.cmdExport = Button(
            self.frmHist, text="Save selected", state=DISABLED)
        self.cmdExport["command"] = self.export
        self.cmdExport.grid(row=2, column=0, columnspan=2, sticky=E + W)

        self.lbldet = Label(self.frmSpec, text="Detector Response")
        self.lbldetTT = CreateToolTip(self.lblZ,
                                    "Choose a detector response to modify the detector")
        self.lbldet.grid(row=3, column=0, sticky=W)
        self.cmbdet = Combobox(self.frmSpec, textvariable=self.det)
        self.cmbdet.grid(row=3, column=1, sticky=W + E)
        self.cmbdetTT = CreateToolTip(self.cmbdet,
                                    "Choose a detectro response to modify")

        # Available cross-section data
        target_list_det = list(map(lambda x: (os.path.split(
            x)[-1]), glob(os.path.join(fc.data_path, "Detectors", '*'))))

        # list(map(_add_element_name, available_list))
        self.cmbdet["values"] = target_list_det

        self.cmddet = Button(self.frmSpec, text="Calculate Detector Response")
        self.cmddet["command"] = self.computeKernel
        self.cmddet.bind('<Return>', lambda event: self.computeKernel())
        self.cmddet.bind(
            '<KP_Enter>', lambda event: self.computeKernel())  # Enter (num. kb)
        self.cmddet.grid(row=5, column=0, sticky=E + W)

        Grid.rowconfigure(self.frmHist, 0, weight=1)
        Grid.columnconfigure(self.frmHist, 0, weight=1)
        Grid.columnconfigure(self.frmHist, 1, weight=0)

        # Available cross-section data
        target_list = list(map(lambda x: (os.path.split(x)[1]).split(
            ".csv")[0], glob(os.path.join(fc.data_path, "cs", "*.csv"))))
        target_list.remove("grid")
        # Available csda-data
        csda_list = list(map(lambda x: (os.path.split(x)[1]).split(
            ".csv")[0], glob(os.path.join(fc.data_path, "csda", "*.csv"))))
        # Available attenuation data
        mu_list = list(map(lambda x: (os.path.split(x)[1]).split(
            ".csv")[0], glob(os.path.join(fc.data_path, "mu", "*.csv"))))
        mu_list.sort(key=_human_order_key)  # Used later

        available_list = list(
            set(target_list) & set(csda_list) & set(mu_list))
        available_list.sort(key=_human_order_key)        

        # -Operations frame
        self.frmOper = LabelFrame(self.frmSpec, text="Spectrum operations")
        self.frmOper.grid(row=1, column=0, sticky=N + S + E + W)
        # --Attenuation
        self.frmOperAtten = LabelFrame(self.frmOper, text="Attenuate")
        self.frmOperAtten.grid(row=0, column=0, sticky=N + S + E + W)
        self.lblAttenMaterial = Label(self.frmOperAtten, text="Material")
        self.lblAttenMaterial.grid()
        self.cmbAttenMaterial = Combobox(
            self.frmOperAtten, textvariable=self.AttenMaterial)
        self.cmbAttenMaterial["values"] = list(map(_add_element_name, mu_list))
        self.cmbAttenMaterial.grid(row=0, column=1, sticky=E + W)
        self.ParAttenThick = ParBox(
            self.frmOperAtten, self.AttenThick, lblText="Thickness", unitsTxt="cm", row=1)
        self.cmdAtten = Button(
            self.frmOperAtten, text="Add attenuation", state=DISABLED)
        self.cmdAtten["command"] = self.attenuate
        self.cmdAtten.grid(row=2, column=0, columnspan=3, sticky=E + W)
        Grid.columnconfigure(self.frmOperAtten, 0, weight=0)
        Grid.columnconfigure(self.frmOperAtten, 1, weight=1)
        Grid.columnconfigure(self.frmOperAtten, 2, weight=0)

        # --Normalize
        self.frmOperNorm = LabelFrame(self.frmOper, text="Normalize")
        self.frmOperNorm.grid(row=1, column=0, sticky=N + S + E + W)
        self.lblNormCriterion = Label(self.frmOperNorm, text="Criterion")
        self.lblNormCriterion.grid()
        self.cmbNormCriterion = Combobox(
            self.frmOperNorm, textvariable=self.NormCriterion)
        self.criteriaList = ["Number", "Energy (keV)", "Dose (mGy)"]
        self.cmbNormCriterion["values"] = self.criteriaList
        self.cmbNormCriterion.grid(row=0, column=1, sticky=E + W)
        self.ParNormValue = ParBox(
            self.frmOperNorm, self.NormValue, lblText="Value", unitsTxt="", row=1)
        self.cmdNorm = Button(
            self.frmOperNorm, text="Normalize", state=DISABLED)
        self.cmdNorm["command"] = self.normalize
        self.cmdNorm.grid(row=2, column=0, columnspan=3, sticky=E + W)
        Grid.columnconfigure(self.frmOperNorm, 0, weight=0)
        Grid.columnconfigure(self.frmOperNorm, 1, weight=1)
        Grid.columnconfigure(self.frmOperNorm, 2, weight=0)

        Grid.columnconfigure(self.frmOper, 0, weight=1)
        Grid.rowconfigure(self.frmOper, 0, weight=1)

        self.frmPlot = Frame(self.frmSpec)

        try:
            self.fig = Figure(figsize=(5, 4), dpi=100,
                            facecolor=self.master["bg"])
            self.subfig = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frmPlot)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

            self.canvasToolbar = NavigationToolbar2Tk(
                self.canvas, self.frmPlot)
            self.canvasToolbar.update()
            self.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

            self.frmPlot.grid(row=0, column=1, rowspan=3, sticky=N + S + E + W)

            self.matplotlib_embedded = True
        except Exception:
            self.matplotlib_embedded = False
            # self.cmdShowPlot = Button(self.frmPlot,text="Open plot window")
            # self.cmdShowPlot["command"] = self.update_plot
            # self.cmdShowPlot.grid(row=0,column=0)
            print("WARNING: Matplotlib couldn't be embedded in TkAgg.\nUsing independent window instead",
                file=sys.stderr)

        # --Spectral parameters frame
        self.frmSpectralParameters = LabelFrame(
            self.frmSpec, text="Spectral parameters")
        self.frmSpectralParameters.grid(row=2, column=0, sticky=S + E + W)
        self.ParHVL1 = ParBox(self.frmSpectralParameters, self.HVL1, lblText="1HVL Al", unitsTxt="cm", row=0,
                            read_only=True,
                            helpTxt="Thickness of Al at which the dose produced by the spectrum is halved, according to the exponential attenuation model.")
        self.ParHVL2 = ParBox(self.frmSpectralParameters, self.HVL2, lblText="2HVL Al", unitsTxt="cm", row=1,
                            read_only=True,
                            helpTxt="Thickness of Al at which the dose produced by the spectrum after crossing a HVL is halved again, according to the exponential attenuation model.")
        self.ParHVL3 = ParBox(self.frmSpectralParameters, self.HVL3, lblText="1HVL Cu", unitsTxt="cm", row=2,
                            read_only=True,
                            helpTxt="Thickness of Cu at which the dose produced by the spectrum is halved, according to the exponential attenuation model.")
        self.ParHVL4 = ParBox(self.frmSpectralParameters, self.HVL4, lblText="2HVL Cu", unitsTxt="cm", row=3,
                            read_only=True,
                            helpTxt="Thickness of Cu at which the dose produced by the spectrum after crossing a HVL is halved again, according to the exponential attenuation model.")
        self.ParNorm = ParBox(self.frmSpectralParameters, self.number, lblText="Photon number", unitsTxt="", row=4,
                            read_only=True, helpTxt="Number of photons in the spectrum.")
        self.ParEnergy = ParBox(self.frmSpectralParameters, self.energy, lblText="Energy", unitsTxt="keV", row=5,
                                read_only=True, helpTxt="Total energy in the spectrum.")
        self.ParDose = ParBox(self.frmSpectralParameters, self.dose, lblText="Dose", unitsTxt="mGy", row=6,
                            read_only=True,
                            helpTxt="Dose produced in air by the spectrum, assuming it is describing the differential fluence in particles/keV/cm^2.")

        Grid.columnconfigure(self.frmSpec, 0, weight=1)

        if self.matplotlib_embedded:
            # If not embedding, use the whole window
            Grid.columnconfigure(self.frmSpec, 1, weight=3)

        Grid.rowconfigure(self.frmSpec, 0, weight=1)
        Grid.rowconfigure(self.frmSpec, 1, weight=1)

    def init_kern(self):

        # -Plot frame2
        self.frmPlotKern = Frame(self.frmKern)

        try:
            self.figKern = Figure(figsize=(8, 4), dpi=100,
                                facecolor=self.master["bg"])
            self.subfigKern = self.figKern.add_subplot(121)
            self.subfigKern2 = self.figKern.add_subplot(122)
            self.canvasKern = FigureCanvasTkAgg(
                self.figKern, master=self.frmPlotKern)
            self.canvasKern.draw()
            self.canvasKern.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

            self.canvasToolbarKern = NavigationToolbar2Tk(
                self.canvasKern, self.frmPlotKern)
            self.canvasToolbarKern.update()
            self.canvasKern._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

            self.frmPlotKern.grid(
                row=0, column=1, rowspan=3, sticky=N + S + E + W)

            self.matplotlib_embedded = True
        except Exception:
            self.matplotlib_embedded = False
            print("WARNING: Matplotlib couldn't be embedded in TkAgg.\nUsing independent window instead",
                file=sys.stderr)

        geo_list = list(map(lambda x: (os.path.split(x)[1]).split(
            ".npy")[0], glob(os.path.join(fc.data_path, "phantoms", "*.npy"))))
        geo_list.sort(key=_human_order_key)  # Used later

        self.frmGeo = LabelFrame(self.frmKern, text="Geometries")
        self.frmGeo.grid(row=0, column=0, sticky=N + S + E + W)
        self.lblgeo = Label(self.frmGeo, text="Choose Geometry")
        self.lblgeoTT = CreateToolTip(self.lblgeo,
                                    "Choose a geometry")
        self.lblgeo.grid(row=0, column=0, sticky=W)
        self.cmbgeo = Combobox(self.frmGeo, textvariable=self.geo)
        self.cmbgeo["values"] = ['Catphan_515','Catphan_MTF','XCAT','Catphan_404','Catphan_projections']#geo_list # these names correspond to xpecgen classes
        self.cmbgeo.grid(row=0, column=1, sticky=W + E)
        self.cmbgeoTT = CreateToolTip(self.cmbgeo,
                                    "Choose a geometry")

        self.cmdgeo = Button(self.frmKern, text="Geometry Viewer")
        self.cmdgeo["command"] = self.computeGeometry
        self.cmdgeo.bind('<Return>', lambda event: self.computeGeometry())
        self.cmdgeo.bind(
            '<KP_Enter>', lambda event: self.computeGeometry())  # Enter (num. kb)
        self.cmdgeo.grid(row=3, column=0, sticky=S + E + W)

        self.cmbgeofs = Combobox(self.frmGeo, textvariable=self.geo)
        self.cmbgeofs = ParBox(
            self.frmGeo, self.fs_size, lblText="Focal spot size", unitsTxt="mm", row=2)
        # self.cmbgeofs["values"] = geo_list
        # self.cmbgeofs.grid(row=2, column=0, sticky=W + E)        

        self.cmdgeofs = Button(self.frmGeo, text="Add focal spot")
        self.cmdgeofs["command"] = self.add_focal_spot
        self.cmdgeofs.bind('<Return>', lambda event: self.add_focal_spot())
        self.cmdgeofs.bind(
            '<KP_Enter>', lambda event: self.add_focal_spot())  # Enter (num. kb)
        self.cmdgeofs.grid(row=5, column=0, columnspan=2, sticky=N + S + E + W)

        Grid.columnconfigure(self.frmKern, 0, weight=1)
        Grid.columnconfigure(self.frmKern, 1, weight=1)
        Grid.rowconfigure(self.frmKern, 0, weight=1)
        Grid.rowconfigure(self.frmKern, 1, weight=1)

        # --Spectral parameters frame
        self.frmSpectralParameters2 = LabelFrame(
            self.frmKern, text="Spectral parameters")
        self.frmSpectralParameters2.grid(row=1, column=0, sticky=N+E + W)
        self.ParHVL12 = ParBox(self.frmSpectralParameters2, self.load, lblText="Loaded Spectra", row=0,
                            read_only=True,
                            helpTxt="Thickness of Al at which the dose produced by the spectrum is halved, according to the exponential attenuation model.")
        # self.ParHVL22 = ParBox(self.frmSpectralParameters2, self.geo, lblText="Geo file", row=1,
        #                     read_only=True,
        #                     helpTxt="Thickness of Al at which the dose produced by the spectrum after crossing a HVL is halved again, according to the exponential attenuation model.")

        Grid.columnconfigure(self.frmKern, 0, weight=1)

        if self.matplotlib_embedded:
            # If not embedding, use the whole window
            Grid.columnconfigure(self.frmKern, 1, weight=3)

        Grid.rowconfigure(self.frmKern, 0, weight=1)
        Grid.rowconfigure(self.frmKern, 1, weight=1)

    def init_geom(self):

        """Initialize geom frame"""

        # -History frame
        self.frmHist2 = LabelFrame(self.frmGeom, text="Phantom Mapping")
        self.frmHist2.grid(row=0, column=0, sticky=N + S + E + W)

        self.lstHistory2 = Listbox(self.frmHist2, selectmode=BROWSE)
        self.lstHistory2.grid(row=0, column=0, sticky=N + S + E + W)

        self.scrollHistory2 = Scrollbar(self.frmHist2, orient=VERTICAL)
        self.scrollHistory2.grid(row=0, column=1, sticky=N + S)

        self.lstHistory2.config(yscrollcommand=self.scrollHistory2.set)
        self.scrollHistory2.config(command=self.lstHistory2.yview)

        mu_list = list(map(lambda x: (os.path.split(x)[1]).split(
            ".csv")[0], glob(os.path.join(fc.data_path, "mu", "*.csv"))))
        mu_list.sort(key=_human_order_key)  # Used later

        self.cmbChangeMaterial = Combobox(
            self.frmHist2, textvariable=self.AttenMaterial)
        self.lblChange = Label(self.frmHist2, text="Material")
        self.lblChangeTT = CreateToolTip(self.lblChange,
                                    "Change a selected material in the phantom mapping to the material above")
        self.lblChange.grid(row=1, column=0, sticky=W)
        self.cmbChangeMaterial["values"] = list(map(_add_element_name, mu_list))
        self.cmbChangeMaterial.grid(row=2, column=0, sticky=E + W)
        self.cmdCleanHistory2 = Button(
            self.frmHist2, text="Change Selected")
        self.cmdCleanHistory2["command"] = self.update_phan_map
        self.cmdCleanHistory2.grid(row=3, column=0, columnspan=2, sticky=E + W)
        # self.cmdExport2 = Button(
        #     self.frmHist2, text="Save selected", state=DISABLED)
        # self.cmdExport2["command"] = self.export
        # self.cmdExport2.grid(row=2, column=0, columnspan=2, sticky=E + W)

        self.cmddet2 = Button(self.frmGeom, text="Generate Projections")
        self.cmddet2["command"] = self.computeProjection
        self.cmddet2.bind('<Return>', lambda event: self.computeProjection())
        self.cmddet2.bind(
            '<KP_Enter>', lambda event: self.computeProjection())  # Enter (num. kb)
        self.cmddet2.grid(row=5, column=0, sticky=E + W)

        Grid.rowconfigure(self.frmHist2, 0, weight=1)
        Grid.columnconfigure(self.frmHist2, 0, weight=1)
        Grid.columnconfigure(self.frmHist2, 1, weight=0)

        # -Operations frame
        self.frmOper2 = LabelFrame(self.frmGeom, text="Operations")
        self.frmOper2.grid(row=1, column=0, sticky=N + S + E + W)
        # --Attenuation
        self.frmOperAtten2 = LabelFrame(self.frmOper2, text="Simulation Parameters")
        self.frmOperAtten2.grid(row=0, column=0, sticky=N + S + E + W)

        self.radio1 = Checkbutton(self.frmOperAtten2,text='Scatter Correction',variable=self.scatter_on).pack(anchor=W)
        self.radio2 = Checkbutton(self.frmOperAtten2,text='Detector Convolution',variable=self.det_on).pack(anchor=W)

        Grid.columnconfigure(self.frmOperAtten2, 0, weight=0)
        Grid.columnconfigure(self.frmOperAtten2, 1, weight=1)
        Grid.columnconfigure(self.frmOperAtten2, 2, weight=0)
        # self.ParAttenThick = ParBox(
        #     self.frmOperAtten2, self.current, lblText="Exposure", unitsTxt="mA s", row=1)

        # --Normalize
        self.frmOperNorm2 = LabelFrame(self.frmOper2, text="Configure Noise")
        self.frmOperNorm2.grid(row=1, column=0, sticky=N + S + E + W)
        self.lblNormCriterion2 = Label(self.frmOperNorm2, text="Criterion")
        self.lblNormCriterion2.grid()
        self.cmbNormCriterion2 = Combobox(
            self.frmOperNorm2, textvariable=self.NormCriterion2)
        self.criteriaList2 = ["None", "Dose per View (mGy)","Dose per CT (mGy)", "Fluence (n photons per view)"]
        self.cmbNormCriterion2["values"] = self.criteriaList2
        self.cmbNormCriterion2.grid(row=0, column=1, sticky=E + W)
        self.ParNormValue2 = ParBox(
            self.frmOperNorm2, self.current, lblText="Value", unitsTxt="", row=1)
        self.cmdNorm = Button(
            self.frmOperNorm2, text="Update", state=DISABLED)
        self.cmdNorm["command"] = self.normalize2
        self.cmdNorm.grid(row=2, column=0, columnspan=3, sticky=E + W)

        Grid.columnconfigure(self.frmOperNorm2, 0, weight=0)
        Grid.columnconfigure(self.frmOperNorm2, 1, weight=1)
        Grid.columnconfigure(self.frmOperNorm2, 2, weight=0)

        Grid.columnconfigure(self.frmOper2, 0, weight=1)
        Grid.rowconfigure(self.frmOper2, 0, weight=1)

        # -Plot frame
        self.frmPlot3 = Frame(self.frmGeom)

        try:
            self.fig3 = Figure(figsize=(5, 4), dpi=100,
                            facecolor=self.master["bg"])
            self.subfig3 = self.fig3.add_subplot(121)
            self.subfig4 = self.fig3.add_subplot(122)
            self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.frmPlot3)
            self.canvas3.draw()
            self.canvas3.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

            self.canvasToolbar3 = NavigationToolbar2Tk(
                self.canvas3, self.frmPlot3)
            self.canvasToolbar3.update()
            self.canvas3._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

            self.frmPlot3.grid(row=0, column=1, rowspan=3,
                            sticky=N + S + E + W)

            self.matplotlib_embedded = True
        except Exception:
            self.matplotlib_embedded = False

            print("WARNING: Matplotlib couldn't be embedded in TkAgg.\nUsing independent window instead",
                file=sys.stderr)

        # --Spectral parameters frame
        self.frmSpectralParameters2 = LabelFrame(
            self.frmGeom, text="Spectral parameters")
        self.frmSpectralParameters2.grid(row=2, column=0, sticky=S + E + W)
        self.ParHVL12 = ParBox(self.frmSpectralParameters2, self.load, lblText="Loaded Spectra", row=0,
                            read_only=True,
                            helpTxt="Thickness of Al at which the dose produced by the spectrum is halved, according to the exponential attenuation model.")
        self.ParHVL22 = ParBox(self.frmSpectralParameters2, self.geo, lblText="Geo file", row=1,
                            read_only=True,
                            helpTxt="Thickness of Al at which the dose produced by the spectrum after crossing a HVL is halved again, according to the exponential attenuation model.")
        self.ParHVL32 = ParBox(self.frmSpectralParameters2, self.HVL32, lblText="# of Views", row=2,
                            read_only=False,
                            helpTxt="How many projections")
        self.ParHVL42 = ParBox(self.frmSpectralParameters2, self.HVL42, lblText="Starting Angle", row=3,
                            read_only=False,
                            helpTxt="Starting angle")
        self.ParHVL52 = ParBox(self.frmSpectralParameters2, self.HVL52, lblText="Ending Angle", row=4,
                            read_only=False,
                            helpTxt="Ending angle of projection")
        self.ParHVL62 = ParBox(self.frmSpectralParameters2, self.doseperproj, lblText="mGy", row=5,
                            read_only=True,
                            helpTxt="Thickness of Cu at which the dose produced by the spectrum is halved, according to the exponential attenuation model.")

        Grid.columnconfigure(self.frmGeom, 0, weight=1)

        if self.matplotlib_embedded:
            # If not embedding, use the whole window
            Grid.columnconfigure(self.frmGeom, 1, weight=3)

        Grid.rowconfigure(self.frmGeom, 0, weight=1)
        Grid.rowconfigure(self.frmGeom, 1, weight=1)

    def update_phan_map(self):

        try:
            now = int(self.lstHistory2.curselection()[0])
            
            self.lstHistory2.delete(0, END)

            self.phantom.phan_map[now] = self.AttenMaterial.get()

            for ii, item in enumerate(self.phantom.phan_map):
                self.lstHistory2.insert(END, f'{ii},' + item)

        except IndexError:  # Ignore if nothing selected
            pass

    def init_sino(self):

        """Initialize sino frame"""

        self.frmRecon = LabelFrame(
            self.frmSino, text="Reconstruction operations")
        self.frmRecon.grid(row=0, column=0, sticky=N + S + E + W)
        self.lblReconPar = Label(
            self.frmRecon, text="Reconstruction Algorithm")
        self.lblReconPar.grid(row=2, column=0,sticky= W)
        self.cmbReconType = Combobox(
            self.frmRecon, textvariable=self.algorithm)
        self.cmbReconType.grid(row=3, column=0,sticky=E + W)
        self.cmbReconType2 = Combobox(
            self.frmRecon, textvariable=self.filt)
        self.lblReconPar2 = Label(
            self.frmRecon, text="Filter")
        self.lblReconPar2.grid(row=4, column=0,sticky=E + W)

        #self.cmbReconType["values"] = list(map(_add_element_name, mu_list))
        # row=5, column=0,rowspan=4, sticky=E + W)
        self.cmbReconType2.grid(row=5, column=0,sticky=E + W)

        self.frmReconParameters = LabelFrame(
            self.frmSino, text="Spectral parameters")
        self.frmReconParameters.grid(row=1, column=0, sticky=N + S + E + W)

        # self.get_proj = ParBox(
        #     self.get_proj, self.AttenThick2, lblText="Thickness", unitsTxt="cm", row=1)
        # self.frmRecon = LabelFrame(self.frmRecon, text="Spectral parameters")
        # self.frmSpectralParameters2.grid(row=2, column=0, sticky=S + E + W)
        self.Parrecon = ParBox(self.frmReconParameters, self.niter, lblText="Number of Iterations", row=0,
                        read_only=False,
                        helpTxt="Thickness of Al at which the dose produced by the spectrum is halved, according to the exponential attenuation model.")
        self.cmdRecon = Button(self.frmReconParameters, text="Reconstruct")
        self.cmdRecon["command"] = self.reconstruct
        self.cmdRecon.bind('<Return>', lambda event: self.reconstruct())
        self.cmdRecon.bind(
            '<KP_Enter>', lambda event: self.reconstruct())  # Enter (num. kb)
        # row=6, column=0, rowspan=4, sticky=E + W)
        self.cmdRecon.grid(row=1, column=0,columnspan=2,sticky=S+E + W)

        self.ParSave = ParBox(self.frmReconParameters, self.filename, lblText="File Name", row=2,
                        read_only=False,
                        helpTxt="Thickness of Al at which the dose produced by the spectrum is halved, according to the exponential attenuation model.")
        
        self.cmdSave = Button(self.frmReconParameters, text="Save Projections")
        self.cmdSave["command"] = self.saveproj
        self.cmdSave.bind('<Return>', lambda event: self.saveproj())
        self.cmdSave.bind(
            '<KP_Enter>', lambda event: self.saveproj())  # Enter (num. kb)
        # row=6, column=0, rowspan=4, sticky=E + W)
        self.cmdSave.grid(row=3, column=0,columnspan=2,sticky=E + W)

        Grid.columnconfigure(self.frmReconParameters, 0, weight=0)
        Grid.columnconfigure(self.frmReconParameters, 1, weight=1)
        Grid.columnconfigure(self.frmReconParameters, 2, weight=0)


        Grid.columnconfigure(self.frmRecon, 0, weight=1)
        Grid.columnconfigure(self.frmRecon, 1, weight=0)
        Grid.rowconfigure(self.frmSino, 0, weight=1)
        Grid.rowconfigure(self.frmSino, 1, weight=0)
        # Grid.rowconfigure(self.frmSino, 2, weight=0)
        # Grid.columnconfigure(self.frmRecon, 2, weight=0)

        self.frmPlot4 = Frame(self.frmSino)

        try:
            self.fig4 = Figure(figsize=(5, 4), dpi=100,
                            facecolor=self.master["bg"])
            self.subfig5 = self.fig4.add_subplot(121)
            self.subfig6 = self.fig4.add_subplot(122)
            self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.frmPlot4)
            self.canvas4.draw()
            self.canvas4.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

            self.canvasToolbar4 = NavigationToolbar2Tk(
                self.canvas4, self.frmPlot4)
            self.canvasToolbar4.update()
            self.canvas4._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

            self.frmPlot4.grid(row=0, column=1, rowspan=5,
                            sticky=N + S + E + W)

            self.matplotlib_embedded = True
        except Exception:
            self.matplotlib_embedded = False
            # self.cmdShowPlot = Button(self.frmPlot,text="Open plot window")
            # self.cmdShowPlot["command"] = self.update_plot
            # self.cmdShowPlot.grid(row=0,column=0)
            print("WARNING: Matplotlib couldn't be embedded in TkAgg.\nUsing independent window instead",
                file=sys.stderr)

        Grid.columnconfigure(self.frmSino, 0, weight=1)

        if self.matplotlib_embedded:
            # If not embedding, use the whole window
            Grid.columnconfigure(self.frmSino, 1, weight=3)

        Grid.rowconfigure(self.frmSino, 0, weight=1)
        Grid.rowconfigure(self.frmSino, 1, weight=1)

    def init_reso(self):
        # self.frmReso.grid(row=0, column=0, rowspan=4,
        #         columnspan=5, sticky=N + S + E + W)

        """Initialize reso frame"""

        # -Operations frame
        self.frmRecon2 = LabelFrame(
            self.frmReso, text="Save file")
        self.frmRecon2.grid(row=0, column=0,
                        sticky=N + S + E + W)
        # self.cmdAtten2.grid(row=2, column=0, columnspan=3, sticky=E + W)
        # --Attenuation
        # self.frmRecon2Par = LabelFrame(self.frmRecon2, text="Attenuate")
        # # row=4, column=0,rowspan=4, sticky=E + W)
        # self.frmRecon2Par.grid(sticky=E + W)
        self.lblRecon2Par = Label(
            self.frmRecon2, text="File Type")
        self.lblRecon2Par.grid(row=0,column=0,sticky=E + W)
        self.cmbRecon2Type = Combobox(
            self.frmRecon2, textvariable=self.filetype)
        self.cmbRecon2Type.grid(row=1,column=1,sticky=E + W)
        self.Par1 = ParBox(self.frmRecon2, self.filename, lblText="File Name", row=3,
                        read_only=False,
                        helpTxt="Thickness of Al at which the dose produced by the spectrum is halved, according to the exponential attenuation model.")
        self.Par2 = ParBox(self.frmRecon2, self.label, lblText="Plotting Label", row=4,
                        read_only=False,
                        helpTxt="label that will appear in the legend of the analysis plots")
        self.cmdRecon2 = Button(self.frmReso, text="Save Reconstruction")
        self.cmdRecon2["command"] = self.saverecon
        self.cmdRecon2.bind('<Return>', lambda event: self.saverecon())
        self.cmdRecon2.bind(
            '<KP_Enter>', lambda event: self.saverecon())  # Enter (num. kb)
        # row=6, column=0, rowspan=4, sticky=E + W)
        self.cmdRecon2.grid(row=1,column=0,sticky=S+E + W)

        self.cmdRecon3 = Button(self.frmReso, text="Analyze Phantom")
        self.cmdRecon3["command"] = self.analyse_phan
        self.cmdRecon3.bind('<Return>', lambda event: self.analyse_phan())
        self.cmdRecon3.bind(
            '<KP_Enter>', lambda event: self.analyse_phan())  # Enter (num. kb)
        self.cmdRecon3.grid(sticky=S+E + W,row=2, column=0)


        Grid.columnconfigure(self.frmRecon2, 0, weight=0)
        Grid.columnconfigure(self.frmRecon2, 1, weight=1)
        Grid.columnconfigure(self.frmRecon2, 2, weight=0)
        Grid.rowconfigure(self.frmReso, 0, weight=1)
        Grid.rowconfigure(self.frmReso, 1, weight=0)
        Grid.rowconfigure(self.frmReso, 2, weight=0)


        self.frmPlot5 = Frame(self.frmReso)

        try:
            self.fig5 = Figure(figsize=(5, 4), dpi=100,
                            facecolor=self.master["bg"])
            self.subfig7 = self.fig5.add_subplot(121)
            self.subfig8 = self.fig5.add_subplot(122)
            self.canvas5 = FigureCanvasTkAgg(self.fig5, master=self.frmPlot5)
            self.canvas5.draw()
            self.canvas5.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

            self.canvasToolbar5 = NavigationToolbar2Tk(
                self.canvas5, self.frmPlot5)
            self.canvasToolbar5.update()
            self.canvas5._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

            self.frmPlot5.grid(row=0, column=1, rowspan=5,
                            sticky=N + S + E + W)

            self.matplotlib_embedded = True
        except Exception:
            self.matplotlib_embedded = False
            print("WARNING: Matplotlib couldn't be embedded in TkAgg.\nUsing independent window instead",
                file=sys.stderr)

        Grid.columnconfigure(self.frmReso, 0, weight=1)

        if self.matplotlib_embedded:
            # If not embedding, use the whole window
            Grid.columnconfigure(self.frmReso, 1, weight=3)

        Grid.rowconfigure(self.frmReso, 0, weight=1)
        Grid.rowconfigure(self.frmReso, 1, weight=1)

    def init_outp(self):

        """Initialize reso frame"""

        # -Operations frame
        self.frmRecon2 = LabelFrame(
            self.frmOutp, text="Reconstruction operations")
        self.frmRecon2.grid(row=0, column=0, sticky=N + S + E + W)
        # self.cmdAtten2.grid(row=2, column=0, columnspan=3, sticky=E + W)
        # --Attenuation
        # self.frmRecon2Par = LabelFrame(self.frmRecon2, text="Attenuate")
        # row=4, column=0,rowspan=4, sticky=E + W)
        # self.frmRecon2Par.grid(sticky=E + W)
        self.lblRecon2Par = Label(
            self.frmRecon2, text="File Type")
        self.lblRecon2Par.grid(sticky=E + W)
        self.cmbRecon2Type = Combobox(
            self.frmRecon2, textvariable=self.filetype)
        self.cmbRecon2Type.grid(sticky=E + W)
        self.Par1 = ParBox(self.frmRecon2, self.filename, lblText="File Name", row=5,
                        read_only=False,
                        helpTxt="Thickness of Al at which the dose produced by the spectrum is halved, according to the exponential attenuation model.")
        self.cmdRecon2 = Button(self.frmRecon2, text="Save Reconstruction")
        self.cmdRecon2["command"] = self.saverecon
        self.cmdRecon2.bind('<Return>', lambda event: self.saverecon())
        self.cmdRecon2.bind(
            '<KP_Enter>', lambda event: self.saverecon())  # Enter (num. kb)
        self.cmdRecon2.grid(sticky=E + W)
        
        Grid.columnconfigure(self.frmOutp, 0, weight=0)
        Grid.columnconfigure(self.frmOutp, 1, weight=1)
        Grid.columnconfigure(self.frmOutp, 2, weight=0)
        Grid.rowconfigure(self.frmOutp, 0, weight=1)
        Grid.rowconfigure(self.frmOutp, 1, weight=1)

        self.frmPlot6 = Frame(self.frmOutp)

        try:
            self.fig6 = Figure(figsize=(5, 4), dpi=100,
                            facecolor=self.master["bg"])
            self.subfig9 = self.fig6.add_subplot(121)
            self.subfig10 = self.fig6.add_subplot(122)
            self.canvas6 = FigureCanvasTkAgg(self.fig6, master=self.frmPlot6)
            self.canvas6.draw()
            self.canvas6.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

            self.canvasToolbar6 = NavigationToolbar2Tk(
                self.canvas6, self.frmPlot6)
            self.canvasToolbar6.update()
            self.canvas6._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

            self.frmPlot6.grid(row=0, column=1, rowspan=5,
                            sticky=N + S + E + W)

            self.matplotlib_embedded = True
        except Exception:
            self.matplotlib_embedded = False
            print("WARNING: Matplotlib couldn't be embedded in TkAgg.\nUsing independent window instead",
                file=sys.stderr)

        if self.matplotlib_embedded:
            # If not embedding, use the whole window
            Grid.columnconfigure(self.frmReso, 1, weight=3)

        Grid.rowconfigure(self.frmReso, 0, weight=1)
        Grid.rowconfigure(self.frmReso, 1, weight=1)
        
    def enable_analyze_buttons(self):
        """
        Enable widgets requiring a calculated spectrum to work.

        """
        self.cmdCleanHistory["state"] = "normal"
        self.cmdExport["state"] = "normal"
        self.cmdAtten["state"] = "normal"
        self.cmdNorm["state"] = "normal"

    def update_plot(self):
        """
        Update the canvas after plotting something.
        If matplotlib is not embedded, show it in an independent window.

        """
        if self.matplotlib_embedded:
            self.subfig.clear()
            self.spectra[self.active_spec].get_plot(self.subfig)
            self.canvas.draw()
            self.canvasToolbar.update()
            self.fig.tight_layout()
        else:
            # FIXME: Update if independent window is opened
            self.spectra[self.active_spec].show_plot(block=False)
        self.update_param()

    def update_plot_kern(self):
        """
        Update the canvas after plotting something.
        If matplotlib is not embedded, show it in an independent window.

        """
        if self.matplotlib_embedded:
            self.subfigKern.clear()
            self.kernel.get_plot(self.subfigKern)
            self.kernel.get_plot_mtf_real(self.subfigKern2,label=self.load.get())
            self.figKern.tight_layout()
            self.canvasKern.draw()
            self.canvasToolbarKern.update()

        else:
            # FIXME: Update if independent window is opened
            print('not going')
        self.update_param()

    def update_plot_proj(self):
        """
        Update the canvas after plotting something.
        If matplotlib is not embedded, show it in an independent window.

        """
        if self.matplotlib_embedded:
            # self.subfigGeo.clear()
            if len(self.phantom.proj.shape) > 2:
                self.tracker3 = fc.IndexTracker(
                    self.subfig5, self.phantom.proj.transpose([1,2,0]))
                self.fig4.canvas.mpl_connect(
                    'scroll_event', self.tracker3.onscroll)
                self.tracker4 = fc.IndexTracker(
                    self.subfig6, self.phantom.proj)
                self.fig4.canvas.mpl_connect(
                    'scroll_event', self.tracker4.onscroll)
                self.fig4.tight_layout()
                self.canvas4.draw()
                self.canvasToolbar4.update()
            else:
                self.subfig5.imshow(self.phantom.proj.T)
                self.subfig6.imshow(np.transpose(self.phantom.proj))
                self.fig4.tight_layout()
                self.canvas4.draw()
                self.canvasToolbar4.update()                

    def update_plot_recon(self):
        """
        Update the canvas after plotting something.
        If matplotlib is not embedded, show it in an independent window.

        """
        if self.matplotlib_embedded:
            print('Starting FDK reconstruction ...')
            self.tracker5 = fc.IndexTracker(self.subfig7, self.phantom.img.T)
            self.fig5.canvas.mpl_connect(
                'scroll_event', self.tracker5.onscroll)
            self.tracker6 = fc.IndexTracker(self.subfig8, self.phantom.img)
            self.fig5.canvas.mpl_connect(
                'scroll_event', self.tracker6.onscroll)
            self.canvas5.draw()
            self.canvasToolbar5.update()

    def update_plot_geo(self):
        """
        Update the canvas after plotting something.
        If matplotlib is not embedded, show it in an independent window.

        """
        if self.matplotlib_embedded:
            self.tracker = fc.IndexTracker(self.subfig3, self.phantom.phantom.T)
            self.fig3.canvas.mpl_connect('scroll_event', self.tracker.onscroll)
            self.tracker2 = fc.IndexTracker(
                self.subfig4, self.phantom.phantom)
            self.fig3.canvas.mpl_connect(
                'scroll_event', self.tracker2.onscroll)
            self.canvas3.draw()
            self.canvasToolbar3.update()

        else:
            # FIXME: Update if independent window is opened
            print('Failed to update plot.')


    def update_param(self):
        """
        Update parameters calculated from the active spectrum.

        """
        if self.kV.get():

            hvlAl = self.spectra[self.active_spec].hvl(
                0.5, self.fluence_to_dose, self.mu_Al)
            qvlAl = self.spectra[self.active_spec].hvl(
                0.25, self.fluence_to_dose, self.mu_Al)

            hvlCu = self.spectra[self.active_spec].hvl(
                0.5, self.fluence_to_dose, self.mu_Cu)
            qvlCu = self.spectra[self.active_spec].hvl(
                0.25, self.fluence_to_dose, self.mu_Cu)

            # TODO: (?) cache the results
            self.HVL1.set('%s' % float('%.3g' % hvlAl))
            self.HVL2.set('%s' % float('%.3g' % (qvlAl - hvlAl)))
            self.HVL3.set('%s' % float('%.3g' % hvlCu))
            self.HVL4.set('%s' % float('%.3g' % (qvlCu - hvlCu)))

            self.number.set('%s' % float('%.3g' %
                                         (self.spectra[self.active_spec].get_norm())))
            self.energy.set('%s' % float(
                '%.3g' % (self.spectra[self.active_spec].get_norm(lambda x: x))))
            self.dose.set('%s' % float(
                '%.3g' % (self.spectra[self.active_spec].get_norm(self.fluence_to_dose))))

    def monitor_bar(self, a, b):
        """
        Update the progress bar.

        Args:
            a (int): The number of items already calculated.
            b (int): The total number of items to calculate.

        """
        self.barProgress["value"] = a
        self.barProgress["maximum"] = b

    def clean_history(self):
        """
        Clean the spectra history.
        """

        try:
            now = int(self.lstHistory.curselection()[0])
            if now == len(self.spectra) - 1:  # No need to slice
                return
            self.spectra = self.spectra[0:now + 1]
            self.lstHistory.delete(now + 1, END)

        except IndexError:  # Ignore if nothing selected
            pass

    def export(self):
        """
        Export the selected spectrum in xlsx format, showing a file dialog to choose the route.

        """
        if self.lstHistory.curselection() == ():
            selection = -1
        else:
            selection = int(self.lstHistory.curselection()[0])
        file_opt = options = {}
        options['defaultextension'] = '.xlsx'
        options['filetypes'] = [
            ('Excel Spreadsheet', '.xlsx'), ('Comma-separated values', '.csv')]
        options['initialfile'] = 'spectrum.xlsx'
        options['parent'] = self
        options['title'] = 'Export spectrum'
        filename = tkinter.filedialog.asksaveasfilename(**file_opt)
        if not filename:  # Ignore if canceled
            return
        ext = filename.split(".")[-1]

        if ext == "xlsx":
            self.spectra[selection].export_xlsx(filename)
        elif ext == "csv":
            self.spectra[selection].export_csv(filename)
        else:
            messagebox.showerror("Error",
                                 "Unknown file extension: " + ext + "\nUse the file types from the dialog to export.")

    def calculate(self):
        """
        Calculates a new spectrum using the parameters in the GUI.

        """
        # If a calculation was being held, abort it instead
        try:
            if self.calc_thread.is_alive():
                self.abort_calculation = True
                self.cmdCalculate["text"] = "(Aborting)"
                self.cmdCalculate["state"] = "disabled"
                return
        except AttributeError:  # If there is no calculation thread, there is nothing to worry about
            pass

        self.calculation_count = 0
        self.calculation_total = self.NumE.get()
        z = int(_remove_element_name(self.Z.get()))

        def monitor(a, b):
            # Will be executed in calculation thread. Values are only collected,
            # Tk must be updated from main thread only.
            self.calculation_count = a
            self.calculation_total = b
            if self.abort_calculation:
                self.queue_calculation.put(False)
                exit(1)

        def callback():  # Carry the calculation in a different thread to avoid blocking
            try:
                s = fc.calculate_spectrum(self.E0.get(), self.Theta.get(), self.EMin.get(
                ), self.NumE.get(), phi=self.Phi.get(), epsrel=self.Eps.get(), monitor=monitor, z=z)
                self.spectra = [s]
                self.queue_calculation.put(True)
            except Exception as e:
                print_exc()
                self.queue_calculation.put(False)
                messagebox.showerror(
                    "Error", "An error occurred during the calculation:\n%s\nCheck the parameters are valid." % str(e))

        self.queue_calculation = queue.Queue(maxsize=1)
        # The child will fill the queue with a value indicating whether an error occured.
        # Ask the calculation thread to end (when monitor is executed)
        self.abort_calculation = False
        self.calc_thread = threading.Thread(target=callback)
        self.calc_thread.setDaemon(True)
        self.calc_thread.start()

        self.cmdCalculate["text"] = "Abort"
        self.after(250, self.wait_for_calculation)

    def loadfile(self):
        """
        Calculates a new spectrum using the parameters in the GUI.

        """
        # If a calculation was being held, abort it instead
        try:
            self.load.get()
        except AttributeError:  # If there is no calculation thread, there is nothing to worry about
            messagebox.showerror(
                "Error", "An error occurred duringing loading:\n%s\nCheck that a file is selected." % str(e))

        self.calculation_count = 0
        self.calculation_total = self.NumE.get()

        z = int(_remove_element_name(self.Z.get()))

        def callback():  # Carry the calculation in a different thread to avoid blocking
            try:

                s = fc.Spectrum()

                energies = []
                fluence = []

                with open(os.path.join(fc.data_path, "MV_spectra", f'{self.load.get()}.txt')) as f:
                    for line in f:
                        # import pdb; pdb.set_trace()
                        energies.append(float(line.split()[0]))
                        fluence.append(float(line.split()[1]))

                # Check if MV

                s.x = np.array(energies)*1000  # to keV
                s.y = np.array(fluence)

                if max(s.x) > 500:
                    self.kV.set(False)

                self.spectra = [s]
                self.queue_calculation.put(True)

            except Exception as e:
                print_exc()
                self.queue_calculation.put(False)
                messagebox.showerror(
                    "Error", "An error occurred during the calculation:\n%s\nCheck the parameters are valid." % str(e))

        self.queue_calculation = queue.Queue(maxsize=1)
        self.abort_calculation = False
        self.calc_thread = threading.Thread(target=callback)
        self.calc_thread.setDaemon(True)
        self.calc_thread.start()

        self.cmdload["text"] = "Abort"
        self.after(250, self.wait_for_calculation)

    def computeKernel(self):
        """
        Calculates a new spectrum using the parameters in the GUI.
        """

        self.calculation_count = 0
        self.calculation_total = self.NumE.get()

        def callback():  # Carry the calculation in a different thread to avoid blocking
            try:

                self.kernel = fc.Kernel(
                    self.spectra[-1], self.det.get())

                self.queue_calculation.put(True)

            except Exception as e:
                print_exc()
                self.queue_calculation.put(False)
                messagebox.showerror(
                    "Error", "An error occurred during the calculation:\n%s\nCheck the parameters are valid." % str(e))

        self.queue_calculation = queue.Queue(maxsize=1)
        # The child will fill the queue with a value indicating whether an error occured.
        # Ask the calculation thread to end (when monitor is executed)
        self.abort_calculation = False
        self.calc_thread = threading.Thread(target=callback)
        self.calc_thread.setDaemon(True)
        self.calc_thread.start()

        self.cmdCalculate["text"] = "Abort"
        self.after(250, self.wait_for_kernel)

    def analyse_phan(self):
        """
        Calculates a new spectrum using the parameters in the GUI.

        """
        
        if self.matplotlib_embedded:

            self.phantom.analyse_515(self.phantom.img[5],[self.subfig9,self.subfig10],self.label.get())
            self.canvas6.draw()
            self.canvasToolbar6.update()
            self.fig6.tight_layout()
            self.select(6)           

    def reconstruct(self):
        """
        Calculates a new spectrum using the parameters in the GUI.

        """

        self.calculation_count = 0
        self.calculation_total = self.NumE.get()

        def callback():  # Carry the calculation in a different thread to avoid blocking
            try:

                print('FDK')

                self.phantom.reconstruct('FDK',self.filt.get())

                self.queue_calculation.put(True)

            except Exception as e:
                print_exc()
                self.queue_calculation.put(False)
                messagebox.showerror(
                    "Error", "An error occurred during the calculation:\n%s\nTigre not working resorting to astra" % str(e))

        self.queue_calculation = queue.Queue(maxsize=1)
        # The child will fill the queue with a value indicating whether an error occured.
        # Ask the calculation thread to end (when monitor is executed)
        self.abort_calculation = False
        self.calc_thread = threading.Thread(target=callback)
        self.calc_thread.setDaemon(True)
        self.calc_thread.start()

        self.cmdCalculate["text"] = "Abort"
        self.after(250, self.wait_for_recon)

    def saverecon(self):
        """
        Calculates a new spectrum using the parameters in the GUI.
        """
                
        print('saving reconstruction...')

        np.save(os.path.join(fc.data_path,'recons',self.filename.get()),self.phantom.img)

        print('Reconstruction saved to ',os.path.join(fc.data_path,'recons',self.filename.get()))

    def saveproj(self):
        """
        Calculates a new spectrum using the parameters in the GUI.
        """
                
        print('saving projections...')
        np.save(os.path.join(fc.data_path,'projs',self.filename.get()),self.phantom.proj)

        print('Projections saved to ',os.path.join(fc.data_path,'projs',self.filename.get()))

    def add_focal_spot(self):

        # fs size times magnification divided by the pixel pitch

        # self.computeGeometry()
        # self.select(2)

        fs_size_in_pix = (self.fs_size.get() * 1536 / 1000)/ 0.672

        self.kernel.add_focal_spot(fs_size_in_pix)

        self.update_plot_kern()

    def computeProjection(self):
        """
        Calculates a new spectrum using the parameters in the GUI.

        """

        self.calculation_count = 0
        self.calculation_total = self.NumE.get()

        def callback():  # Carry the calculation in a different thread to avoid blocking
            try:

                print(np.squeeze(np.array(self.phantom.phantom)).shape)
                self.angles = np.linspace(self.HVL42.get(), self.HVL52.get(
                )*(np.pi)/180, int(self.HVL32.get()), dtype=np.float32)
                # print(self.angles)
                energy_deposition_file = os.path.join(
                    fc.data_path, "Detectors", self.det.get(), 'EnergyDeposition.npy')
                self.phan_map = ['air','water',"G4_BONE_COMPACT_ICRU"]
                # real one ['air','water','G4_LUNG_ICRP',"G4_BONE_COMPACT_ICRU","G4_BONE_CORTICAL_ICRP","G4_ADIPOSE_TISSUE_ICRP","G4_BRAIN_ICRP","G4_B-100_BONE"]
            #     ['air','water','Spongiosa_Bone_ICRP','G4_BONE_COMPACT_ICRU',
            #  'G4_BONE_CORTICAL_ICRP','C4_Vertebra_ICRP','D6_Vertebra_ICRP','G4_B-100_BONE']
                # self.phan_map = ['air','water','adipose','adipose','adipose','adipose','adipose','adipose']
               

                # value = self.current.get()
                self.normalize2()

                self.phantom.return_projs(self.kernel,
                            self.spectra[self.active_spec],
                            self.angles,
                            nphoton = self.nphoton,
                            mgy =  self.noise,
                            scat_on = self.scatter_on.get(),
                            det_on = self.det_on.get())# I think it should be inverse
                
                print(np.array(self.phantom.proj).shape)
                self.queue_calculation.put(True)

            except Exception as e:
                print_exc()
                self.queue_calculation.put(False)
                messagebox.showerror(
                    "Error", "An error occurred during the calculation:\n%s\nCheck the parameters are valid." % str(e))

        self.queue_calculation = queue.Queue(maxsize=1)
        # The child will fill the queue with a value indicating whether an error occured.
        # Ask the calculation thread to end (when monitor is executed)
        self.abort_calculation = False
        self.calc_thread = threading.Thread(target=callback)
        self.calc_thread.setDaemon(True)
        self.calc_thread.start()

        self.cmdCalculate["text"] = "Abort"
        self.after(250, self.wait_for_proj)

    def computeGeometry(self):
        """
        Calculates a new spectrum using the parameters in the GUI.

        """

        self.calculation_count = 0
        self.calculation_total = self.NumE.get()

        def callback():  # Carry the calculation in a different thread to avoid blocking
            try:
                dispatcher={'Catphan_515':fc.Catphan_515,
                            'XCAT':fc.XCAT,
                            'Catphan_projections':fc.Catphan_projections,
                            'Catphan_MTF':fc.Catphan_MTF,
                            'Catphan_404':fc.Catphan_404}
                try:
                    function=dispatcher[self.geo.get()]
                except KeyError:
                    raise ValueError('Invalid phantom module name')

                self.phantom = function()
                self.queue_calculation.put(True)

                self.lstHistory2.delete(0,END)
                for ii, item in enumerate(self.phantom.phan_map):
                    self.lstHistory2.insert(END, f'{ii},' + item)

            except Exception as e:
                print_exc()
                self.queue_calculation.put(False)
                messagebox.showerror(
                    "Error", "An error occurred during the calculation:\n%s\nCheck the parameters are valid." % str(e))

        self.queue_calculation = queue.Queue(maxsize=1)
        # The child will fill the queue with a value indicating whether an error occured.
        # Ask the calculation thread to end (when monitor is executed)
        self.abort_calculation = False
        self.calc_thread = threading.Thread(target=callback)
        self.calc_thread.setDaemon(True)
        self.calc_thread.start()

        self.cmdCalculate["text"] = "Abort"
        self.after(250, self.wait_for_geo)

    def wait_for_calculation(self):
        """
        Polling method to wait for the calculation thread to finish. Also updates monitor_bar.

        """
        self.monitor_bar(self.calculation_count, self.calculation_total)
        if self.queue_calculation.full():  # Calculation ended
            self.cmdCalculate["text"] = "Calculate"
            self.cmdload["text"] = "Load"
            self.cmdCalculate["state"] = "normal"
            self.monitor_bar(0, 0)
            if self.queue_calculation.get_nowait():  # Calculation ended successfully
                self.lstHistory.delete(0, END)
                self.lstHistory.insert(END, "Calculated")
                self.enable_analyze_buttons()
                self.active_spec = 0
                self.update_plot()
                self.select(1)  # Open analyse tab
            else:
                pass
        else:
            self.after(250, self.wait_for_calculation)

    def wait_for_recon(self):
        """
        Polling method to wait for the calculation thread to finish. Also updates monitor_bar.

        """
        self.monitor_bar(self.calculation_count, self.calculation_total)
        if self.queue_calculation.full():  # Calculation ended
            self.cmdCalculate["text"] = "Calculate"
            self.cmdCalculate["state"] = "normal"
            self.monitor_bar(0, 0)
            if self.queue_calculation.get_nowait():  # Calculation ended successfully
                # self.lstHistory.delete(0, END)
                # self.lstHistory.insert(END, "Calculated")
                self.enable_analyze_buttons()
                # self.active_spec = 0
                self.update_plot_recon()
                self.select(5)  # Open analyse tab
            else:
                pass
        else:
            self.after(250, self.wait_for_recon)

    def wait_for_proj(self):
        """
        Polling method to wait for the calculation thread to finish. Also updates monitor_bar.

        """
        self.monitor_bar(self.calculation_count, self.calculation_total)
        if self.queue_calculation.full():  # Calculation ended
            self.cmdCalculate["text"] = "Calculate"
            self.cmdCalculate["state"] = "normal"
            self.monitor_bar(0, 0)
            if self.queue_calculation.get_nowait():  # Calculation ended successfully
                # self.lstHistory.delete(0, END)
                # self.lstHistory.insert(END, "Calculated")
                self.enable_analyze_buttons()
                # self.active_spec = 0
                self.update_plot_proj()
                self.select(4)  # Open analyse tab
            else:
                pass
        else:
            self.after(250, self.wait_for_proj)

    def wait_for_kernel(self):
        """
        Polling method to wait for the calculation thread to finish. Also updates monitor_bar.

        """
        self.monitor_bar(self.calculation_count, self.calculation_total)
        if self.queue_calculation.full():  # Calculation ended
            self.cmdCalculate["text"] = "Calculate"
            self.cmdCalculate["state"] = "normal"
            self.monitor_bar(0, 0)
            if self.queue_calculation.get_nowait():  # Calculation ended successfully
                # self.lstHistory.delete(0, END)
                # self.lstHistory.insert(END, "Calculated")
                self.enable_analyze_buttons()
                # self.active_spec = 0
                self.update_plot_kern()
                self.select(2)  # Open analyse tab
            else:
                pass
        else:
            self.after(250, self.wait_for_kernel)

    def wait_for_geo(self):
        """
        Polling method to wait for the calculation thread to finish. Also updates monitor_bar.

        """
        self.monitor_bar(self.calculation_count, self.calculation_total)
        if self.queue_calculation.full():  # Calculation ended
            self.cmdCalculate["text"] = "Calculate"
            self.cmdCalculate["state"] = "normal"
            self.monitor_bar(0, 0)
            if self.queue_calculation.get_nowait():  # Calculation ended successfully
                # self.lstHistory.delete(0, END)
                # self.lstHistory.insert(END, "Calculated")
                self.enable_analyze_buttons()
                # self.active_spec = 0
                self.update_plot_geo()
                self.select(3)  # Open analyse tab
            else:
                pass
        else:
            self.after(250, self.wait_for_geo)

    def attenuate(self):
        """
        Attenuate the active spectrum according to the parameters in the GUI.

        """
        s2 = self.spectra[-1].clone()
        s2.attenuate(self.AttenThick.get(),
                     fc.get_mu(_remove_element_name(self.AttenMaterial.get())))
        self.spectra.append(s2)
        self.lstHistory.insert(
            END, "Attenuated: " + str(self.AttenThick.get()) + "cm of " + self.AttenMaterial.get())
        self.lstHistory.selection_clear(0, len(self.spectra) - 2)
        self.lstHistory.selection_set(len(self.spectra) - 1)
        self.update_plot()
        pass

    def normalize(self):
        """
        Normalize the active spectrum according to the parameters in the GUI.

        """
        value = self.NormValue.get()
        crit = self.NormCriterion.get()
        if value <= 0:
            messagebox.showerror(
                "Error", "The norm of a spectrum must be a positive number.")
            return
        if crit not in self.criteriaList:
            messagebox.showerror("Error", "An unknown criterion was selected.")
            return
        s2 = self.spectra[-1].clone()
        if crit == self.criteriaList[0]:
            s2.set_norm(value)
        elif crit == self.criteriaList[1]:
            s2.set_norm(value, lambda x: x)
        else:  # criteriaList[2]
            s2.set_norm(value, self.fluence_to_dose)
        self.spectra.append(s2)
        self.lstHistory.insert(
            END, "Normalized: " + crit + " = " + str(value))
        self.lstHistory.selection_clear(0, len(self.spectra) - 2)
        self.lstHistory.selection_set(len(self.spectra) - 1)
        self.update_plot()
        pass

    def normalize2(self):
        """
        Normalize the active spectrum according to the parameters in the GUI.

        """
        value = self.current.get()
        crit = self.NormCriterion2.get()
        if value < 0:
            messagebox.showerror(
                "Error", "The norm of a spectrum must be a positive number.")
            return
        if crit not in self.criteriaList2:
            print('here')
            messagebox.showerror("Error", "An unknown criterion was selected.")
            return
        s2 = self.spectra[-1].clone()
        if crit == self.criteriaList2[0]:
            self.noise = 0.0
            self.nphoton = None
        elif crit == self.criteriaList2[1]:
            self.noise = value
            self.nphoton = None
        elif crit == self.criteriaList2[2]:
            self.noise = value/self.HVL32.get()
            self.nphoton = None
        elif crit == self.criteriaList2[3]:
            self.noise = 0.0
            self.nphoton = value
            # self.noise = self.doseperproj.get()/value
            # print(self.noise,'Scaling of noise')
        else:  # criteriaList[2]
            s2.set_norm2(value, self.fluence_to_dose)
        pass

def main():
    """
    Start an instance of the GUI.

    """
    root = Tk()
    root.style = Style()
    root.style.theme_use("clam")
    root.configure()#background='black')
    app = XpecgenGUI(master=root)
    app.mainloop()


if __name__ == "__main__":
    main()

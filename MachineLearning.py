from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import  QTableWidget, QTableWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("D:\MachineLearning\DataCancer\survey data cancer.csv")
# print("\nNhận dữ liệu đầu vào:")
# print(df)
# print("Mô tả thống kê dữ liệu đầu vào:")
# print(df.describe())
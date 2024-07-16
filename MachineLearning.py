from PyQt6 import QtCore, QtGui, QtWidgets  # Import necessary modules from PyQt6
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem  # Import specific widgets from PyQt6
from matplotlib.backends.backend_qt5agg import \
     FigureCanvasQTAgg as FigureCanvas  # Import FigureCanvas for displaying plots on the interface
from sklearn.metrics import roc_curve, auc  # Import evaluation metrics functions from sklearn
from matplotlib.legend_handler import HandlerLine2D  # Import HandlerLine2D for handling legends in plots
from sklearn import tree  # Import tree from sklearn for decision tree visualization
import pandas as pd  # Import pandas for data handling
import numpy as np  # Import numpy for numerical array handling
import matplotlib.pyplot as plt  # Import matplotlib for plotting

### Step: Data input collection
df = pd.read_csv("D:\MachineLearning\datatrain.csv")
# print("\nReceive input data:")
# print(df)
# print("Statistical description of input data:")
# print(df.describe())

### Step: Data preprocessing
### Attributes are of object data type. Therefore, data must be converted to numerical form.
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['AGE'] = le.fit_transform(df['AGE'])
df['INCOME'] = le.fit_transform(df['INCOME'])
df['LOANAMOUNT'] = le.fit_transform(df['LOANAMOUNT'])
df['CREDITSCORE'] = le.fit_transform(df['CREDITSCORE'])
df['MONTHSEMPLOYED'] = le.fit_transform(df['MONTHSEMPLOYED'])
df['NUMCREDITLINES'] = le.fit_transform(df['NUMCREDITLINES'])
df['INTERESTRATE'] = le.fit_transform(df['INTERESTRATE'])
df['LOANTERM'] = le.fit_transform(df['LOANTERM'])
df['DTIRATIO'] = le.fit_transform(df['DTIRATIO'])
df['EDUCATION'] = le.fit_transform(df['EDUCATION'])
df['EMPLOYMENTTYPE'] = le.fit_transform(df['EMPLOYMENTTYPE'])
df['MARITALSTATUS'] = le.fit_transform(df['MARITALSTATUS'])
df['HASMORTGAGE'] = le.fit_transform(df['HASMORTGAGE'])
df['HASDEPENDENTS'] = le.fit_transform(df['HASDEPENDENTS'])
df['LOANPURPOSE'] = le.fit_transform(df['LOANPURPOSE'])
df['HASCOSIGNER'] = le.fit_transform(df['HASCOSIGNER'])
df['DEFAULT'] = le.fit_transform(df['DEFAULT'])
# print("Dataset after conversion:")
# print(df)

### Split dataset into training and testing files
inputs = df.drop('DEFAULT', axis="columns")
target = df['DEFAULT']
### Train the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(inputs, target, test_size=0.3, random_state=50)
### Step: Build a tree forest model from the training dataset
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=2000,  # Number of trees in the random forest is 2000.
    max_features=16,  # Maximum number of features to consider when splitting at each node is 16.
    min_samples_split=3,  # Minimum number of samples required to split a node is 3.
    max_depth=16,  # Maximum depth of each tree is 16.
    random_state=0  # Random seed used to initialize the random number generator.
)

### Step: Train the model
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)  # Predict labels on the test set
# print("Prediction with input data:")
# print(Y_pred)

### Importance of attributes
inputsss = df.drop('DEFAULT', axis="columns")
feature_imp = pd.Series(clf.feature_importances_, index=inputsss.columns).sort_values(ascending=False)

# print("\nImportance of attributes:")
# print(feature_imp)
### Step: Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(Y_pred, Y_test)  # Accuracy
precision = precision_score(Y_pred, Y_test)  # Precision
recall = recall_score(Y_pred, Y_test)  # Recall

# print("\nModel evaluation:")
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
class show_tree(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots()  # Create a figure and axes object.
        super().__init__(self.fig)  # Initialize the FigureCanvas base class with the figure object.
        plt.close()  # Close the figure object (to prevent unwanted display).
        plt.ion()  # Turn on matplotlib's interactive mode.
        self.axes = self.fig.subplots(2, 2)  # Create a 2x2 subplot grid.
        self.ax.set_frame_on(False)  # Remove frame from the main axes.
        self.ax.set_xticks([])  # Hide x-axis ticks.
        self.ax.set_yticks([])  # Hide y-axis ticks.
        for i, ax in enumerate(self.axes.flat):  # Iterate through each subplot (child axes) in the 2x2 grid.
            trees = clf.estimators_[i]  # Get the i-th decision tree from the Random Forest model.
            _ = tree.plot_tree(trees, feature_names=None, class_names=None, filled=True,
                               ax=ax)  # Plot the decision tree on the corresponding subplot.
        self.draw()  # Redraw the entire figure to display the decision trees.



class show_auc(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots()  # Create a figure and axes object.
        super().__init__(self.fig)  # Initialize the FigureCanvas base class with the figure object.
        plt.close()  # Close the figure object (to prevent unwanted display).
        plt.ion()  # Turn on matplotlib's interactive mode.

        # Different values of n_estimators to be tested
        n_estimators = [1, 2, 4, 8, 16, 32, 64, 128, 300]

        train_results = []  # List to store AUC results on training set.
        test_results = []  # List to store AUC results on test set.

        # Iterate through each n_estimators value to train the model and calculate AUC.
        for estimator in n_estimators:
            rf = RandomForestClassifier(n_estimators=estimator,
                                        n_jobs=-1)  # Initialize RandomForest model with corresponding number of trees.
            rf.fit(X_train, Y_train)  # Train the model on the training set.

            train_pred = rf.predict(X_train)  # Predict on the training set.
            false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train,
                                                                            train_pred)  # Compute FPR and TPR for ROC curve on training set.
            roc_auc = auc(false_positive_rate, true_positive_rate)  # Compute AUC for ROC curve on training set.
            train_results.append(roc_auc)  # Store AUC result for training set.

            y_pred = rf.predict(X_test)  # Predict on the test set.
            false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test,
                                                                            y_pred)  # Compute FPR and TPR for ROC curve on test set.
            roc_auc = auc(false_positive_rate, true_positive_rate)  # Compute AUC for ROC curve on test set.
            test_results.append(roc_auc)  # Store AUC result for test set.

        # Plot AUC curve for training and test sets.
        line1, = self.ax.plot(n_estimators, train_results, "b",
                              label="Train AUC")  # Plot line representing AUC of training set.
        line2, = self.ax.plot(n_estimators, test_results, "r",
                              label="Test AUC")  # Plot line representing AUC of test set.

        self.fig.legend(handler_map={line1: HandlerLine2D(numpoints=2)})  # Create legend for the plot.
        self.ax.set_ylabel("auc score")  # Set y-axis label.
        self.ax.set_xlabel("n_estimators")  # Set x-axis label.
        self.fig.suptitle('Area Under The Curve (AUC)', size=8)  # Set title for the plot.


class show_roc(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots()  # Create a figure and axes object.
        super().__init__(self.fig)  # Initialize the FigureCanvas base class with the figure object.
        plt.close()  # Close the figure object (to prevent unwanted display).
        plt.ion()  # Turn on matplotlib's interactive mode.

        # Calculate values for ROC curve plotting
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)  # Calculate AUC for ROC curve

        # Plot ROC curve
        self.ax.plot(false_positive_rate, true_positive_rate, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))

        # Plot dashed line representing random guessing
        self.ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')

        # Set labels for x and y axes
        self.ax.set_xlabel('False Positive Rate')  # Set x-axis label as "False Positive Rate"
        self.ax.set_ylabel('True Positive Rate')  # Set y-axis label as "True Positive Rate"

        # Set title for the plot
        self.fig.suptitle('Receiver Operating Characteristic (ROC)',
                          size=8)  # Set title for the plot as "Receiver Operating Characteristic (ROC)" with font size 8.

        # Add legend to the plot
        self.fig.legend(loc='lower right')  # Set legend position at lower right corner.


class show_importance(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots()  # Create a figure and axes object.
        super().__init__(self.fig)  # Initialize the FigureCanvas base class with the figure object.
        plt.close()  # Close the figure object (to prevent unwanted display).
        plt.ion()  # Turn on matplotlib's interactive mode.

        # Get labels and importance values of features
        labels = feature_imp.index  # Get labels from feature_imp index.
        values = feature_imp  # Get values from feature_imp.

        # Set x-axis and width of bars
        x = np.arange(len(labels))  # Create an array of numbers from 0 to number of labels.
        width = 0.5  # Set width of bars to 0.5.

        # Plot horizontal bar chart
        rects = self.ax.barh(x, values, width,
                             label="importance")  # Plot horizontal bar chart with labels and feature importance values.

        # Add labels to the bars
        self.ax.bar_label(rects)  # Add value labels to the bars.

        # Set labels for y-axis
        self.ax.set_yticks(x)  # Set y-axis ticks positions with values from array x.
        self.ax.set_yticklabels(['AGE', 'INCOME', 'LOANAMOUNT', 'CREDITSCORE', 'MONTHSEMPLOYED', 'NUMCREDITLINES', "INTERESTRATE",
             'LOANTERM','DTIRATIO', 'EDUCATION', 'EMPLOYMENTTYPE', 'MARITALSTATUS',
             'HASMORTGAGE', 'HASDEPENDENTS','LOANPURPOSE', 'HASCOSIGNER',])  # Set labels for y-axis ticks.

        # Set title for the plot
        self.fig.suptitle("Feature Importance Score",
                          size=8)  # Set title for the plot as "Feature Importance Score" with font size 8.



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Set up the main window
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1510, 890)

        # Create a ScrollArea and make it resizable
        self.scrollArea = QtWidgets.QScrollArea(MainWindow)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        # Create the central widget
        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setGeometry(QtCore.QRect(0, 0, 1500, 1500))
        self.centralwidget.setObjectName("centralwidget")

        # Create a GroupBox for input controls
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(1010, 20, 231, 791))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")

        # Create and set up labels, checkboxes, spinboxes for input factors
        self.label = QtWidgets.QLabel(parent=self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 30, 55, 16))
        self.label.setObjectName("label")
        self.income = QtWidgets.QSpinBox(parent=self.groupBox)
        self.income.setGeometry(QtCore.QRect(100, 30, 100, 20))
        self.income.setObjectName("income")


        self.label_2 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 70, 41, 21))
        self.label_2.setObjectName("label_2")
        self.age = QtWidgets.QSpinBox(parent=self.groupBox)
        self.age.setGeometry(QtCore.QRect(100, 70, 42, 22))
        self.age.setObjectName("age")

        self.label_3 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 110, 71, 16))
        self.label_3.setObjectName("label_3")
        self.loanamount = QtWidgets.QSpinBox(parent=self.groupBox)
        self.loanamount.setGeometry(QtCore.QRect(100, 110, 100, 16))
        self.loanamount.setObjectName("loanamount")


        self.label_4 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(10, 150, 91, 16))
        self.label_4.setObjectName("label_4")
        self.creditscore = QtWidgets.QSpinBox(parent=self.groupBox)
        self.creditscore.setGeometry(QtCore.QRect(100, 150, 100, 16))
        self.creditscore.setObjectName("creditscore")


        self.label_5 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(10, 190, 55, 16))
        self.label_5.setObjectName("label_5")
        self.monthsemployed = QtWidgets.QSpinBox(parent=self.groupBox)
        self.monthsemployed.setGeometry(QtCore.QRect(100, 190, 100, 16))
        self.monthsemployed.setObjectName("monthsemployed")


        self.label_6 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(10, 230, 55, 16))
        self.label_6.setObjectName("label_6")
        self.numbercreditlines = QtWidgets.QSpinBox(parent=self.groupBox)
        self.numbercreditlines.setGeometry(QtCore.QRect(100, 230, 100, 16))
        self.numbercreditlines.setObjectName("numbercreditlines")

        self.label_7 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(10, 270, 91, 16))
        self.label_7.setObjectName("label_7")
        self.interestrate = QtWidgets.QSpinBox(parent=self.groupBox)
        self.interestrate.setGeometry(QtCore.QRect(100, 270, 100, 16))
        self.interestrate.setObjectName("interestrate")

        self.label_8 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(10, 310, 55, 16))
        self.label_8.setObjectName("label_8")
        self.loanterm = QtWidgets.QSpinBox(parent=self.groupBox)
        self.loanterm.setGeometry(QtCore.QRect(100, 310, 100, 16))
        self.loanterm.setObjectName("loanterm")

        self.label_9 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(10, 350, 55, 16))
        self.label_9.setObjectName("label_9")
        self.dtiratio = QtWidgets.QSpinBox(parent=self.groupBox)
        self.dtiratio.setGeometry(QtCore.QRect(100, 350, 100, 16))
        self.dtiratio.setObjectName("dtiratio")

        self.label_10 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(10, 390, 81, 16))
        self.label_10.setObjectName("label_10")

        self.educationComboBox = QtWidgets.QComboBox(parent=self.groupBox)
        self.educationComboBox.setGeometry(QtCore.QRect(100, 390, 150, 22))
        self.educationComboBox.setObjectName("educationComboBox")
        self.educationComboBox.addItems(["Bachelor's", "High School", "Master's", "PhD"])


        self.label_12 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_12.setGeometry(QtCore.QRect(10, 430, 71, 16))
        self.label_12.setObjectName("label_12")
        self.employmenttype = QtWidgets.QSpinBox(parent=self.groupBox)
        self.employmenttype.setGeometry(QtCore.QRect(100, 430, 100, 16))
        self.employmenttype.setObjectName("employmenttype")


        self.label_11 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_11.setGeometry(QtCore.QRect(10, 470, 71, 16))
        self.label_11.setObjectName("label_11")
        self.maritalstatus = QtWidgets.QSpinBox(parent=self.groupBox)
        self.maritalstatus.setGeometry(QtCore.QRect(100, 470, 100, 16))
        self.maritalstatus.setObjectName("maritalstatus")

        self.label_13 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_13.setGeometry(QtCore.QRect(10, 510, 55, 16))
        self.label_13.setObjectName("label_13")
        self.hasmortgage = QtWidgets.QSpinBox(parent=self.groupBox)
        self.hasmortgage.setGeometry(QtCore.QRect(100, 510, 100, 16))
        self.hasmortgage.setObjectName("hasmortgage")

        self.label_14 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_14.setGeometry(QtCore.QRect(10, 550, 70, 16))
        self.label_14.setObjectName("label_14")
        self.hasdependents= QtWidgets.QSpinBox(parent=self.groupBox)
        self.hasdependents.setGeometry(QtCore.QRect(100, 550, 100, 16))
        self.hasdependents.setObjectName("hasdependents")


        self.label_15 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_15.setGeometry(QtCore.QRect(10, 590, 61, 16))
        self.label_15.setObjectName("label_15")
        self.loanpurposeComboBox = QtWidgets.QComboBox(parent=self.groupBox)
        self.loanpurposeComboBox.setGeometry(QtCore.QRect(100, 590, 100, 20))
        self.loanpurposeComboBox.setObjectName("loanpurposeComboBox")
        self.loanpurposeComboBox.addItems(["Auto=0", "Business = 1", "Education=2", "Home =3", "Other=4"])

        self.ketqua = QtWidgets.QTextEdit(parent=self.groupBox)
        self.ketqua.setGeometry(QtCore.QRect(20, 690, 191, 71))
        self.ketqua.setObjectName("ketqua")

        self.chuandoan = QtWidgets.QPushButton(parent=self.groupBox)
        self.chuandoan.setGeometry(QtCore.QRect(70, 600, 93, 28))
        self.chuandoan.setObjectName("chuandoan")
        self.label_16 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_16.setGeometry(QtCore.QRect(10, 660, 55, 21))
        self.label_16.setObjectName("label_16")
        self.nhandulieu = QtWidgets.QPushButton(parent=self.centralwidget)
        self.nhandulieu.setGeometry(QtCore.QRect(20, 10, 141, 21))
        self.nhandulieu.setObjectName("nhandulieu")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 30, 800, 171))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.bangdulieu = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.bangdulieu.setContentsMargins(0, 0, 0, 0)
        self.bangdulieu.setObjectName("bangdulieu")
        self.huanluyen = QtWidgets.QPushButton(parent=self.centralwidget)
        self.huanluyen.setGeometry(QtCore.QRect(20, 207, 261, 21))
        self.huanluyen.setObjectName("huanluyen")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(20, 230, 800, 211))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.cay = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.cay.setContentsMargins(0, 0, 0, 0)
        self.cay.setObjectName("cay")
        self.label_17 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(490, 210, 211, 16))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(30, 450, 391, 16))
        self.label_18.setObjectName("label_18")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(20, 450, 350, 341))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.dochinhxac = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.dochinhxac.setContentsMargins(20, 20, 20, 20)
        self.dochinhxac.setObjectName("dochinhxac")
        self.label_19 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(590, 450, 141, 16))
        self.label_19.setObjectName("label_19")
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(parent = self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(400, 450, 350, 341))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.hieusuat = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.hieusuat.setContentsMargins(20, 20, 20, 20)
        self.hieusuat.setObjectName("hieusuat")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(parent = self.centralwidget)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(800, 450, 350, 341))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.doquantrong = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.doquantrong.setContentsMargins(20, 20, 20, 20)
        self.doquantrong.setObjectName("doquantrong")
        self.label_20 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(790, 450, 201, 16))
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(870, 210, 111, 21))
        self.label_21.setObjectName("label_21")
        self.recallScore = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.recallScore.setGeometry(QtCore.QRect(850, 400, 141, 31))
        self.recallScore.setObjectName("recallScore")
        self.label_22 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(880, 230, 81, 16))
        self.label_22.setObjectName("label_22")
        self.precisionScore = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.precisionScore.setGeometry(QtCore.QRect(850, 340, 141, 31))
        self.precisionScore.setObjectName("precisionScore")
        self.accuracyScore = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.accuracyScore.setGeometry(QtCore.QRect(850, 280, 141, 31))
        self.accuracyScore.setObjectName("accuracyScore")
        self.label_23 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(880, 260, 101, 16))
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(880, 320, 101, 16))
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(880, 380, 81, 16))
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_26.setGeometry(QtCore.QRect(560, 10, 161, 16))
        self.label_26.setObjectName("label_26")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1100, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.nhandulieu.clicked.connect(self.nhan)
        self.huanluyen.clicked.connect(self.train)
        self.chuandoan.clicked.connect(self.chandoan)

        self.scrollArea.setWidget(self.centralwidget)
        MainWindow.setCentralWidget(self.scrollArea)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1100, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Nhóm 8:RandomForest"))
        self.groupBox.setTitle(_translate("MainWindow", "Dự đoán rủi ro tài chính"))

        self.label.setText(_translate("MainWindow", "INCOME:"))
        self.income.setMaximum(1000000)
        self.income.setSpecialValueText("1000000")

        self.label_2.setText(_translate("MainWindow", "Age:"))
        self.age.setSpecialValueText("100")

        self.label_3.setText(_translate("MainWindow", "LOANAMOUNT:"))
        self.loanamount.setMaximum(1000000)
        self.loanamount.setSpecialValueText("1000000")

        self.label_4.setText(_translate("MainWindow", "CREDITSCORE:"))
        self.creditscore.setMaximum(1000000)
        self.creditscore.setSpecialValueText("1000000")

        self.label_5.setText(_translate("MainWindow", "MONTHSEMPLOYED:"))
        self.monthsemployed.setMaximum(1000000)
        self.monthsemployed.setSpecialValueText("1000000")

        self.label_6.setText(_translate("MainWindow", "NUMCREDITLINES:"))
        self.numbercreditlines.setMaximum(3)
        self.numbercreditlines.setSpecialValueText("3")

        self.label_7.setText(_translate("MainWindow", "INTERESTRATE:"))
        self.interestrate.setMaximum(1000000)
        self.interestrate.setSpecialValueText("10000")


        self.label_8.setText(_translate("MainWindow", "LOANTERM:"))
        self.loanterm.setMaximum(1000000)
        self.loanterm.setSpecialValueText("10000")


        self.label_9.setText(_translate("MainWindow", "DTIRATIO:"))
        self.dtiratio.setMaximum(1000000)
        self.dtiratio.setSpecialValueText("10000")

        self.label_10.setText(_translate("MainWindow", "EDUCATION:"))
        self.educationComboBox.setCurrentText("Bachelor's")

        self.label_12.setText(_translate("MainWindow", "EMPLOYMENTTYPE:"))
        self.employmenttype.setMaximum(1000000)
        self.employmenttype.setSpecialValueText("10000")


        self.label_11.setText(_translate("MainWindow", "MARITALSTATUS:"))
        self.maritalstatus.setMaximum(1000000)
        self.maritalstatus.setSpecialValueText("")

        self.label_13.setText(_translate("MainWindow", "HASMORTGAGE:"))
        self.hasmortgage.setMaximum(1000000)
        self.hasmortgage.setSpecialValueText("")

        self.label_14.setText(_translate("MainWindow", "HASDEPENDENTS:"))
        self.hasdependents.setMaximum(1000000)
        self.hasdependents.setSpecialValueText("100000")

        self.label_15.setText(_translate("MainWindow", "LOANPURPOSE:"))
        self.loanpurposeComboBox.setCurrentText("Auto=0")


        self.chuandoan.setText(_translate("MainWindow", "Chẩn đoán"))

        self.label_16.setText(_translate("MainWindow", "Kết quả:"))

        self.nhandulieu.setText(_translate("MainWindow", "Nhận dữ liệu đầu vào"))

        self.huanluyen.setText(_translate("MainWindow", "Chẩn đoán tập dữ liệu bằng RandomForest"))
        self.label_17.setText(_translate("MainWindow", "4 cây quyết định đầu tiên được tạo:"))
        self.label_18.setText(
            _translate("MainWindow", "Độ chính xác của tập dữ liệu huấn luyện so với tập dữ liệu thử nghiệm"))
        self.label_19.setText(_translate("MainWindow", "Hiệu suất của mô hình:"))
        self.label_20.setText(_translate("MainWindow", "Độ quan trọng của các thuộc tính:"))
        self.label_21.setText(_translate("MainWindow", "Điểm độ chính xác"))
        self.label_22.setText(_translate("MainWindow", "của mô hình:"))
        self.label_23.setText(_translate("MainWindow", "Độ chính xác:"))
        self.label_24.setText(_translate("MainWindow", "Điểm chính xác:"))
        self.label_25.setText(_translate("MainWindow", "Điểm thu hôi:"))
        self.label_26.setText(_translate("MainWindow", "Bảng tập dữ liệu nhận vào:"))

    def nhan(self):
        # Tạo một QTableWidget để hiển thị dữ liệu
        table = QTableWidget()

        # Đặt số lượng cột của bảng bằng số lượng cột của DataFrame
        table.setColumnCount(len(df.columns))

        # Đặt số lượng hàng của bảng bằng số lượng hàng của DataFrame
        table.setRowCount(len(df.index))

        # Thêm dữ liệu vào QTableWidget
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                # Đặt giá trị của ô tại vị trí hàng i và cột j trong bảng
                table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))

        # Đặt nhãn cho tiêu đề các cột
        table.setHorizontalHeaderLabels(
            ["AGE", "INCOME", "LOANAMOUNT", "CREDITSCORE", "MONTHSEMPLOYED", "NUMCREDITLINES", "INTERESTRATE",
             "LOANTERM","DTIRATIO", "EDUCATION", "EMPLOYMENTTYPE", "MARITALSTATUS",
             "HASMORTGAGE", "HASDEPENDENTS","LOANPURPOSE", "HASCOSIGNER", "DEFAULT"])

        # Thêm bảng vào bố cục (layout) để hiển thị trên giao diện người dùng
        self.bangdulieu.addWidget(table)

    def train(self):
        self.cay.addWidget(show_tree())
        self.dochinhxac.addWidget(show_auc())
        self.hieusuat.addWidget(show_roc())
        self.doquantrong.addWidget(show_importance())
        a = str(accuracy)
        p = str(precision)
        r = str(recall)
        self.accuracyScore.setPlainText(a)
        self.precisionScore.setPlainText(p)
        self.recallScore.setPlainText(r)

    def chandoan(self):
        try:
            loanpurposei = self.loanpurposeComboBox.currentIndex()
            educationi = self.educationComboBox.currentIndex()

            def get_positive_int_value(text):
                value = int(text)
                if value < 0:
                    raise ValueError("Value cannot be negative")
                return value

            # Lấy giá trị trực tiếp từ các trường đầu vào và chuyển đổi chúng thành số nguyên
            hasdependentsi = get_positive_int_value(self.hasdependents.text())
            employmenttypei = get_positive_int_value(self.employmenttype.text())
            dtiratioi = get_positive_int_value(self.dtiratio.text())
            loantermi = get_positive_int_value(self.loanterm.text())
            interestratei = get_positive_int_value(self.interestrate.text())
            monthsemployedi = get_positive_int_value(self.monthsemployed.text())
            creditscorei = get_positive_int_value(self.creditscore.text())
            loanamounti = get_positive_int_value(self.loanamount.text())
            incomei = get_positive_int_value(self.income.text())
            agei = get_positive_int_value(self.age.text())
            numbercreditlinesi = get_positive_int_value(self.numbercreditlines.text())
            maritalstatusi = get_positive_int_value(self.maritalstatus.text())
            hasmortgagei = get_positive_int_value(self.hasmortgage.text())

            chandoann = ''
            # Đảm bảo clf đã được định nghĩa và huấn luyện
            try:
                predict = clf.predict([[incomei, agei, loanamounti, creditscorei, monthsemployedi,
                                        numbercreditlinesi, interestratei, loantermi, dtiratioi,
                                        educationi, employmenttypei, maritalstatusi, hasmortgagei,
                                        hasdependentsi, loanpurposei]])
                if predict == [1]:
                    chandoann = "Dự đoán : \n Người này rủi ro tài chính"
                else:
                    chandoann = "Dự đoán : \n Người này không rủi ro tài chính"
            except NameError:
                chandoann = "Error: Model clf is not defined or trained."

            self.ketqua.setPlainText(chandoann)

        except ValueError as e:
            self.ketqua.setPlainText(
                f"Error: {str(e)}. Please ensure all input fields are filled with valid non-negative numbers.")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import  QTableWidget, QTableWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###Bươc thu thập dữ liệu đầu vào
df=pd.read_csv("D:\TriTueNhanTao\LungCancer\survey lung cancer.csv")
# print("\nNhận dữ liệu đầu vào:")
# print(df)
# print("Mô tả thống kê dữ liệu đầu vào:")
# print(df.describe())

###Bước: tiền sử lý dữ liệu
### các thuộc tính thuộc loại dữ liệu đối tượng. Vì vậy phải chuyển đổi dữ liệu về dạng số hóa
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['GENDER']=le.fit_transform(df['GENDER'])
df['AGE']=le.fit_transform(df['AGE'])
df['SMOKING']=le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS']=le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY']=le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE']=le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE']=le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE ']=le.fit_transform(df['FATIGUE '])
df['ALLERGY ']=le.fit_transform(df['ALLERGY '])
df['WHEEZING']=le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING']=le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING']=le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH']=le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY']=le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN']=le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
# print("Tập dữ liệu sau khi chuyển đổi :")
# print(df)

###Chia tập dữ liệu thành 2 tệp, tệp dữ liệu huấn luyện và tệp dữ liệu thử nghiệm 
inputs=df.drop('LUNG_CANCER',axis="columns")
target=df['LUNG_CANCER']
###Huấn luyện dữ liệu
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(inputs,target,test_size=0.3,random_state=50)

###Bước: Xây dựng mô hình rừng cây từ tập dữ liệu huấn luyện
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=300,max_features=15,min_samples_split=3,max_depth=16,random_state=0)
###Bước: Huấn luyện mô hình
clf.fit(X_train,Y_train) 
Y_pred = clf.predict(X_test)# Dự đoán nhãn trên tập kiểm tra
# print("Dự đoán với mẫu dữ liệu đầu vào :")
# print(Y_pred)

###Mức độ quan trọng của các thuộc tính
inputsss=df.drop('LUNG_CANCER',axis="columns")
feature_imp = pd.Series(clf.feature_importances_,index=inputsss.columns).sort_values(ascending=False)
# print("\nMức độ quan trọng của các thuộc tính:")
# print(feature_imp)

###Bước: đánh giá mô hình
from sklearn.metrics import accuracy_score , precision_score, recall_score
accuracy = accuracy_score(Y_pred,Y_test)
precision = precision_score(Y_pred,Y_test)
recall = recall_score(Y_pred,Y_test)
# print("\nĐánh giá mô hình :")
# print("Accuracy:", accuracy) độ chính xác
# print("Precision:", precision) độ chính xác dương tính 
# print("Recall:", recall) độ phủ

class show_tree(FigureCanvas):
    def __init__(self):
        self.fig, self.ax=plt.subplots()
        super().__init__(self.fig)
        plt.close()
        plt.ion() 
        self.axes = self.fig.subplots(2,2)
        self.ax.set_frame_on(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for i, ax in enumerate(self.axes.flat):
            trees = clf.estimators_[i]
            _ = tree.plot_tree(trees, feature_names=None, class_names=None, filled=True, ax=ax)
        self.draw()

class show_auc(FigureCanvas):
    def __init__(self):
        self.fig, self.ax=plt.subplots()
        super().__init__(self.fig)
        plt.close()
        plt.ion()
        n_estimators = [1,2,4,8,16,32,64,128,300]
        train_results=[]
        test_results=[]
        for estimator in n_estimators:
            rf=RandomForestClassifier(n_estimators=estimator,n_jobs=-1)
            rf.fit(X_train,Y_train)
            train_pred=rf.predict(X_train)
            false_positive_rate, true_positive_rate,thresholds = roc_curve(Y_train,train_pred)
            roc_auc =auc(false_positive_rate,true_positive_rate)
            train_results.append(roc_auc)
            y_pred=rf.predict(X_test)
            false_positive_rate, true_positive_rate,thresholds = roc_curve(Y_test,y_pred)
            roc_auc =auc(false_positive_rate,true_positive_rate)
            test_results.append(roc_auc)
        line1, =self.ax.plot(n_estimators,train_results,"b",label="Train AUC")
        line2, =self.ax.plot(n_estimators,test_results,"r",label="Test AUC")
        self.fig.legend(handler_map={line1:HandlerLine2D(numpoints=2)})
        self.ax.set_ylabel("auc score")
        self.ax.set_xlabel("n_estimators")
        self.fig.suptitle(' Area Under The Curve (AUC)',size=8)

class show_roc(FigureCanvas):
    def __init__(self):
        self.fig, self.ax=plt.subplots()
        super().__init__(self.fig)
        plt.close()
        plt.ion()        
        false_positive_rate, true_positive_rate,thresholds = roc_curve(Y_test,Y_pred)
        roc_auc =auc(false_positive_rate,true_positive_rate)
        self.ax.plot(false_positive_rate, true_positive_rate, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
        self.ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        self.ax.set_xlabel('False Positive Rate')
        self.ax.set_ylabel('True Positive Rate')
        self.fig.suptitle('Receiver Operating Characteristic (ROC)',size=8)
        self.fig.legend(loc='lower right')

class show_importance(FigureCanvas):
    def __init__(self):
        self.fig, self.ax=plt.subplots()
        super().__init__(self.fig)
        plt.close()
        plt.ion()
        labels=feature_imp.index
        values=feature_imp
        x=np.arange(len(labels))
        width=0.5
        rects=self.ax.barh(x,values,width,label="importance")
        self.ax.bar_label(rects)
        self.ax.set_yticks(x)
        self.ax.set_yticklabels(['AGE', 'PEER_PRESSURE', 'ALLERGY ', 'ALCOHOL CONSUMING', 'FATIGUE ',
       'SWALLOWING DIFFICULTY', 'WHEEZING', 'CHRONIC DISEASE', 'GENDER',
       'CHEST PAIN', 'YELLOW_FINGERS', 'SHORTNESS OF BREATH', 'ANXIETY',
       'COUGHING', 'SMOKING'])
        self.fig.suptitle("Feature Importance Score",size=8)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1580, 882)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(1310, 20, 231, 791))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(parent=self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 30, 55, 16))
        self.label.setObjectName("label")
        self.nam = QtWidgets.QCheckBox(parent=self.groupBox)
        self.nam.setGeometry(QtCore.QRect(100, 30, 51, 20))
        self.nam.setObjectName("nam")
        self.nu = QtWidgets.QCheckBox(parent=self.groupBox)
        self.nu.setGeometry(QtCore.QRect(150, 30, 51, 20))
        self.nu.setObjectName("nu")
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 70, 41, 21))
        self.label_2.setObjectName("label_2")
        self.tuoi = QtWidgets.QSpinBox(parent=self.groupBox)
        self.tuoi.setGeometry(QtCore.QRect(100, 70, 42, 22))
        self.tuoi.setObjectName("tuoi")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 110, 71, 16))
        self.label_3.setObjectName("label_3")
        self.hutthuoc1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.hutthuoc1.setGeometry(QtCore.QRect(100, 110, 41, 16))
        self.hutthuoc1.setObjectName("hutthuoc1")
        self.hutthuoc0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.hutthuoc0.setGeometry(QtCore.QRect(150, 110, 61, 20))
        self.hutthuoc0.setObjectName("hutthuoc0")
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(10, 150, 91, 16))
        self.label_4.setObjectName("label_4")
        self.ngontayvang1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.ngontayvang1.setGeometry(QtCore.QRect(100, 150, 41, 16))
        self.ngontayvang1.setObjectName("ngontayvang1")
        self.ngontayvang0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.ngontayvang0.setGeometry(QtCore.QRect(150, 150, 61, 20))
        self.ngontayvang0.setObjectName("ngontayvang0")
        self.label_5 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(10, 190, 55, 16))
        self.label_5.setObjectName("label_5")
        self.lolang1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.lolang1.setGeometry(QtCore.QRect(100, 190, 41, 16))
        self.lolang1.setObjectName("lolang1")
        self.lolang0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.lolang0.setGeometry(QtCore.QRect(150, 190, 61, 20))
        self.lolang0.setObjectName("lolang0")
        self.benhmantinh1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.benhmantinh1.setGeometry(QtCore.QRect(100, 270, 41, 16))
        self.benhmantinh1.setObjectName("benhmantinh1")
        self.benhmantinh0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.benhmantinh0.setGeometry(QtCore.QRect(150, 270, 61, 20))
        self.benhmantinh0.setObjectName("benhmantinh0")
        self.metmoi1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.metmoi1.setGeometry(QtCore.QRect(100, 310, 41, 16))
        self.metmoi1.setObjectName("metmoi1")
        self.metmoi0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.metmoi0.setGeometry(QtCore.QRect(150, 310, 61, 20))
        self.metmoi0.setObjectName("metmoi0")
        self.diung1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.diung1.setGeometry(QtCore.QRect(100, 350, 41, 16))
        self.diung1.setObjectName("diung1")
        self.diung0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.diung0.setGeometry(QtCore.QRect(150, 350, 61, 20))
        self.diung0.setObjectName("diung0")
        self.thokhokhe1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.thokhokhe1.setGeometry(QtCore.QRect(100, 390, 41, 16))
        self.thokhokhe1.setObjectName("thokhokhe1")
        self.thokhokhe0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.thokhokhe0.setGeometry(QtCore.QRect(150, 390, 61, 20))
        self.thokhokhe0.setObjectName("thokhokhe0")
        self.unguou1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.unguou1.setGeometry(QtCore.QRect(100, 430, 41, 16))
        self.unguou1.setObjectName("unguou1")
        self.unguou0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.unguou0.setGeometry(QtCore.QRect(150, 430, 61, 20))
        self.unguou0.setObjectName("unguou0")
        self.ho1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.ho1.setGeometry(QtCore.QRect(100, 470, 41, 16))
        self.ho1.setObjectName("ho1")
        self.ho0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.ho0.setGeometry(QtCore.QRect(150, 470, 61, 20))
        self.ho0.setObjectName("ho0")
        self.ketqua = QtWidgets.QTextEdit(parent=self.groupBox)
        self.ketqua.setGeometry(QtCore.QRect(20, 690, 191, 71))
        self.ketqua.setObjectName("ketqua")
        self.label_6 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(10, 230, 55, 16))
        self.label_6.setObjectName("label_6")
        self.apluc1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.apluc1.setGeometry(QtCore.QRect(100, 230, 41, 16))
        self.apluc1.setObjectName("apluc1")
        self.apluc0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.apluc0.setGeometry(QtCore.QRect(150, 230, 61, 20))
        self.apluc0.setObjectName("apluc0")
        self.label_7 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(10, 270, 91, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(10, 310, 55, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(10, 350, 55, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(10, 390, 81, 16))
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_12.setGeometry(QtCore.QRect(10, 430, 71, 16))
        self.label_12.setObjectName("label_12")
        self.label_11 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_11.setGeometry(QtCore.QRect(10, 470, 31, 16))
        self.label_11.setObjectName("label_11")
        self.label_13 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_13.setGeometry(QtCore.QRect(10, 510, 55, 16))
        self.label_13.setObjectName("label_13")
        self.khotho1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.khotho1.setGeometry(QtCore.QRect(100, 510, 41, 16))
        self.khotho1.setObjectName("khotho1")
        self.khotho0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.khotho0.setGeometry(QtCore.QRect(150, 510, 61, 20))
        self.khotho0.setObjectName("khotho0")
        self.label_14 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_14.setGeometry(QtCore.QRect(10, 550, 55, 16))
        self.label_14.setObjectName("label_14")
        self.khonuot1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.khonuot1.setGeometry(QtCore.QRect(100, 550, 41, 16))
        self.khonuot1.setObjectName("khonuot1")
        self.khonuot0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.khonuot0.setGeometry(QtCore.QRect(150, 550, 61, 20))
        self.khonuot0.setObjectName("khonuot0")
        self.label_15 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_15.setGeometry(QtCore.QRect(10, 590, 61, 16))
        self.label_15.setObjectName("label_15")
        self.daunguc0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.daunguc0.setGeometry(QtCore.QRect(150, 590, 61, 20))
        self.daunguc0.setObjectName("daunguc0")
        self.daunguc1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.daunguc1.setGeometry(QtCore.QRect(100, 590, 41, 16))
        self.daunguc1.setObjectName("daunguc1")
        self.chuandoan = QtWidgets.QPushButton(parent=self.groupBox)
        self.chuandoan.setGeometry(QtCore.QRect(70, 630, 93, 28))
        self.chuandoan.setObjectName("chuandoan")
        self.label_16 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_16.setGeometry(QtCore.QRect(10, 660, 55, 21))
        self.label_16.setObjectName("label_16")
        self.nhandulieu = QtWidgets.QPushButton(parent=self.centralwidget)
        self.nhandulieu.setGeometry(QtCore.QRect(20, 10, 141, 21))
        self.nhandulieu.setObjectName("nhandulieu")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 30, 1271, 171))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.bangdulieu = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.bangdulieu.setContentsMargins(0, 0, 0, 0)
        self.bangdulieu.setObjectName("bangdulieu")
        self.huanluyen = QtWidgets.QPushButton(parent=self.centralwidget)
        self.huanluyen.setGeometry(QtCore.QRect(20, 207, 261, 21))
        self.huanluyen.setObjectName("huanluyen")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(20, 230, 1091, 211))
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
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(20, 470, 411, 341))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.dochinhxac = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.dochinhxac.setContentsMargins(20, 20, 20, 20)
        self.dochinhxac.setObjectName("dochinhxac")
        self.label_19 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(590, 450, 141, 16))
        self.label_19.setObjectName("label_19")
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(450, 470, 411, 341))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.hieusuat = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.hieusuat.setContentsMargins(20, 20, 20, 20)
        self.hieusuat.setObjectName("hieusuat")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(880, 470, 411, 341))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.doquantrong = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.doquantrong.setContentsMargins(20, 20, 20, 20)
        self.doquantrong.setObjectName("doquantrong")
        self.label_20 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(990, 450, 201, 16))
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(1170, 210, 111, 21))
        self.label_21.setObjectName("label_21")
        self.recallScore = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.recallScore.setGeometry(QtCore.QRect(1150, 400, 141, 31))
        self.recallScore.setObjectName("recallScore")
        self.label_22 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(1180, 230, 81, 16))
        self.label_22.setObjectName("label_22")
        self.precisionScore = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.precisionScore.setGeometry(QtCore.QRect(1150, 340, 141, 31))
        self.precisionScore.setObjectName("precisionScore")
        self.accuracyScore = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.accuracyScore.setGeometry(QtCore.QRect(1150, 280, 141, 31))
        self.accuracyScore.setObjectName("accuracyScore")
        self.label_23 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(1180, 260, 101, 16))
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(1180, 320, 101, 16))
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(1180, 380, 81, 16))
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_26.setGeometry(QtCore.QRect(560, 10, 161, 16))
        self.label_26.setObjectName("label_26")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1580, 26))
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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Nhom8:RandomForest_LungCancer"))
        self.groupBox.setTitle(_translate("MainWindow", "Chẩn đoán bệnh ung thư phổi"))
        self.label.setText(_translate("MainWindow", "Giới tính:"))
        self.nam.setText(_translate("MainWindow", "Nam"))
        self.nu.setText(_translate("MainWindow", "Nữ"))
        self.label_2.setText(_translate("MainWindow", "Tuổi:"))
        self.label_3.setText(_translate("MainWindow", "Hút thuốc:"))
        self.hutthuoc1.setText(_translate("MainWindow", "Có"))
        self.hutthuoc0.setText(_translate("MainWindow", "Không"))
        self.label_4.setText(_translate("MainWindow", "Ngón tay vàng:"))
        self.ngontayvang1.setText(_translate("MainWindow", "Có"))
        self.ngontayvang0.setText(_translate("MainWindow", "Không"))
        self.label_5.setText(_translate("MainWindow", "Lo lắng:"))
        self.lolang1.setText(_translate("MainWindow", "Có"))
        self.lolang0.setText(_translate("MainWindow", "Không"))
        self.benhmantinh1.setText(_translate("MainWindow", "Có"))
        self.benhmantinh0.setText(_translate("MainWindow", "Không"))
        self.metmoi1.setText(_translate("MainWindow", "Có"))
        self.metmoi0.setText(_translate("MainWindow", "Không"))
        self.diung1.setText(_translate("MainWindow", "Có"))
        self.diung0.setText(_translate("MainWindow", "Không"))
        self.thokhokhe1.setText(_translate("MainWindow", "Có"))
        self.thokhokhe0.setText(_translate("MainWindow", "Không"))
        self.unguou1.setText(_translate("MainWindow", "Có"))
        self.unguou0.setText(_translate("MainWindow", "Không"))
        self.ho1.setText(_translate("MainWindow", "Có"))
        self.ho0.setText(_translate("MainWindow", "Không"))
        self.label_6.setText(_translate("MainWindow", "Áp lực:"))
        self.apluc1.setText(_translate("MainWindow", "Có"))
        self.apluc0.setText(_translate("MainWindow", "Không"))
        self.label_7.setText(_translate("MainWindow", "Bệnh mãn tính:"))
        self.label_8.setText(_translate("MainWindow", "Mệt mỏi:"))
        self.label_9.setText(_translate("MainWindow", "Dị ứng:"))
        self.label_10.setText(_translate("MainWindow", "Thở khò khè:"))
        self.label_12.setText(_translate("MainWindow", "Uống rượu:"))
        self.label_11.setText(_translate("MainWindow", "Ho:"))
        self.label_13.setText(_translate("MainWindow", "Khó thở:"))
        self.khotho1.setText(_translate("MainWindow", "Có"))
        self.khotho0.setText(_translate("MainWindow", "Không"))
        self.label_14.setText(_translate("MainWindow", "Khó nuốt:"))
        self.khonuot1.setText(_translate("MainWindow", "Có"))
        self.khonuot0.setText(_translate("MainWindow", "Không"))
        self.label_15.setText(_translate("MainWindow", "Đau ngực:"))
        self.daunguc0.setText(_translate("MainWindow", "Không"))
        self.daunguc1.setText(_translate("MainWindow", "Có"))
        self.chuandoan.setText(_translate("MainWindow", "Chẩn đoán"))
        self.label_16.setText(_translate("MainWindow", "Kết quả:"))
        self.nhandulieu.setText(_translate("MainWindow", "Nhận dữ liệu đầu vào"))
        self.huanluyen.setText(_translate("MainWindow", "Chẩn đoán tập dữ liệu bằng RandomForest"))
        self.label_17.setText(_translate("MainWindow", "4 cây quyết định đầu tiên được tạo:"))
        self.label_18.setText(_translate("MainWindow", "Độ chính xác của tập dữ liệu huấn luyện so với tập dữ liệu thử nghiệm"))
        self.label_19.setText(_translate("MainWindow", "Hiệu suất của mô hình:"))
        self.label_20.setText(_translate("MainWindow", "Độ quan trọng của các thuộc tính:"))
        self.label_21.setText(_translate("MainWindow", "Điểm độ chính xác"))
        self.label_22.setText(_translate("MainWindow", "của mô hình:"))
        self.label_23.setText(_translate("MainWindow", "accuracy_score:"))
        self.label_24.setText(_translate("MainWindow", "precision_score:"))
        self.label_25.setText(_translate("MainWindow", "recall_score:"))
        self.label_26.setText(_translate("MainWindow", "Bảng tập dữ liệu nhận vào:"))
        self.tuoi.setSpecialValueText("43")

    def nhan(self): 
        table = QTableWidget()
        table.setColumnCount(len(df.columns))
        table.setRowCount(len(df.index))
        # Thêm dữ liệu vào QTableWidget
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))    
        table.setHorizontalHeaderLabels(["GENDER","AGE", "SMOKING", "YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC DISEASE","FATIGUE" ,"ALLERGY" ,"WHEEZING","ALCOHOL CONSUMING","COUGHING","SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN","LUNG_CANCER"])
        self.bangdulieu.addWidget(table)

    def train(self):
        self.cay.addWidget(show_tree())
        self.dochinhxac.addWidget(show_auc()) 
        self.hieusuat.addWidget(show_roc()) 
        self.doquantrong.addWidget(show_importance())
        a=str(accuracy)
        p=str(precision)
        r=str(recall)
        self.accuracyScore.setPlainText(a)
        self.precisionScore.setPlainText(p)
        self.recallScore.setPlainText(r)
     
    def chandoan(self):
        gioitinh=0;tuoii=0;hutthuoc=0;ngontayvang=0;lolang=0;apluc=0;benhmantinh=0;metmoi=0;diung=0;thokhokhe=0;uongruou=0;ho=0;khotho=0;khonuot=0;daunguc=0
        if self.nam.isChecked()==True:         gioitinh=1
        if self.nu.isChecked()==True:          gioitinh=0
        if self.hutthuoc1.isChecked()==True:   hutthuoc=1
        if self.hutthuoc0.isChecked()==True:   hutthuoc=0
        if self.ngontayvang1.isChecked()==True:ngontayvang=1
        if self.ngontayvang0.isChecked()==True:ngontayvang=0
        if self.lolang1.isChecked()==True:     lolang=1
        if self.lolang0.isChecked()==True:     lolang=0
        if self.apluc1.isChecked()==True:      apluc=1
        if self.apluc0.isChecked()==True:      apluc=0    
        if self.benhmantinh1.isChecked()==True:benhmantinh=1
        if self.benhmantinh0.isChecked()==True:benhmantinh=0
        if self.metmoi1.isChecked()==True:     metmoi=1
        if self.metmoi0.isChecked()==True:     metmoi=0
        if self.diung1.isChecked()==True:      diung=1
        if self.diung0.isChecked()==True:      diung=0
        if self.thokhokhe1.isChecked()==True:  thokhokhe=1
        if self.thokhokhe0.isChecked()==True:  thokhokhe=0
        if self.unguou1.isChecked()==True:     uongruou=1
        if self.unguou0.isChecked()==True:     uongruou=0
        if self.ho1.isChecked()==True:         ho=1
        if self.ho0.isChecked()==True:         ho=0
        if self.khotho1.isChecked()==True:     khotho=1
        if self.khotho0.isChecked()==True:     khotho=0
        if self.khonuot1.isChecked()==True:    khonuot=1
        if self.khonuot0.isChecked()==True:    khonuot=0
        if self.daunguc1.isChecked()==True:    daunguc=1
        if self.daunguc0.isChecked()==True:    daunguc=0
        tuoii=int(self.tuoi.text())-43
        if tuoii<0 : tuoii=0
        chandoann=''
        predict=clf.predict([[gioitinh,tuoii,hutthuoc,ngontayvang,lolang,apluc,benhmantinh,metmoi,diung,thokhokhe,uongruou,ho,khotho,khonuot,daunguc]])
        if(predict==[1]):  chandoann="Kết quả chẩn đoán : \n Người này có bị ung thư phổi"
        else: chandoann="Kết quả chẩn đoán : \n Người này không bị ung thư phổi"
        self.ketqua.setPlainText(chandoann)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())

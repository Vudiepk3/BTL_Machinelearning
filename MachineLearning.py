from PyQt6 import QtCore, QtGui, QtWidgets  # Import các module cần thiết từ PyQt6
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem  # Import các widget cụ thể từ PyQt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Import FigureCanvas cho việc hiển thị đồ thị trên giao diện
from sklearn.metrics import roc_curve, auc  # Import các hàm đánh giá mô hình từ sklearn
from matplotlib.legend_handler import HandlerLine2D  # Import HandlerLine2D để xử lý legend trong biểu đồ
from sklearn import tree  # Import tree từ sklearn để vẽ cây quyết định
import pandas as pd  # Import pandas để xử lý dữ liệu
import numpy as np  # Import numpy để xử lý các mảng số học
import matplotlib.pyplot as plt  # Import matplotlib để vẽ biểu đồ

###Bươc thu thập dữ liệu đầu vào
df = pd.read_csv("D:\MachineLearning\LungCancer\data.csv")
# print("\nNhận dữ liệu đầu vào:")
# print(df)
# print("Mô tả thống kê dữ liệu đầu vào:")
# print(df.describe())

###Bước: tiền sử lý dữ liệu
### các thuộc tính thuộc loại dữ liệu đối tượng. Vì vậy phải chuyển đổi dữ liệu về dạng số hóa
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['JOB'] = le.fit_transform(df['JOB'])
df['AGE'] = le.fit_transform(df['AGE'])
df['MARITAL'] = le.fit_transform(df['MARITAL'])
df['EDUCATION'] = le.fit_transform(df['EDUCATION'])
df['DEFAULT'] = le.fit_transform(df['DEFAULT'])
df['BALANCE'] = le.fit_transform(df['BALANCE'])
df['HOUSING'] = le.fit_transform(df['HOUSING'])
df['LOAN'] = le.fit_transform(df['LOAN'])
df['CONTACT'] = le.fit_transform(df['CONTACT'])
df['DAY'] = le.fit_transform(df['DAY'])
df['MONTH'] = le.fit_transform(df['MONTH'])
df['DURATION'] = le.fit_transform(df['DURATION'])
df['CAMPAIGN'] = le.fit_transform(df['CAMPAIGN'])
df['PREVIOUS'] = le.fit_transform(df['PREVIOUS'])
df['DEPOSIT'] = le.fit_transform(df['DEPOSIT'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
# print("Tập dữ liệu sau khi chuyển đổi :")
# print(df)

###Chia tập dữ liệu thành 2 tệp, tệp dữ liệu huấn luyện và tệp dữ liệu thử nghiệm
inputs = df.drop('LUNG_CANCER', axis="columns")
target = df['LUNG_CANCER']
###Huấn luyện dữ liệu
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(inputs, target, test_size=0.3, random_state=50)

###Bước: Xây dựng mô hình rừng cây từ tập dữ liệu huấn luyện
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=300, max_features=15, min_samples_split=3, max_depth=16, random_state=0)
###Bước: Huấn luyện mô hình
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)  # Dự đoán nhãn trên tập kiểm tra
# print("Dự đoán với mẫu dữ liệu đầu vào :")
# print(Y_pred)

###Mức độ quan trọng của các thuộc tính
inputsss = df.drop('LUNG_CANCER', axis="columns")
feature_imp = pd.Series(clf.feature_importances_, index=inputsss.columns).sort_values(ascending=False)
# print("\nMức độ quan trọng của các thuộc tính:")
# print(feature_imp)

###Bước: đánh giá mô hình
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = accuracy_score(Y_pred, Y_test)
precision = precision_score(Y_pred, Y_test)
recall = recall_score(Y_pred, Y_test)


# print("\nĐánh giá mô hình :")
# print("Accuracy:", accuracy) độ chính xác
# print("Precision:", precision) độ chính xác dương tính
# print("Recall:", recall) độ phủ
class show_tree(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots()  # Tạo một đối tượng figure và axes.
        super().__init__(self.fig)  # Khởi tạo lớp cha FigureCanvas với đối tượng figure.
        plt.close()  # Đóng đối tượng figure (tránh hiển thị không mong muốn).
        plt.ion()  # Bật chế độ tương tác của matplotlib.
        self.axes = self.fig.subplots(2, 2)  # Tạo một lưới subplot 2x2.
        self.ax.set_frame_on(False)  # Loại bỏ khung của axes chính.
        self.ax.set_xticks([])  # Ẩn các nhãn của trục x.
        self.ax.set_yticks([])  # Ẩn các nhãn của trục y.
        for i, ax in enumerate(self.axes.flat):  # Lặp qua từng subplot (axes con) trong lưới 2x2.
            trees = clf.estimators_[i]  # Lấy cây quyết định thứ i từ mô hình Random Forest.
            _ = tree.plot_tree(trees, feature_names=None, class_names=None, filled=True, ax=ax)  # Vẽ cây quyết định lên subplot tương ứng.
        self.draw()  # Vẽ lại toàn bộ figure để hiển thị các cây quyết định.


class show_auc(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots()  # Tạo một đối tượng figure và axes.
        super().__init__(self.fig)  # Khởi tạo lớp cha FigureCanvas với đối tượng figure.
        plt.close()  # Đóng đối tượng figure (tránh hiển thị không mong muốn).
        plt.ion()  # Bật chế độ tương tác của matplotlib.

        # Các giá trị khác nhau của n_estimators sẽ được thử nghiệm
        n_estimators = [1, 2, 4, 8, 16, 32, 64, 128, 300]

        train_results = []  # Danh sách để lưu trữ kết quả AUC trên tập huấn luyện.
        test_results = []  # Danh sách để lưu trữ kết quả AUC trên tập kiểm tra.

        # Lặp qua từng giá trị của n_estimators để huấn luyện mô hình và tính toán AUC.
        for estimator in n_estimators:
            rf = RandomForestClassifier(n_estimators=estimator,
                                        n_jobs=-1)  # Khởi tạo mô hình RandomForest với số lượng cây tương ứng.
            rf.fit(X_train, Y_train)  # Huấn luyện mô hình trên tập huấn luyện.

            train_pred = rf.predict(X_train)  # Dự đoán trên tập huấn luyện.
            false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train,
                                                                            train_pred)  # Tính FPR và TPR cho ROC curve trên tập huấn luyện.
            roc_auc = auc(false_positive_rate, true_positive_rate)  # Tính AUC cho ROC curve trên tập huấn luyện.
            train_results.append(roc_auc)  # Lưu trữ kết quả AUC của tập huấn luyện.

            y_pred = rf.predict(X_test)  # Dự đoán trên tập kiểm tra.
            false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test,
                                                                            y_pred)  # Tính FPR và TPR cho ROC curve trên tập kiểm tra.
            roc_auc = auc(false_positive_rate, true_positive_rate)  # Tính AUC cho ROC curve trên tập kiểm tra.
            test_results.append(roc_auc)  # Lưu trữ kết quả AUC của tập kiểm tra.

        # Vẽ đồ thị AUC cho tập huấn luyện và tập kiểm tra.
        line1, = self.ax.plot(n_estimators, train_results, "b",
                              label="Train AUC")  # Vẽ đường biểu diễn AUC của tập huấn luyện.
        line2, = self.ax.plot(n_estimators, test_results, "r",
                              label="Test AUC")  # Vẽ đường biểu diễn AUC của tập kiểm tra.

        self.fig.legend(handler_map={line1: HandlerLine2D(numpoints=2)})  # Tạo chú thích (legend) cho đồ thị.
        self.ax.set_ylabel("auc score")  # Đặt nhãn trục y.
        self.ax.set_xlabel("n_estimators")  # Đặt nhãn trục x.
        self.fig.suptitle('Area Under The Curve (AUC)', size=8)  # Đặt tiêu đề cho đồ thị.


class show_roc(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots()  # Tạo một đối tượng figure và axes.
        super().__init__(self.fig)  # Khởi tạo lớp cha FigureCanvas với đối tượng figure.
        plt.close()  # Đóng đối tượng figure (tránh hiển thị không mong muốn).
        plt.ion()  # Bật chế độ tương tác của matplotlib.

        # Tính toán các giá trị để vẽ đường cong ROC
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)  # Tính AUC cho ROC curve

        # Vẽ đường cong ROC
        self.ax.plot(false_positive_rate, true_positive_rate, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))

        # Vẽ đường chấm chấm biểu diễn cho dự đoán ngẫu nhiên
        self.ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')

        # Thiết lập nhãn trục x và y
        self.ax.set_xlabel('False Positive Rate')  # Đặt nhãn trục x là "False Positive Rate"
        self.ax.set_ylabel('True Positive Rate')  # Đặt nhãn trục y là "True Positive Rate"

        # Đặt tiêu đề cho đồ thị
        self.fig.suptitle('Receiver Operating Characteristic (ROC)',
                          size=8)  # Đặt tiêu đề cho đồ thị là "Receiver Operating Characteristic (ROC)" với kích thước chữ là 8.

        # Thêm chú thích vào đồ thị
        self.fig.legend(loc='lower right')  # Đặt vị trí chú thích ở góc dưới bên phải.

class show_importance(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots()  # Tạo một đối tượng figure và axes.
        super().__init__(self.fig)  # Khởi tạo lớp cha FigureCanvas với đối tượng figure.
        plt.close()  # Đóng đối tượng figure (tránh hiển thị không mong muốn).
        plt.ion()  # Bật chế độ tương tác của matplotlib.

        # Lấy nhãn và giá trị mức độ quan trọng của các thuộc tính
        labels = feature_imp.index  # Lấy nhãn từ chỉ số của feature_imp.
        values = feature_imp  # Lấy giá trị từ feature_imp.

        # Thiết lập trục x và độ rộng của các thanh
        x = np.arange(len(labels))  # Tạo một mảng số từ 0 đến số lượng nhãn.
        width = 0.5  # Đặt độ rộng của các thanh là 0.5.

        # Vẽ biểu đồ thanh ngang
        rects = self.ax.barh(x, values, width, label="importance")  # Vẽ biểu đồ thanh ngang với nhãn và giá trị mức độ quan trọng.

        # Gắn nhãn vào các thanh
        self.ax.bar_label(rects)  # Thêm nhãn giá trị vào các thanh.

        # Thiết lập các nhãn cho trục y
        self.ax.set_yticks(x)  # Đặt các vị trí ticks trên trục y bằng các giá trị của mảng x.
        self.ax.set_yticklabels(['AGE', 'BALANCE', 'CONTACT ', 'MONTH', 'LOAN',
                                 'PREVIOUS', 'DAY', 'HOUSING', 'JOB',
                                 'DEPOSIT', 'EDUCATION', 'CAMPAIGN', 'DEFAULT',
                                 'DURATION', 'MARITAL'])  # Đặt nhãn cho các ticks trên trục y.

        # Đặt tiêu đề cho biểu đồ
        self.fig.suptitle("Feature Importance Score", size=8)  # Đặt tiêu đề cho biểu đồ là "Feature Importance Score" với kích thước chữ là 8.


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Thiết lập cửa sổ chính
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1510, 882)

        # Tạo widget trung tâm
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Tạo GroupBox cho các điều khiển nhập liệu
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(1010, 20, 231, 791))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")

        # Tạo và thiết lập các nhãn, checkbox, spinbox cho các yếu tố đầu vào
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
        self.age = QtWidgets.QSpinBox(parent=self.groupBox)
        self.age.setGeometry(QtCore.QRect(100, 70, 42, 22))
        self.age.setObjectName("age")

        self.label_3 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 110, 71, 16))
        self.label_3.setObjectName("label_3")
        self.maritual1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.maritual1.setGeometry(QtCore.QRect(100, 110, 41, 16))
        self.maritual1.setObjectName("maritual1")
        self.maritual0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.maritual0.setGeometry(QtCore.QRect(150, 110, 61, 20))
        self.maritual0.setObjectName("maritual0")

        self.label_4 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(10, 150, 91, 16))
        self.label_4.setObjectName("label_4")
        self.education1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.education1.setGeometry(QtCore.QRect(100, 150, 41, 16))
        self.education1.setObjectName("education1")
        self.education0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.education0.setGeometry(QtCore.QRect(150, 150, 61, 20))
        self.education0.setObjectName("education0")

        self.label_5 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(10, 190, 55, 16))
        self.label_5.setObjectName("label_5")
        self.default1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.default1.setGeometry(QtCore.QRect(100, 190, 41, 16))
        self.default1.setObjectName("default1")
        self.default0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.default0.setGeometry(QtCore.QRect(150, 190, 61, 20))
        self.default0.setObjectName("default0")

        self.label_6 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(10, 230, 55, 16))
        self.label_6.setObjectName("label_6")
        self.blance = QtWidgets.QSpinBox(parent=self.groupBox)
        self.blance.setGeometry(QtCore.QRect(100, 230, 41, 16))
        self.blance.setObjectName("blance")


        self.label_7 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(10, 270, 91, 16))
        self.label_7.setObjectName("label_7")
        self.housing1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.housing1.setGeometry(QtCore.QRect(100, 270, 41, 16))
        self.housing1.setObjectName("housing1")
        self.housing0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.housing0.setGeometry(QtCore.QRect(150, 270, 61, 20))
        self.housing0.setObjectName("housing0")

        self.label_8 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(10, 310, 55, 16))
        self.label_8.setObjectName("label_8")
        self.loan1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.loan1.setGeometry(QtCore.QRect(100, 310, 41, 16))
        self.loan1.setObjectName("loan1")
        self.loan0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.loan0.setGeometry(QtCore.QRect(150, 310, 61, 20))
        self.loan0.setObjectName("loan0")

        self.label_9 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(10, 350, 55, 16))
        self.label_9.setObjectName("label_9")
        self.contact1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.contact1.setGeometry(QtCore.QRect(100, 350, 41, 16))
        self.contact1.setObjectName("contact1")
        self.contact0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.contact0.setGeometry(QtCore.QRect(150, 350, 61, 20))
        self.contact0.setObjectName("contact0")

        self.label_10 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(10, 390, 81, 16))
        self.label_10.setObjectName("label_10")
        self.day = QtWidgets.QSpinBox(parent=self.groupBox)
        self.day.setGeometry(QtCore.QRect(100, 390, 41, 16))
        self.day.setObjectName("day")


        self.label_12 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_12.setGeometry(QtCore.QRect(10, 430, 71, 16))
        self.label_12.setObjectName("label_12")
        self.month1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.month1.setGeometry(QtCore.QRect(100, 430, 41, 16))
        self.month1.setObjectName("month1")
        self.month0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.month0.setGeometry(QtCore.QRect(150, 430, 61, 20))
        self.month0.setObjectName("month0")

        self.label_11 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_11.setGeometry(QtCore.QRect(10, 470, 31, 16))
        self.label_11.setObjectName("label_11")
        self.duration = QtWidgets.QSpinBox(parent=self.groupBox)
        self.duration.setGeometry(QtCore.QRect(100, 470, 41, 16))
        self.duration.setObjectName("duration")


        self.label_13 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_13.setGeometry(QtCore.QRect(10, 510, 55, 16))
        self.label_13.setObjectName("label_13")
        self.campaign = QtWidgets.QSpinBox(parent=self.groupBox)
        self.campaign.setGeometry(QtCore.QRect(100, 510, 41, 16))
        self.campaign.setObjectName("campaign")


        self.label_14 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_14.setGeometry(QtCore.QRect(10, 550, 55, 16))
        self.label_14.setObjectName("label_14")
        self.previous1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.previous1.setGeometry(QtCore.QRect(100, 550, 41, 16))
        self.previous1.setObjectName("previous1")
        self.previous0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.previous0.setGeometry(QtCore.QRect(150, 550, 61, 20))
        self.previous0.setObjectName("previous0")

        self.label_15 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_15.setGeometry(QtCore.QRect(10, 590, 61, 16))
        self.label_15.setObjectName("label_15")
        self.deposit0 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.deposit0.setGeometry(QtCore.QRect(150, 590, 61, 20))
        self.deposit0.setObjectName("deposit0")
        self.deposit1 = QtWidgets.QCheckBox(parent=self.groupBox)
        self.deposit1.setGeometry(QtCore.QRect(100, 590, 41, 16))
        self.deposit1.setObjectName("deposit1")

        self.ketqua = QtWidgets.QTextEdit(parent=self.groupBox)
        self.ketqua.setGeometry(QtCore.QRect(20, 690, 191, 71))
        self.ketqua.setObjectName("ketqua")
        
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
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(400, 450, 350, 341))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.hieusuat = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.hieusuat.setContentsMargins(20, 20, 20, 20)
        self.hieusuat.setObjectName("hieusuat")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(parent=self.centralwidget)
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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Nhóm 8:RandomForest"))
        self.groupBox.setTitle(_translate("MainWindow", "Dự đoán rủi ro tài chính"))
        self.label.setText(_translate("MainWindow", "Job:"))

        self.nam.setText(_translate("MainWindow", "Yes"))
        self.nu.setText(_translate("MainWindow", "No"))

        self.label_2.setText(_translate("MainWindow", "Age:"))
        self.age.setSpecialValueText("100")
        
        self.label_3.setText(_translate("MainWindow", "MARITUAL:"))
        self.maritual1.setText(_translate("MainWindow", "Yes"))
        self.maritual0.setText(_translate("MainWindow", "No"))

        self.label_4.setText(_translate("MainWindow", "EDUCATION:"))
        self.education1.setText(_translate("MainWindow", "Yes"))
        self.education0.setText(_translate("MainWindow", "No"))

        self.label_5.setText(_translate("MainWindow", "DEFAULT:"))
        self.default1.setText(_translate("MainWindow", "Yes"))
        self.default0.setText(_translate("MainWindow", "No"))

        self.label_6.setText(_translate("MainWindow", "BLANCE:"))
        self.blance.setSpecialValueText("100")

        self.label_7.setText(_translate("MainWindow", "HOUSING:"))
        self.housing1.setText(_translate("MainWindow", "Yes"))
        self.housing0.setText(_translate("MainWindow", "No"))

        self.label_8.setText(_translate("MainWindow", "LOAN:"))
        self.loan1.setText(_translate("MainWindow", "Yes"))
        self.loan0.setText(_translate("MainWindow", "No"))

        self.label_9.setText(_translate("MainWindow", "CONTACT:"))
        self.contact1.setText(_translate("MainWindow", "Yes"))
        self.contact0.setText(_translate("MainWindow", "No"))

        self.label_10.setText(_translate("MainWindow", "DAY:"))
        self.day.setSpecialValueText("31")
        
        self.label_12.setText(_translate("MainWindow", "MONTH:"))
        self.month1.setText(_translate("MainWindow", "Yes"))
        self.month0.setText(_translate("MainWindow", "No"))

        self.label_11.setText(_translate("MainWindow", "DURATION:"))
        self.duration.setSpecialValueText("100")

        self.label_13.setText(_translate("MainWindow", "CAMPAIGN:"))
        self.campaign.setSpecialValueText("100")

        self.label_14.setText(_translate("MainWindow", "PREVIOUS:"))
        self.previous1.setText(_translate("MainWindow", "Yes"))
        self.previous0.setText(_translate("MainWindow", "No"))

        self.label_15.setText(_translate("MainWindow", "DEPOSIT:"))
        self.deposit0.setText(_translate("MainWindow", "No"))
        self.deposit1.setText(_translate("MainWindow", "Yes"))

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
        table = QTableWidget()
        table.setColumnCount(len(df.columns))
        table.setRowCount(len(df.index))
        # Thêm dữ liệu vào QTableWidget
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        table.setHorizontalHeaderLabels(
            ["JOB", "AGE", "MARITAL", "EDUCATION", "DEFAULT", "BALANCE", "HOUSING", "FATIGUE",
             "CONTACT", "DAY", "MONTH", "DURATION", "CAMPAIGN", "PREVIOUS",
             "DEPOSIT", "LUNG_CANCER"])
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
        gioitinh = 1 if self.nam.isChecked() else 0
        maritual = 1 if self.maritual1.isChecked() else 0
        education = 1 if self.education1.isChecked() else 0
        default = 1 if self.default1.isChecked() else 0
        housing = 1 if self.housing1.isChecked() else 0
        loan = 1 if self.loan1.isChecked() else 0
        contact = 1 if self.contact1.isChecked() else 0
        month = 1 if self.month1.isChecked() else 0
        previous = 1 if self.previous1.isChecked() else 0
        deposit = 1 if self.deposit1.isChecked() else 0

        def get_valid_value(value):
            return max(int(value) - 100, 0)

        agei = get_valid_value(self.age.text())
        blancei = get_valid_value(self.blance.text())
        dayi = get_valid_value(self.day.text())
        durationi = get_valid_value(self.duration.text())
        campaigni = get_valid_value(self.campaign.text())

        chandoann = ''
        predict = clf.predict([[gioitinh, agei, maritual, education, default, blance, housing, loan, contact,
                                dayi, month, durationi, campaigni, previous, deposit]])
        if (predict == [1]):
            chandoann = "Dự đoán : \n Người này rủi ro tài chính"
        else:
            chandoann = "Dự đoán : \n Người này không rủi ro tài chính"
        self.ketqua.setPlainText(chandoann)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())

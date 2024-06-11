from PyQt6 import QtCore, QtGui, QtWidgets  # Import các module cần thiết từ PyQt6
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem  # Import các widget cụ thể từ PyQt6
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas  # Import FigureCanvas cho việc hiển thị đồ thị trên giao diện
from sklearn.metrics import roc_curve, auc  # Import các hàm đánh giá mô hình từ sklearn
from matplotlib.legend_handler import HandlerLine2D  # Import HandlerLine2D để xử lý legend trong biểu đồ
from sklearn import tree  # Import tree từ sklearn để vẽ cây quyết định
import pandas as pd  # Import pandas để xử lý dữ liệu
import numpy as np  # Import numpy để xử lý các mảng số học
import matplotlib.pyplot as plt  # Import matplotlib để vẽ biểu đồ

###Bươc thu thập dữ liệu đầu vào
df = pd.read_csv("D:\MachineLearning\datatrain.csv")
# print("\nNhận dữ liệu đầu vào:")
# print(df)
# print("Mô tả thống kê dữ liệu đầu vào:")
# print(df.describe())

###Bước: tiền sử lý dữ liệu
### các thuộc tính thuộc loại dữ liệu đối tượng. Vì vậy phải chuyển đổi dữ liệu về dạng số hóa
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
# print("Tập dữ liệu sau khi chuyển đổi :")
# print(df)

###Chia tập dữ liệu thành 2 tệp, tệp dữ liệu huấn luyện và tệp dữ liệu thử nghiệm
inputs = df.drop('DEFAULT', axis="columns")
target = df['DEFAULT']
###Huấn luyện dữ liệu
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(inputs, target, test_size=0.3, random_state=50)

###Bước: Xây dựng mô hình rừng cây từ tập dữ liệu huấn luyện
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=2000,  # Số lượng cây trong rừng ngẫu nhiên là 2000.
    max_features=16,  # Số lượng đặc trưng tối đa được xem xét để tách tại mỗi nút trong cây là 16.
    min_samples_split=3,  # Số lượng mẫu tối thiểu cần thiết để tách một nút là 3.
    max_depth=16,  # Độ sâu tối đa của mỗi cây là 16.
    random_state=0  # Hạt giống ngẫu nhiên được sử dụng để khởi tạo bộ sinh số ngẫu nhiên.
)

###Bước: Huấn luyện mô hình
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)  # Dự đoán nhãn trên tập kiểm tra
# print("Dự đoán với mẫu dữ liệu đầu vào :")
# print(Y_pred)

###Mức độ quan trọng của các thuộc tính
inputsss = df.drop('DEFAULT', axis="columns")
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
            _ = tree.plot_tree(trees, feature_names=None, class_names=None, filled=True,
                               ax=ax)  # Vẽ cây quyết định lên subplot tương ứng.
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
        rects = self.ax.barh(x, values, width,
                             label="importance")  # Vẽ biểu đồ thanh ngang với nhãn và giá trị mức độ quan trọng.

        # Gắn nhãn vào các thanh
        self.ax.bar_label(rects)  # Thêm nhãn giá trị vào các thanh.

        # Thiết lập các nhãn cho trục y
        self.ax.set_yticks(x)  # Đặt các vị trí ticks trên trục y bằng các giá trị của mảng x.
        self.ax.set_yticklabels(['AGE', 'INCOME', 'LOANAMOUNT', 'CREDITSCORE', 'MONTHSEMPLOYED', 'NUMCREDITLINES', "INTERESTRATE",
             'LOANTERM','DTIRATIO', 'EDUCATION', 'EMPLOYMENTTYPE', 'MARITALSTATUS',
             'HASMORTGAGE', 'HASDEPENDENTS','LOANPURPOSE', 'HASCOSIGNER',])  # Đặt nhãn cho các ticks trên trục y.

        # Đặt tiêu đề cho biểu đồ
        self.fig.suptitle("Feature Importance Score",
                          size=8)  # Đặt tiêu đề cho biểu đồ là "Feature Importance Score" với kích thước chữ là 8.


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Thiết lập cửa sổ chính
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1510, 890)

        # Tạo ScrollArea và thiết lập nó làm widget trung tâm
        self.scrollArea = QtWidgets.QScrollArea(MainWindow)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        # Tạo widget trung tâm
        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setGeometry(QtCore.QRect(0, 0, 1500, 1500))
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
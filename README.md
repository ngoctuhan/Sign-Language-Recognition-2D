# SignLanguageRecognition2D

* Nội dung 
Thực hiện nhận diện cử chỉ tay áp dụng trong việc nhận diện ngôn ngữ kí hiệu cho người câm điếc. 

* Công nghệ sử dụng 
1. Sử dụng ROI để detect khu vực bàn tay.
2. Thực hiện tiền xử lí hình ảnh: lọc nhiễu Gause, làm mịn, phân ngưỡng tự động Ostu.
3. Đầu ra của quá trình xử lí ảnh là ảnh nhị phận ảnh được đưa qua mô hình VGG 16 để phân loại đó thuộc loại nào trong bảng chữ cái.
4. ánh xạ đầu ra các từ trong bản chữ cái.

* Cấu trúc Project

1. static: các file html, css, js
2. models: foder chưa model đã được trainning trước, tranfer từ mô hình VGG16
3. webstreaming.py file để run project dạng giao diện website.

* Công nghệ sử dụng:
1. Flask backend end website
2. Tensorflow==1.14
3. OpenCV==4.1
4. Numpy==1.16

* Hướng dẫn sử dụng 
1. Run file python và chờ load model và khởi tạo tham số.
2. Truy cập 127.0.0.2:5000 để ra giao diện sử dụng.
3. Chờ một vài s để hệ thống capture lại nền phục vụ quá trình xóa nền.
4. Thực hiện thao tác nhận dạng tay trong khu vực đã được ROI sẵn.
5. Nhân Predict để dự đoán từ được nối từ các chữ cái nhận được.

Các chữ cái được nhận liên tiếp trong 2s xem như nhận đúng chữ đó.

Link demo: https://www.youtube.com/watch?v=DvuglmDpWIY

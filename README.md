# NLP Project

## Giới thiệu
Dự án này là một hệ thống xử lý ngôn ngữ tự nhiên (NLP) đa chức năng, được xây dựng bằng Python và Streamlit. Hệ thống cung cấp các công cụ và ứng dụng liên quan đến NLP, bao gồm gán nhãn dữ liệu, hệ thống gợi ý, chatbot, và nhiều tính năng khác.

## Demo
[![Demo Video](https://img.youtube.com/vi/ZM68c_O6b5U/0.jpg)](https://www.youtube.com/watch?v=ZM68c_O6b5U)
- Xem video demo tại: [https://www.youtube.com/watch?v=ZM68c_O6b5U](https://www.youtube.com/watch?v=ZM68c_O6b5U)

## Yêu cầu hệ thống

- **Python**: Phiên bản 3.8 trở lên.
- **Các biến môi trường**: Cần thiết lập các biến môi trường trong file `.env` dựa trên mẫu `env.example` để cấu hình các thông tin như API key, client ID, v.v.

## Thiết lập biến môi trường

1. Sao chép file mẫu:
   ```bash
   copy env.example .env
   ```
2. Mở file `.env` và điền các thông tin cấu hình phù hợp với hệ thống của bạn.

## Cấu trúc dự án
Dự án bao gồm các thành phần chính sau:

### 1. **Hệ thống gán nhãn**
- **Mô tả**: Công cụ hỗ trợ gán nhãn dữ liệu văn bản, bao gồm các bước thu thập, tăng cường, tiền xử lý, biểu diễn và phân loại văn bản.
- **File liên quan**: [`GanNhan.py`](GanNhan.py)

### 2. **Hệ thống gợi ý**
Hệ thống gợi ý bao gồm nhiều phương pháp khác nhau:
- **Gợi ý dựa trên nội dung**:
  - Gợi ý phim dựa trên thể loại và nội dung.
  - **File liên quan**: [`GoiYDuaTrenNoiDung.py`](GoiYDuaTrenNoiDung.py)
- **Gợi ý dựa trên đánh giá người dùng**:
  - Gợi ý phim dựa trên sở thích của những người dùng tương tự.
  - **File liên quan**: [`goiyduatrendanhgiannguoidungvanguoidung.py`](goiyduatrendanhgiannguoidungvanguoidung.py)
- **Gợi ý dựa trên đánh giá nội dung**:
  - Gợi ý phim dựa trên độ tương đồng giữa các phim.
  - **File liên quan**: [`goiyduatrendanhgianoidung.py`](goiyduatrendanhgianoidung.py)
- **Gợi ý dựa trên ngữ cảnh**:
  - Gợi ý địa điểm dựa trên sở thích, thời tiết, mùa và thời gian trong ngày.
  - **File liên quan**: [`goiycongtac.py`](goiycongtac.py)

### 3. **Hệ thống chatbot**
Hệ thống chatbot bao gồm:
- **Chatbot agent**:
  - Chatbot được nhúng trong giao diện web thông qua iframe.
  - **File liên quan**: [`chatbox_agent.py`](chatbox_agent.py)
- **Chatbot tạo sinh**:
  - Chatbot sử dụng mô hình GPT-3.5 để trả lời câu hỏi của người dùng.
  - **File liên quan**: [`HeThongChatBotAPI.py`](HeThongChatBotAPI.py)
- **Chatbot dựa trên SentenceTransformer**:
  - Chatbot sử dụng mô hình SentenceTransformer để tìm câu trả lời phù hợp nhất từ tập dữ liệu câu hỏi và câu trả lời.
  - **File liên quan**: [`chatbox_local.py`](chatbox_local.py)

### 4. **Giao diện điều hướng**
- Giao diện điều hướng chính của dự án, cho phép người dùng truy cập các tính năng khác nhau thông qua menu.
- **File liên quan**: [`DashBoard.py`](DashBoard.py)

## Cách chạy dự án
1. **Cài đặt các thư viện cần thiết**:
   - Đảm bảo đã cài đặt Python và thiết lập biến môi trường như hướng dẫn ở trên.
   - Sử dụng `pip` để cài đặt các thư viện trong dự án:
     ```bash
     pip install -r requirements.txt
     ```
   - Một số thư viện chính bao gồm: `streamlit`, `scikit-learn`, `pandas`, `numpy`, `nltk`, `transformers`, `sentence-transformers`, `deep-translator`, `torch`, `spacy`, `kagglehub`, v.v.

2. **Chạy ứng dụng**:
   - Sử dụng lệnh sau để chạy ứng dụng Streamlit:
     ```bash
     streamlit run DashBoard.py
     ```

3. **Truy cập giao diện**:
   - Mở trình duyệt và truy cập vào địa chỉ được cung cấp (thường là `http://localhost:8501`).

## Tính năng nổi bật
- **Gán nhãn dữ liệu**: Hỗ trợ các bước xử lý dữ liệu từ thô đến phân loại.
- **Hệ thống gợi ý**: Cung cấp nhiều phương pháp gợi ý dựa trên nội dung, đánh giá người dùng, và ngữ cảnh.
- **Chatbot**:
  - Chatbot tạo sinh thông minh sử dụng mô hình GPT-3.5.
  - Chatbot dựa trên SentenceTransformer để tìm kiếm câu trả lời từ tập dữ liệu.
- **Giao diện thân thiện**: Sử dụng Streamlit để tạo giao diện người dùng dễ sử dụng.

## Đóng góp
Nếu bạn muốn đóng góp cho dự án, vui lòng tạo một pull request hoặc mở issue trên GitHub.
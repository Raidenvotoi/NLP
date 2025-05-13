

import streamlit as st
import streamlit_antd_components as sac

# (1) Thiết lập layout
st.set_page_config(layout="wide")

# (2) Định nghĩa menu items
menu_items = [
    sac.MenuItem('Hệ thống gán nhãn', icon='tags-fill'),
    sac.MenuItem('Hệ thống gợi ý', icon='hand-thumbs-up', children=[
        sac.MenuItem('Hệ thống gợi ý dựa trên nội dung', icon='hand-thumbs-up-2'),
        sac.MenuItem('Hệ thống gợi ý dựa trên đánh giá người dùng', icon='hand-thumbs-up-2'),
        sac.MenuItem('Hệ thống gợi ý dựa trên đánh giá nội dung', icon='hand-thumbs-up-2'),
        sac.MenuItem('Hệ thống gợi ý dựa trên ngữ cảnh', icon='hand-thumbs-up-2'),
        
    ]),
    sac.MenuItem('Hệ thống chatbot', icon='robot', children=[
        sac.MenuItem('Hệ thống chatbot agent', icon='robot-2'),
        sac.MenuItem('Hệ thống chatbot tạo sinh', icon='robot-2')
    ]),
]

# (3) Định nghĩa các trang
page_gan_nhan = st.Page('GanNhan.py', title='Gán nhãn')
# page_he_thong_goi_y = st.Page('HeThongGoiY.py', title='Hệ thống gợi ý')
page_he_thong_goi_y_dua_tren_noi_dung = st.Page('GoiYDuaTrenNoiDung.py', title='Hệ thống gợi ý dựa trên nội dung',icon='👍')
page_he_thong_goi_y_dua_tren_danh_gia_nguoi_dung = st.Page('goiyduatrendanhgiannguoidungvanguoidung.py', title='Hệ thống gợi ý dựa trên nội dung')
page_he_thong_goi_y_dua_tren_danh_gia_noi_dung = st.Page('goiyduatrendanhgianoidung.py', title='Hệ thống gợi ý dựa trên nội dung')
page_he_thong_goi_y_dua_tren_ngu_canh= st.Page('goiycongtac.py', title='Hệ thống gợi ý dựa trên nội dung')
page_chatbot_agent = st.Page('chatbox_agent.py', title='Hệ thống chatbot agent')
page_chatbot_generative = st.Page('HeThongChatBotAPI.py', title='Hệ thống chatbot tạo sinh')

# (4) Chèn menu vào sidebar và lấy key được chọn
with st.sidebar:
    selected = sac.menu(
        menu_items,
        open_all=True,
        key="my_antd_menu"
    )

# (5) Điều hướng dựa trên lựa chọn
# st.write(f"Bạn đang chọn: **{selected}**")
# Navigation logic
if selected == 'Hệ thống gán nhãn':
    st.navigation([page_gan_nhan]).run()
elif selected == 'Hệ thống gợi ý dựa trên nội dung':
    st.navigation([page_he_thong_goi_y_dua_tren_noi_dung]).run()
elif selected == 'Hệ thống gợi ý dựa trên đánh giá người dùng':
    st.navigation([page_he_thong_goi_y_dua_tren_danh_gia_nguoi_dung]).run()
elif selected == 'Hệ thống gợi ý dựa trên đánh giá nội dung':
    st.navigation([page_he_thong_goi_y_dua_tren_danh_gia_noi_dung]).run()
elif selected == 'Hệ thống gợi ý dựa trên ngữ cảnh':
    st.navigation([page_he_thong_goi_y_dua_tren_ngu_canh]).run()
elif selected == 'Hệ thống chatbot agent':
    st.navigation([page_chatbot_agent]).run()
elif selected == 'Hệ thống chatbot tạo sinh':
    st.navigation([page_chatbot_generative]).run()
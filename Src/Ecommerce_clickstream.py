#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install dask[complete]


# In[ ]:


import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import plotly.express as px
import numpy as np
import seaborn as sns
from matplotlib import colors
from dask.diagnostics import ProgressBar


# #  Chuẩn bị dữ liệu
# 
# ### Nguồn: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?resource=download

# In[ ]:


ProgressBar().register()  # hiện tiến độ cho .compute()
# Đọc 2 file
path = r"D:\TRƯỜNG TỰ NHIÊN\DATA ANALYTICS\PROJECT\Tối ưu chiến dịch quảng cáo Shopee - Lazada\*.csv"
df=dd.read_csv(path, dtype={'category_id': 'object'})

# Xem dòng đầu
print(df.head())

# Kiểm tra kích thước
print("Tổng số dòng:", len(df))


# In[ ]:


# Xem lại các cột
print(df.columns)


# # Xử lý và làm sạch dữ liệu

# In[ ]:


# Chuyển event_time sang Datetime
df['event_time'] = dd.to_datetime(df['event_time'], errors='coerce')


# In[ ]:


# Thêm các cột phụ để phân tích theo thời gian
df['event_date'] = df['event_time'].dt.date
df['event_hour'] = df['event_time'].dt.hour
df['event_dayofweek'] = df['event_time'].dt.dayofweek #0 = Monday, 6 = Sunday


# In[ ]:


# Xem tất cả các hành vi có trong dataset
print(df['event_type'].unique().compute())


# In[ ]:


# Lọc các hành vi chính: view, cart, purchase
df = df[df['event_type'].isin(['view', 'cart', 'purchase'])]


# In[ ]:


# Kiểm tra null
null_counts = df.isnull().sum().compute()
print(null_counts)


# ### Xử lý Missing Values (Giá trị thiếu)
# #### 1. category_code: 35,413,780 giá trị null
# - Chiếm tỷ lệ lớn (~33% tổng dữ liệu).
# - Chứa thông tin ngành hàng (vd: electronics.smartphone).
# - 👉 Giải pháp:
#     - Không drop để tránh mất dữ liệu lớn.
#     - Tạo cột mới main_category và thay null bằng 'unknown'
# #### 2. brand: 15,341,158 giá trị null
# - Nhiều giá trị thiếu, nhưng không quan trọng bằng category.
# - 👉 Giải pháp:
#     - Giữ lại, hoặc thay thế bằng 'unknown'
# #### 3. user_session: 12 giá trị null
# - Cực kỳ quan trọng khi phân tích theo phiên người dùng.
# - Chỉ thiếu 12 dòng → nên loại bỏ.

# In[ ]:


# category_code
df['main_category'] = df['category_code'].fillna('unknown').str.split('.').str[0]


# In[ ]:


# brand
df['brand'] = df['brand'].fillna('unknown')


# In[ ]:


# user_session
df = df[~df['user_session'].isnull()]


# In[ ]:


# Xem phân bổ hành vi
event_counts = df['event_type'].value_counts().compute()
print(event_counts)


# # EDA + FUNNEL

# ## [A] Phân tích theo thời gian

# In[ ]:


# 📊 1. Biểu đồ theo ngày
event_by_date = df.groupby(['event_date', 'event_type']).size().compute().unstack()
event_by_date.plot(figsize=(15,5), title='Số lượt View/Cart/Purchase theo ngày')
plt.xlabel('Ngày')
plt.ylabel('Số lượt')
plt.grid(True)
plt.show()


# ### 📈 Biểu đồ: Số lượt View / Cart / Purchase theo ngày
# #### 🧠 Nhận xét:
# - Trong suốt giai đoạn từ 1/10 đến 1/12/2019, hành vi view luôn chiếm tỷ lệ vượt trội so với cart và purchase, đúng với đặc điểm người dùng thường xuyên xem nhiều sản phẩm nhưng ít hành động.
# 
# - Có một đỉnh spike rất lớn vào khoảng ngày 11–13/11, đạt hơn 6 triệu lượt xem/ngày. Đây rất có thể trùng với chiến dịch sale 11.11, một trong những đợt khuyến mãi lớn nhất năm trên các sàn thương mại điện tử như Shopee hoặc Lazada.
# 
# - Hành vi cart và purchase cũng tăng mạnh cùng thời điểm, cho thấy người dùng không chỉ xem mà còn thực hiện chuyển đổi trong giai đoạn này.
# 
# - Sau đỉnh 11.11, lưu lượng hành vi nhanh chóng giảm trở lại và ổn định quanh mức trung bình, cho thấy tác động ngắn hạn của chiến dịch sale.
# 
# #### 🎯 Gợi ý hành động (dành cho team Marketing):
# - Cần tận dụng các dịp lớn như 11.11, 12.12, Black Friday, vì lượng người dùng chủ động tăng rất mạnh.
# - Có thể xem xét chạy các chương trình retargeting sau khi traffic tăng đột biến để tối ưu chuyển đổi.

# In[ ]:


# 📊 1. Biểu đồ theo giờ trong ngày
event_by_hour = df.groupby(['event_hour', 'event_type']).size().compute().unstack()
event_by_hour.plot(kind='bar', figsize=(15,5), title='Số lượt View/Cart/Purchase theo giờ trong ngày')
plt.xlabel('Giờ')
plt.ylabel('Số lượt')
plt.grid(True)
plt.show()


# ### 📊 Biểu đồ: Số lượt View / Cart / Purchase theo giờ trong ngày
# 
# #### 🧠 Nhận xét:
# 
# - Hành vi người dùng tăng mạnh từ **3h sáng**, đạt đỉnh từ **14h đến 17h**, đặc biệt là `view`, đạt hơn **7 triệu lượt xem** vào lúc cao nhất.
# - Hành vi `cart` và `purchase` tuy nhỏ hơn nhưng cũng có xu hướng **tăng dần về chiều**, phản ánh thời điểm người dùng bắt đầu "ra quyết định" mua sắm.
# - Sau **18h**, tất cả các hành vi bắt đầu **giảm dần**, có thể do người dùng chuyển sang các hoạt động khác như ăn tối hoặc giải trí.
# - Thấp nhất là khung **0h–2h sáng**, đúng với thời điểm ít người dùng truy cập.
# 
# #### 🎯 Gợi ý chiến lược (dành cho team Marketing):
# 
# - Nên **đặt ngân sách quảng cáo cao hơn từ 14h–17h** để tận dụng peak traffic.
# - Các chương trình flash sale có thể chạy thử vào lúc **15h** để tăng chuyển đổi.
# 
# 

# In[ ]:


# 📊 3. Biểu đồ theo thứ trong tuần
event_by_dayofweek = df.groupby(['event_dayofweek', 'event_type']).size().compute().unstack()
event_by_dayofweek.plot(kind='bar', figsize=(12,5), title='Số lượt View/Cart/Purchase theo thứ trong tuần')
plt.xlabel('Thứ (0 = Thứ hai, 6 = Chủ nhật)')
plt.ylabel('Số lượt')
plt.grid(True)
plt.show()


# ### 📊 Phân tích hành vi người dùng theo thứ trong tuần
# Biểu đồ: Lượt view, cart, purchase theo ngày trong tuần (0 = Thứ hai, 6 = Chủ nhật)
# 
# #### 🟢 Tổng quan:
# - Lượt view tăng dần từ đầu tuần và đạt đỉnh vào Thứ Sáu và Thứ Bảy.
# - Lượt cart và purchase cũng có xu hướng tương tự, cho thấy người dùng mua sắm nhiều hơn vào cuối tuần.
# - Chủ nhật vẫn có lượng truy cập cao nhưng thấp hơn Thứ Sáu và Bảy, có thể do chuẩn bị đầu tuần mới.
# 
# #### 📌 Insight chính:
# - Thứ Sáu & Thứ Bảy là thời điểm vàng để chạy các chiến dịch quảng cáo, thúc đẩy chuyển đổi.
# - Có thể tận dụng Retargeting Ads vào Thứ Năm – Thứ Sáu, khi người dùng đang có xu hướng quyết định mua sắm.
# 
# #### ✅ Đề xuất:
# - Tối ưu hóa ngân sách quảng cáo vào cuối tuần.
# - Tăng cường khuyến mãi, voucher hấp dẫn trong các ngày cao điểm (5, 6, 0).

# In[ ]:


### Funnel trong ngày
# Group theo ngày và loại hành vi
daily_counts = df.groupby(['event_date', 'event_type'])['user_session'].count().compute().unstack() 
#compute() được dùng để thực thi chuyển sang Pandas


# In[ ]:


# Tính tỉ lệ chuyển đổi trong ngày
# Tính funnel
daily_funnel = pd.DataFrame()
daily_funnel['view'] = daily_counts['view']
daily_funnel['cart'] = daily_counts['cart']
daily_funnel['purchase'] = daily_counts['purchase']

daily_funnel['view to cart rate'] = daily_funnel['cart'] / daily_funnel['view']
daily_funnel['cart to purchase rate'] = daily_funnel['purchase'] / daily_funnel['cart']
daily_funnel['view to purchase rate'] = daily_funnel['purchase'] / daily_funnel['view']


# In[ ]:


# Vẽ biểu đồ Funnel chuyển đổi theo ngày
daily_funnel_reset = daily_funnel.reset_index()

daily_funnel_reset.plot(
    x = 'event_date',
    y = ['view to cart rate', 'cart to purchase rate', 'view to purchase rate'],
    figsize = (14,5)
)

plt.title('Tỉ lệ chuyển đổi View → Cart → Purchase theo ngày')
plt.ylabel('Tỉ lệ')
plt.xlabel('Ngày')
plt.grid(True)
plt.show()


# ### 📊 Tỷ lệ chuyển đổi View → Cart → Purchase theo ngày
# #### 🟦 View to Cart Rate
# - Ổn định ở mức thấp trong khoảng 2–3%, phản ánh rằng chỉ một tỷ lệ nhỏ người dùng có hành động thêm sản phẩm vào giỏ hàng sau khi xem.
# - Có một số spike nhẹ sau ngày 11/11 – khả năng do chiến dịch sale khiến nhiều người quyết định mua nhanh hơn.
# 
# #### 🟧 Cart to Purchase Rate
# - Tỷ lệ này dao động khá mạnh.
# - Một số ngày có tỷ lệ > 100%, có thể do:
#     - Người dùng mua không cần thêm vào giỏ hàng
#     - Có hành vi mua trực tiếp từ gợi ý/sản phẩm đã xem trước đó (không thông qua hành vi cart).
# - Sau ngày 11/11, tỷ lệ này giảm mạnh và ổn định ở mức khoảng 35%.
# 
# #### 🟩 View to Purchase Rate
# - Ổn định ở mức dưới 2%, cho thấy hành vi mua hàng là rất nhỏ so với tổng lượt xem.
# - Vẫn có dấu hiệu tăng nhẹ quanh ngày 11/11.
# 
# #### 🗓️ Nhận xét theo thời gian:
# - Có dấu hiệu chuyển đổi tốt hơn trong dịp sale 11.11, dù sau đó tỷ lệ lại giảm.
# - Các tỷ lệ này giúp xác định điểm yếu trong hành trình mua sắm (tỷ lệ chuyển đổi thấp ở bước nào → tối ưu bước đó).

# In[ ]:


### Funnel trước và sau dịp sale (11/11)
# Gán nhãn thành 3 giai đoạn
def label_sale_period(date):
    # Chỉ ép kiểu nếu chưa phải datetime.date
    if isinstance(date, datetime.datetime):
        date = date.date()
    if date < datetime.date(2019, 11, 11):
        return 'before_sale'
    elif date == datetime.date(2019, 11, 11):
        return 'sale_day'
    else:
        return 'after_sale'
    
daily_funnel['sale_period'] = daily_funnel.index.map(label_sale_period)


# In[ ]:


# Tính trung bình chuyển đổi theo từng giai đoạn
sale_funnel_summary = daily_funnel.groupby('sale_period')[['view to cart rate', 'cart to purchase rate', 'view to purchase rate']].mean()


# In[ ]:


# Reorder theo thứ tự: Before -> sale -> after
order = ['before_sale', 'sale_day', 'after_sale']
sale_funnel_summary = sale_funnel_summary.loc[order]

# Vẽ biểu đồ so sánh
sale_funnel_summary.plot(kind='bar', figsize=(12,5))
plt.title('So sánh tỉ lệ chuyển đổi trước - trong - sau ngày sale 11.11')
plt.ylabel('Tỉ lệ')
plt.xticks(rotation = 0)
plt.legend(loc = 'upper right')
plt.grid(True)
plt.show()


# ### 📊 So sánh Tỉ lệ Chuyển đổi Trước – Trong – Sau Ngày Sale 11.11
# #### 💡 Mục tiêu:
# - Phân tích hiệu quả của chiến dịch sale 11.11 dựa trên các tỉ lệ chuyển đổi chính:
#     - View to Cart Rate (Tỉ lệ xem → thêm giỏ)
#     - Cart to Purchase Rate (Tỉ lệ giỏ → mua)
#     - View to Purchase Rate (Tỉ lệ xem → mua)
# #### 📌 Insight:
# - Tỉ lệ Cart to Purchase trong giai đoạn trước ngày sale cao đột biến so với các giai đoạn khác. Điều này cho thấy người dùng thêm hàng vào giỏ rất lâu trước đó và chờ đến đúng ngày sale để mua.
# - Trong ngày sale 11.11, tỉ lệ View to Cart và View to Purchase tăng nhẹ so với trước đó, cho thấy tác động từ các chiến dịch truyền thông và khuyến mãi đã thu hút thêm người xem và mua hàng.
# - Sau sale, các tỉ lệ quay về mức gần như ban đầu, phản ánh đúng tính chất ngắn hạn của chiến dịch.
# #### ✅ Gợi ý hành động:
# - Tận dụng giai đoạn trước ngày sale để thúc đẩy người dùng thêm sản phẩm vào giỏ hàng qua các chiến dịch remarketing.
# - Trong ngày sale, tập trung vào việc tối ưu hóa trải nghiệm mua hàng (giảm lag, tăng tốc thanh toán, khuyến mãi flash sale) để chuyển đổi giỏ thành đơn.
# - Sau ngày sale, tiếp tục remarketing tới nhóm người đã xem hoặc đã thêm vào giỏ nhưng chưa mua để gia tăng doanh thu hậu chiến dịch.

# In[ ]:


# Funnel chuyển đổi ngược lại
# Tỉ lệ cart bị bỏ quên
daily_funnel['cart_abandonment_rate'] = 1 - daily_funnel['cart to purchase rate']
# Tỉ lệ view mà không thêm vào cart
daily_funnel['view_no_cart_rate'] = 1 - daily_funnel['view to cart rate']

# Ve biểu đồ
daily_funnel[['cart_abandonment_rate', 'view_no_cart_rate']].plot(figsize=(14,5))
plt.title('Tỉ lệ rớt khách: Bỏ giỏ và không thêm hàng vào giỏ')
plt.ylabel("Tỉ lệ")
plt.xlabel('Ngày')
plt.grid(True)
plt.legend(loc = 'upper right')
plt.show()


# ## 📊 Tỉ lệ rớt khách: Bỏ giỏ và không thêm hàng vào giỏ
# ### 🧮 Định nghĩa chỉ số:
# - Tỉ lệ bỏ giỏ hàng (cart_abandonment_rate): Trong số người đã thêm vào giỏ, bao nhiêu % không hoàn tất mua hàng.
# - Tỉ lệ xem không thêm vào giỏ (view_no_cart_rate): Trong số lượt xem sản phẩm, bao nhiêu % không thêm vào giỏ.
# 
# ### 📌 Insight & Diễn giải biểu đồ:
# #### ✅ Hành vi khách hàng:
# - Tỉ lệ bỏ giỏ hàng (cart_abandonment_rate)
#     - Có sự biến động lớn ở đầu giai đoạn, thậm chí có lúc âm rơi vào thời điểm đầu tháng 10(có thể là do: Một số người bấm "Mua lại" từ đơn cũ hoặc thêm vào giỏ từ hôm trước, nhưng mua vào hôm nay) và một vài đoạn đứt gãy (người dùng có thể thanh toán mà không hề thêm vào giỏ).
#     - Giai đoạn nửa sau (từ tháng 11) có xu hướng ổn định hơn, dao động ~65–70%, tức cứ 10 người thêm hàng vào giỏ thì 6–7 người không thanh toán.
# - Tỉ lệ xem không thêm giỏ (view_no_cart_rate): Luôn cao vượt trội (~97–99%), chứng tỏ khách hàng chủ yếu chỉ xem lướt mà chưa sẵn sàng hành động mua sắm.
# #### 💡 Gợi ý hành động:
# - Tập trung cải thiện tỉ lệ thêm vào giỏ (view_to_cart_rate) bằng:
#     - Giao diện hấp dẫn hơn (ảnh/video đẹp, giá hiển thị rõ).
#     - Tạo ưu đãi khi người dùng thêm vào giỏ (giảm giá nhỏ, voucher ngay khi thêm).
# - Với tỉ lệ bỏ giỏ cao:
#     - Gửi nhắc nhở qua email/app.
#     - Ưu đãi giới hạn thời gian cho giỏ hàng đang chờ.
#     - Giảm friction khi thanh toán (1-click checkout, lưu địa chỉ, ẩn phí ship).

# In[ ]:


# Funnel tỉ lệ chuyển đổi ngược trươc, trong và sau ngày sale
reverse_rate_summary = daily_funnel.groupby('sale_period')[['cart_abandonment_rate', 'view_no_cart_rate']].mean()
# Reorder sale_period theo ý muốn
reverse_rate_summary = reverse_rate_summary.loc[order]
# Vẽ biểu đồ
reverse_rate_summary.plot(kind = 'bar', figsize = (10,5))
plt.title('So sánh tỉ lệ rớt khách theo từng giai đoạn sale')
plt.ylabel('Tỉ lệ')
plt.xticks(rotation = 0)
plt.grid(True)
plt.show()


# ### 📊 So sánh Tỉ lệ Rớt khách theo từng Giai đoạn Sale
# #### ✅ Biểu đồ mô tả
# - cart_abandonment_rate: Tỉ lệ người dùng đã thêm hàng vào giỏ nhưng không mua hàng.
# - view_no_cart_rate: Tỉ lệ người dùng chỉ xem sản phẩm mà không thêm vào giỏ.
# #### 📌 Nhận xét nhanh:
# - Trước sale (before_sale):
#     - Tỉ lệ không thêm hàng vào giỏ rất cao (~98%), chứng tỏ khách chủ yếu tham khảo giá.
#     - Tỉ lệ bỏ giỏ thấp, do chưa có nhiều người thêm hàng.
# - Trong ngày sale (sale_day):
#     - Tỉ lệ bỏ giỏ tăng mạnh (~68%), cho thấy nhiều người so sánh – chọn lọc – chần chừ khi mua.
#     - Tỉ lệ không thêm giỏ vẫn cao (~97%), phản ánh hành vi xem rồi bỏ qua nhiều.
# - Sau sale (after_sale):
#     - Tỉ lệ bỏ giỏ duy trì ở mức cao (~68%), có thể do người dùng hối tiếc – quay lại xem – nhưng không mua.
#     - Tỉ lệ không thêm vào giỏ cũng vẫn cao.
# 
# #### 💡 Insight chính:
# - Khách hàng vẫn giữ hành vi “xem nhưng không hành động” xuyên suốt cả 3 giai đoạn.
# - Tỉ lệ bỏ giỏ tăng vào ngày sale là một dấu hiệu cho thấy họ đã cân nhắc kỹ nhưng không đủ động lực chốt đơn.
# 
# #### 🎯 Gợi ý hành động (Marketing):
# - Before Sale:
#     - Gợi ý thêm sản phẩm nổi bật
#     - Cung cấp thông tin về deal sắp tới
#     - Gợi ý đăng ký nhắc sale
# - Sale Day:
#     - Tăng cấp độ FOMO (sắp hết hàng, sắp kết thúc deal)
#     - Ưu đãi "free ship khi checkout ngay"
#     - Remarketing real-time nếu khách bỏ giỏ
# - After Sale:
#     - Gửi mail “Bạn đã bỏ lỡ?” + coupon nhỏ
#     - Đề xuất sản phẩm tương tự hoặc gói combo
#     - Làm khảo sát lý do không mua

# ## [B] Phân tích Funnel theo ngành hàng

# In[ ]:


# 1) Lọc dữ liệu và bỏ null
df_filtered = df[
    df['event_type'].isin(['view', 'cart', 'purchase'])
].dropna(subset=['main_category', 'user_id'])

# 2) Giữ 1 bản ghi duy nhất cho mỗi (main_category, event_type, user_id)
dedup = df_filtered[['main_category', 'event_type', 'user_id']].drop_duplicates()

# 3) Đếm user cho từng nhóm (trả về Dask DataFrame)
counts_dd = dedup.groupby(['main_category', 'event_type']).user_id.count().reset_index()

# 4) Chuyển sang Pandas để pivot
counts = counts_dd.compute()

# 5) Pivot dạng wide
funnel_by_category = counts.pivot(index='main_category', columns='event_type', values='user_id').fillna(0).reset_index()

# 6) Đảm bảo có đủ cột view/cart/purchase
for col in ['view', 'cart', 'purchase']:
    if col not in funnel_by_category.columns:
        funnel_by_category[col] = 0

# 7) Tính tỷ lệ
funnel_by_category['cart_to_purchase_rate'] = (
    funnel_by_category['purchase'] / funnel_by_category['cart'] * 100
).where(funnel_by_category['cart'] > 0, 0)

funnel_by_category['cart_abandonment_rate'] = (
    (funnel_by_category['cart'] - funnel_by_category['purchase']) / funnel_by_category['cart'] * 100
).where(funnel_by_category['cart'] > 0, 0)

print(funnel_by_category.head())


# In[ ]:


# Tính thêm conversion rate
funnel_by_category['view_to_cart'] = funnel_by_category['cart'] / funnel_by_category['view']
funnel_by_category['cart_to_purchase'] = funnel_by_category['purchase'] / funnel_by_category['cart']
funnel_by_category['view_to_purchase'] = funnel_by_category['purchase'] / funnel_by_category['view']


# In[ ]:


# Chuẩn hoá & kiểm tra
# Mục đích: tránh giá trị vô lý (inf, NaN, âm), chuẩn hoá dạng số.
# 1. fillna + replace inf
funnel_by_category[['view','cart','purchase']] = funnel_by_category[['view','cart','purchase']].fillna(0)
funnel_by_category.replace([np.inf, -np.inf], 0, inplace=True)

# 2. tránh chia 0: những tỷ lệ đã tính thì clip và fillna
for col in ['view_to_cart','cart_to_purchase','view_to_purchase']:
    funnel_by_category[col] = funnel_by_category[col].fillna(0).clip(lower=0)

# 3. tạo bản hiển thị % (tuỳ chọn)
funnel_by_category['view_to_cart_pct'] = (funnel_by_category['view_to_cart'] * 100).round(2)
funnel_by_category['cart_to_purchase_pct'] = (funnel_by_category['cart_to_purchase'] * 100).round(2)
funnel_by_category['view_to_purchase_pct'] = (funnel_by_category['view_to_purchase'] * 100).round(2)

# 4. kiểm tra xem có ngành nào có tỷ lệ >1 (bất thường)
anomalies = funnel_by_category[(funnel_by_category['cart_to_purchase'] > 1) | (funnel_by_category['view_to_cart'] > 1)]
print("Anomalies (tỉ lệ > 1):", anomalies[['main_category','view','cart','purchase','view_to_cart','cart_to_purchase']].head())


# In[ ]:


# Chuẩn bị dữ liệu trực quan hoá
# Chỉ lấy Top 10 ngành hàng có nhiều view nhất
top_categories = funnel_by_category.sort_values('view', ascending=False).head(10)

# Định dạng tỷ lệ %
top_categories['view_to_cart_pct'] = top_categories['view_to_cart'] * 100
top_categories['cart_to_purchase_pct'] = top_categories['cart_to_purchase'] * 100
top_categories['view_to_purchase_pct'] = top_categories['view_to_purchase'] * 100


# #### Vẽ biểu đồ Funnel theo ngành hàng

# In[ ]:


# Lọc unknown và lấy số liệu unknown
unknown_value = (
    top_categories.loc[top_categories['main_category'] == 'unknown', 'view_to_cart_pct'].values[0] 
    if 'unknown' in top_categories['main_category'].values 
    else None
)

# Lọc dữ liệu vẽ biểu đồ chính
filtered_categories = top_categories[top_categories['main_category'] != 'unknown']

# Vẽ biểu đồ
plt.figure(figsize=(10,6))
filtered_categories.sort_values('view_to_cart_pct', ascending=False).plot(
    x='main_category', 
    y='view_to_cart_pct', 
    kind='bar', 
    color='skyblue',
    legend=False
)
plt.ylabel('Tỷ lệ View → Cart (%)')
plt.title('Top ngành hàng theo tỷ lệ View → Cart (loại bỏ Unknown)')
plt.xticks(rotation=45, ha='right')

# Thêm box thông tin unknown
if unknown_value is not None:
    plt.gca().text(
        0.95, 0.95, 
        f"Unknown: {unknown_value:.2f}%", 
        transform=plt.gca().transAxes,
        fontsize=10,
        color='red',
        ha='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
    )

plt.tight_layout()
plt.show()


# In[ ]:


# Lọc ra unknown value nếu có
unknown_value = (
    top_categories.loc[top_categories['main_category'] == 'unknown', 'cart_to_purchase_pct'].values[0]
    if 'unknown' in top_categories['main_category'].values
    else None
)

# Lọc bỏ unknown khỏi dữ liệu vẽ biểu đồ
top_categories_no_unknown = top_categories[top_categories['main_category'] != 'unknown']

# Bar chart – So sánh tỷ lệ chuyển đổi cart → purchase
plt.figure(figsize=(10,6))
top_categories_no_unknown.sort_values('cart_to_purchase_pct', ascending=False).plot(
    x='main_category', 
    y='cart_to_purchase_pct', 
    kind='bar', 
    color='orange',
    legend=False
)

plt.ylabel('Tỷ lệ Cart → Purchase (%)')
plt.title('Top ngành hàng theo tỷ lệ Cart → Purchase')
plt.xticks(rotation=45, ha='right')

# Nếu có unknown, thêm box nhỏ hiển thị giá trị
if unknown_value is not None:
    plt.text(
        0.95, 0.95,
        f"Unknown: {unknown_value:.2f}%",
        transform=plt.gca().transAxes,
        fontsize=10,
        color='red',
        ha='right', va='top',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3')
    )

plt.tight_layout()
plt.show()


# In[ ]:


# Lấy giá trị unknown bằng .item()
if 'unknown' in top_categories['main_category'].values:
    unknown_value = top_categories.loc[
        top_categories['main_category'] == 'unknown', 
        'view_to_purchase_pct'
    ].item()
else:
    unknown_value = None

# Bỏ unknown để không vẽ
df_plot = top_categories[top_categories['main_category'] != 'unknown']

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(10,6))
df_plot.sort_values('view_to_purchase_pct', ascending=False).plot(
    x='main_category', 
    y='view_to_purchase_pct', 
    kind='bar', 
    color='green',
    legend=False,
    ax=ax
)

ax.set_ylabel('Tỷ lệ View → Purchase (%)')
ax.set_title('Top ngành hàng theo tỷ lệ View → Purchase')
plt.xticks(rotation=45, ha='right')

# Chèn box chú thích Unknown
if unknown_value is not None:
    ax.text(
        0.95, 0.95,
        f"Unknown: {unknown_value:.2f}%",
        transform=ax.transAxes,
        fontsize=10,
        color='red',
        ha='right',
        va='top',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3')
    )

plt.tight_layout()
plt.show()


# In[ ]:


# Phân loại nhóm ngành hàng từ 3 tỷ lệ View → Cart, View → Cart, View → Purchase
def scatter_plot(df, x_col, y_col, title, xlabel, ylabel):
    plt.figure(figsize=(8,6))
    plt.scatter(df[x_col], df[y_col], color='skyblue', edgecolor='black')
    
    # Gắn nhãn tên ngành hàng
    for i, txt in enumerate(df['main_category']):
        plt.annotate(txt, (df[x_col].iloc[i]+0.2, df[y_col].iloc[i]+0.2), fontsize=8)
    
    plt.axhline(df[y_col].mean(), color='red', linestyle='--', label=f'Mean {ylabel}')
    plt.axvline(df[x_col].mean(), color='green', linestyle='--', label=f'Mean {xlabel}')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 1. View→Cart vs Cart→Purchase
scatter_plot(
    top_categories, 
    'view_to_cart_pct', 'cart_to_purchase_pct',
    'Phân loại ngành hàng: View→Cart vs Cart→Purchase',
    'Tỷ lệ View→Cart (%)', 'Tỷ lệ Cart→Purchase (%)'
)

# 2. View→Cart vs View→Purchase
scatter_plot(
    top_categories, 
    'view_to_cart_pct', 'view_to_purchase_pct',
    'Phân loại ngành hàng: View→Cart vs View→Purchase',
    'Tỷ lệ View→Cart (%)', 'Tỷ lệ View→Purchase (%)'
)

# 3. Cart→Purchase vs View→Purchase
scatter_plot(
    top_categories, 
    'cart_to_purchase_pct', 'view_to_purchase_pct',
    'Phân loại ngành hàng: Cart→Purchase vs View→Purchase',
    'Tỷ lệ Cart→Purchase (%)', 'Tỷ lệ View→Purchase (%)'
)


# In[ ]:


# Vẽ heatmap cho từng tỷ lệ cũa mỗi ngành hàng 
# Dữ liệu heatmap
heatmap_data = top_categories.set_index('main_category')[
    ['view_to_cart_pct', 'cart_to_purchase_pct', 'view_to_purchase_pct']
]

# Tạo figure và ax trước
fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.get_cmap("RdYlGn")

# Vẽ heatmap không annot trước
sns.heatmap(
    heatmap_data,
    cmap=cmap,
    cbar_kws={'label': 'Tỷ lệ (%)'},
    annot=False,
    ax=ax
)

# Thêm text thủ công với màu chữ auto
norm = colors.Normalize(vmin=heatmap_data.min().min(),
                        vmax=heatmap_data.max().max())
for y in range(heatmap_data.shape[0]):
    for x in range(heatmap_data.shape[1]):
        value = heatmap_data.iloc[y, x]
        color_rgb = cmap(norm(value))[:3]  # màu nền
        brightness = np.dot(color_rgb, [0.299, 0.587, 0.114])  # độ sáng
        text_color = "black" if brightness > 0.5 else "white"
        ax.text(
            x + 0.5, y + 0.5,
            f"{value:.2f}",
            ha='center', va='center',
            color=text_color, fontsize=10
        )

plt.title('Hiệu suất ngành hàng – Heatmap')
plt.ylabel('Ngành hàng')
plt.xlabel('Chỉ số hiệu suất')
plt.tight_layout()
plt.show()


# ## 📊 Insight từ Heatmap Hiệu Suất Ngành Hàng
# 
# | Ngành hàng      | View → Cart (%) | Cart → Purchase (%) | View → Purchase (%) | Nhận xét chính                                                                                           |
# |-----------------|-----------------|---------------------|---------------------|----------------------------------------------------------------------------------------------------------|
# | **Kids**        | 4.77            | **71.39**           | 3.41                | Tỷ lệ chuyển đổi từ giỏ sang mua rất cao, nhưng bước từ xem sang giỏ cực thấp → Cần cải thiện bước đầu phễu |
# | **Auto**        | 7.29            | **70.79**           | 5.16                | Tương tự "Kids", bước đầu phễu còn yếu nhưng giữ khách rất tốt ở giai đoạn cuối                           |
# | **Unknown**     | 11.62           | **69.53**           | 8.08                | Hiệu suất cuối tốt, nhưng danh mục chưa rõ → Nên phân loại để chiến dịch cá nhân hóa                      |
# | **Appliances**  | 15.49           | 65.96               | 10.22               | Bước đầu khá tốt, cần đẩy mạnh chuyển đổi giỏ sang mua                                                     |
# | **Electronics** | 20.62           | 63.07               | 13.01               | Dẫn đầu về bước View → Cart, nhưng tỉ lệ cuối chưa tương xứng → Tối ưu bước chốt đơn                      |
# | **Apparel**     | 3.72            | 60.03               | 2.23                | Tỷ lệ thấp ở mọi bước → Cần xem xét lại chiến lược sản phẩm, giá và quảng cáo                              |
# | **Furniture**   | 4.65            | 66.11               | 3.07                | Bước giữa mạnh nhưng khởi đầu yếu                                                                         |
# | **Computers**   | 10.38           | 63.00               | 6.54                | Cần tối ưu giai đoạn đầu và cuối phễu                                                                     |
# | **Construction**| 8.62            | 60.10               | 5.18                | Hiệu suất vừa phải, chưa nổi bật                                                                          |
# | **Accessories** | 3.52            | 66.85               | 2.35                | Khởi đầu rất yếu, giữ khách khá tốt ở giai đoạn cuối                                                      |
# 
# ---
# 
# ## 💡 Đề xuất hành động
# 
# **1. Tăng tỉ lệ View → Cart cho nhóm "Kids" và "Auto"**
# - Đẩy mạnh hình ảnh, video sản phẩm sinh động, demo thực tế  
# - Tăng khuyến mãi ở bước xem sản phẩm để kích thích thêm vào giỏ
# 
# **2. Tối ưu bước Cart → Purchase cho nhóm "Appliances" và "Electronics"**  
# - Giảm rào cản thanh toán (thêm phương thức trả góp, ví điện tử)  
# - Tạo urgency: flash sale, countdown timer, voucher áp dụng khi thanh toán
# 
# **3. Phân loại rõ nhóm "Unknown"**  
# - Kiểm tra lại dữ liệu gán nhãn sản phẩm  
# - Gắn tag ngành hàng để cá nhân hóa chiến dịch quảng cáo
#    
# **4. Cải thiện toàn bộ funnel cho "Apparel" và "Accessories"**  
# - Xem xét lại giá, chất lượng, mô tả sản phẩm  
# - Chạy chiến dịch remarketing, tập trung vào khách đã từng xem sản phẩm
#    
# **5. Theo dõi hiệu suất liên tục và A/B Testing**  
# - Test 2–3 biến thể landing page, hình ảnh, thông điệp  
# - Đo lường sự thay đổi ở từng bước funnel để điều chỉnh kịp thời  

# # [C] Phân tích theo tần suất & giá trị khách hàng (RFM Analysis)

# In[ ]:


df['event_time'] = dd.to_datetime(df['event_time'], errors='coerce', utc=True).dt.tz_localize(None)
df_purchase = df[df['event_type'] == 'purchase']
# Tính ngày phân tích
analysis_date = df_purchase['event_time'].max().compute().replace(tzinfo=None) + pd.Timedelta(days=1)


# In[ ]:


# Bước 3: Recency
recency_df = df_purchase.groupby('user_id')['event_time'].max().reset_index()
recency_df['event_time'] = recency_df['event_time'].dt.tz_localize(None)  # Bỏ timezone nếu còn
recency_df['Recency'] = (analysis_date - recency_df['event_time']).dt.days
recency_df = recency_df[['user_id', 'Recency']]

# Bước 4: Frequency
freq_df = df_purchase.groupby('user_id').size().reset_index()
freq_df = freq_df.rename(columns={0: 'Frequency'})

# Bước 5: Monetary
monetary_df = df_purchase.groupby('user_id')['price'].sum().reset_index()
monetary_df = monetary_df.rename(columns={'price': 'Monetary'})

# Bước 6: Gộp lại
rfm = recency_df.merge(freq_df, on='user_id').merge(monetary_df, on='user_id')

rfm = rfm.compute()
print(rfm.head())


# In[ ]:


# Phân loại điểm R, F, M ---
# Chia điểm theo phân vị (quintile)
rfm['R_quartile'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])  # Recency càng thấp càng tốt
rfm['F_quartile'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])  # Frequency càng cao càng tốt
rfm['M_quartile'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])  # Monetary càng cao càng tốt

# --- Bước 8: Tạo RFM Score ---
rfm['RFM_Score'] = rfm['R_quartile'].astype(str) + rfm['F_quartile'].astype(str) + rfm['M_quartile'].astype(str)


# In[ ]:


# Biểu đồ phân bố Recency, Frequency, Monetary ---
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.histplot(rfm['Recency'], bins=30, kde=False)
plt.title('Distribution of Recency')

plt.subplot(1,3,2)
sns.histplot(rfm['Frequency'], bins=30, kde=False)
plt.title('Distribution of Frequency')

plt.subplot(1,3,3)
sns.histplot(rfm['Monetary'], bins=30, kde=False)
plt.title('Distribution of Monetary')

plt.tight_layout()
plt.show()


# In[ ]:


# --- Mở rộng phân khúc RFM ---
def rfm_segment_extended(row):
    if row['R_quartile'] == 4 and row['F_quartile'] == 4 and row['M_quartile'] == 4:
        return 'Champions'
    elif row['R_quartile'] >= 3 and row['F_quartile'] >= 3:
        return 'Loyal Customers'
    elif row['R_quartile'] == 4 and row['F_quartile'] <= 2:
        return 'Potential Loyalists'
    elif row['R_quartile'] == 3 and row['F_quartile'] <= 2:
        return 'New Customers'
    elif row['R_quartile'] == 4 and row['F_quartile'] == 1:
        return 'Promising'
    elif row['R_quartile'] == 2 and row['F_quartile'] >= 3:
        return 'Need Attention'
    elif row['R_quartile'] == 2 and row['F_quartile'] <= 2:
        return 'About to Sleep'
    elif row['R_quartile'] == 1 and row['F_quartile'] >= 3:
        return 'At Risk'
    elif row['R_quartile'] == 1 and row['F_quartile'] == 2:
        return 'Hibernating'
    else:
        return 'Lost'

rfm['Segment_Extended'] = rfm.apply(rfm_segment_extended, axis=1)

# --- Thống kê doanh thu & số lượng ---
rfm_summary = rfm.groupby('Segment_Extended').agg({
    'user_id': 'count',
    'Monetary': 'sum',
    'Recency': 'mean',
    'Frequency': 'mean'
}).rename(columns={'user_id': 'CustomerCount', 'Monetary': 'TotalRevenue'})

# Thêm % khách hàng và % doanh thu
rfm_summary['CustomerPct'] = (rfm_summary['CustomerCount'] / rfm_summary['CustomerCount'].sum() * 100).round(2)
rfm_summary['RevenuePct'] = (rfm_summary['TotalRevenue'] / rfm_summary['TotalRevenue'].sum() * 100).round(2)


# In[ ]:


# Biểu đồ tỷ lệ nhóm khách hàng ---
# --- Biểu đồ tỷ lệ nhóm khách hàng (Segment_Extended) ---
plt.figure(figsize=(8,6))
segment_counts_ext = rfm['Segment_Extended'].value_counts()
plt.pie(segment_counts_ext, labels=segment_counts_ext.index, autopct='%1.1f%%', startangle=140)
plt.title('Customer Segments Distribution (Extended)')
plt.show()


# In[ ]:


# =========================
# 1. Xuất bảng Traffic
# =========================
# Tổng hợp theo ngày, giờ, thứ
traffic_df = df.groupby(['event_date', 'event_hour', 'event_dayofweek', 'event_type']).size().compute().unstack().reset_index()
traffic_df.to_csv("traffic.csv", index=False)
print("✅ Đã lưu traffic.csv")

# =========================
# 2. Xuất bảng Funnel
# =========================
# daily_funnel đã có sẵn trong code trước đó
daily_funnel_reset = daily_funnel.reset_index()
daily_funnel_reset.to_csv("funnel.csv", index=False)
print("✅ Đã lưu funnel.csv")

# =========================
# 3. Xuất bảng Category Performance
# =========================
# funnel_by_category đã được tính ở phần trước
funnel_by_category.to_csv("category.csv", index=False)
print("✅ Đã lưu category.csv")

# =========================
# 4. Xuất bảng RFM Extended
# =========================
# rfm đã được gán thêm Segment_Extended ở phần RFM mở rộng
rfm.to_csv("rfm_extended.csv", index=False)
print("✅ Đã lưu rfm_extended.csv")

# =========================
# 5. Xuất bảng RFM Summary
# =========================
# rfm_summary đã được tính ở phần RFM mở rộng
rfm_summary.to_csv("rfm_summary.csv", index=True)
print("✅ Đã lưu rfm_summary.csv")

print("\n🎯 Tất cả file CSV đã sẵn sàng import vào Power BI!")


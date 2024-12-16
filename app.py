import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from vnstock3 import Vnstock

# Hàm lấy dữ liệu lịch sử cổ phiếu
def get_stock_price(symbol, start_date, end_date):
    # Tạo đối tượng Vnstock và lấy dữ liệu lịch sử cổ phiếu
    stock = Vnstock().stock(symbol=symbol, source='VCI')
    stock_hist = stock.quote.history(start=start_date, end=end_date)
    
    # Tính toán phần trăm thay đổi và sự thay đổi giá trị
    stock_hist["%_change"] = round(((stock_hist['close'] - stock_hist['close'].shift(1)) / stock_hist['close'].shift(1)) * 100, 2)
    stock_hist['change'] = stock_hist['close'] - stock_hist['close'].shift(1)
    
    # Chuyển đổi cột 'time' thành kiểu chuỗi nếu có
    if 'time' in stock_hist.columns:
        stock_hist['time'] = stock_hist['time'].astype(str)
    
    # Đổi tên cột 'ticker' thành 'stock_name'
    stock_hist = stock_hist.rename(columns={'ticker': 'stock_name', 'time': 'date'})
    
    # Tính toán các đường MA50 và MA100
    stock_hist['MA50'] = stock_hist['close'].rolling(window=50).mean()
    stock_hist['MA100'] = stock_hist['close'].rolling(window=100).mean()
    
    return stock_hist


# Hàm chuẩn bị dữ liệu và dự đoán n giá cổ phiếu
def predict_stock_price(model, data, n_steps=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = data['close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_prices)

    last_60_days = scaled_data[-60:]
    X_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

    predicted_prices = []
    for _ in range(n_steps):
        predicted_price = model.predict(X_test)
        predicted_prices.append(predicted_price[0][0])
        last_60_days = np.append(last_60_days[1:], predicted_price)
        X_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices.flatten()

# Tải mô hình
model_path = 'stock_prediction.h5'
model = load_model(model_path)

# Giao diện ứng dụng Streamlit
st.sidebar.header('Stock Market Prediction')
symbols = ['TOS', 'BSR', 'PLX', 'PVS', 'PVD', 'POS']
selected_symbol = st.sidebar.selectbox('Chọn mã cổ phiếu', symbols)

start_date = st.sidebar.date_input('Chọn ngày bắt đầu', date(2020, 1, 1))
end_date = st.sidebar.date_input('Chọn ngày kết thúc', date.today())

try:
    with st.spinner('Đang tải dữ liệu...'):
        data = get_stock_price(selected_symbol, str(start_date), str(end_date))
        if data.empty:
            st.warning("Không có dữ liệu trong khoảng thời gian đã chọn.")
        else:
            # Dự đoán giá cổ phiếu trong 5 ngày
            predicted_prices = predict_stock_price(model, data, n_steps=5)
            latest_close = data['close'].iloc[-1]
            latest_volume = data['volume'].iloc[-1]
            latest_change = data['change'].iloc[-1]

            st.subheader(f'Thông tin cổ phiếu {selected_symbol}')
            last_date = data['date'].iloc[-1]
            st.write(f"Dữ liệu ngày: {last_date}")
            # Tạo các card hiển thị thông tin
            col1, col2, col3 = st.columns(3)
            col1.metric("Giá hiện tại", f"{latest_close*1000:.0f} VND", f"{latest_change:.2f} VND")
            col2.metric("Khối lượng giao dịch", f"{latest_volume:,}")
            col3.metric("% Thay đổi", f"{data['%_change'].iloc[-1]:.2f}%")

            # Biểu đồ sparkline
            fig_spark = go.Figure(go.Scatter(x=data['date'], y=data['close'], mode='lines', line=dict(color='blue')))
            fig_spark.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_spark, use_container_width=True)
            # Biểu đồ cột cho khối lượng giao dịch
            fig_volume = go.Figure(go.Bar(x=data['date'], y=data['volume'], marker=dict(color='rgba(0, 204, 255, 10)')))
            fig_volume.update_layout(
                title='Khối lượng giao dịch',
                xaxis_title='Ngày',
                yaxis_title='Khối lượng',
                height=300,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_volume, use_container_width=True)
            # Biểu đồ MA50 và MA100 trên một biểu đồ riêng biệt
            
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=data['date'], y=data['close'], mode='lines', name='Giá đóng cửa'))
            fig_ma.add_trace(go.Scatter(x=data['date'], y=data['MA50'], mode='lines', name='MA50', line=dict(color='orange')))
            fig_ma.add_trace(go.Scatter(x=data['date'], y=data['MA100'], mode='lines', name='MA100', line=dict(color='green')))

            fig_ma.update_layout(title=f'MA50 và MA100 cho cổ phiếu {selected_symbol}', xaxis_title='Ngày', yaxis_title='Giá (VND)')
            st.plotly_chart(fig_ma)
            
            # Biểu đồ chính
            future_dates = pd.date_range(data['date'].iloc[-1], periods=6, freq='D').strftime('%Y-%m-%d')[1:]
            fig = go.Figure()

            # Vẽ giá đóng cửa và dự đoán
            fig.add_trace(go.Scatter(x=data['date'], y=data['close'], mode='lines', name='Giá đóng cửa'))
            fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Dự đoán', line=dict(dash='dot', color='red')))


            fig.update_layout(title=f'Biểu đồ giá cổ phiếu {selected_symbol}', xaxis_title='Ngày', yaxis_title='Giá (VND)')
            st.plotly_chart(fig)

            st.download_button(
                label="Tải xuống dữ liệu",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name=f'{selected_symbol}_stock_data.csv',
                mime='text/csv'
            )

except Exception as e:
    st.error(f"Không thể tải dữ liệu cho {selected_symbol}: {e}")

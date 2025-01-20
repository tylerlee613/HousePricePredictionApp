import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data_path = "data/historical_data.csv"
data = pd.read_csv(data_path)

# Create a lag feature for the previous year's house price
data['Previous_Median_House_Price'] = data['Median_House_Price'].shift(1)
data.dropna(inplace=True)

# Train the machine learning model
features = ['Unemployment_Rate', 'Inflation_Rate', 'Interest_Rate', 'Previous_Median_House_Price']
target = 'Median_House_Price'

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# App title and description
st.title("Median House Price Predictor")
st.write("""
Welcome to the Median House Price Predictor! This tool uses historical data and machine learning to forecast future median house prices in the United States. Adjust the economic indicators and select a year to see the predicted median house price.
""")

# Add spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Adjust Economic Indicators")
unemployment_rate = st.sidebar.slider("Unemployment Rate (%)", 0.0, 20.0, 6.0, 0.1)
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 20.0, 3.0, 0.1)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 20.0, 5.0, 0.1)
target_year = st.sidebar.slider("Select Target Year", 2025, 2030, 2025)

# Calculate historical averages (1975–2024)
average_unemployment = round(data.loc[data['Year'] <= 2024, 'Unemployment_Rate'].mean(), 2)
average_inflation = round(data.loc[data['Year'] <= 2024, 'Inflation_Rate'].mean(), 2)
average_interest = round(data.loc[data['Year'] <= 2024, 'Interest_Rate'].mean(), 2)

# Display historical averages below the sliders
st.sidebar.markdown("### Historical Averages (1975–2024)")
st.sidebar.write(f"**Unemployment Rate**: {average_unemployment}%")
st.sidebar.write(f"**Inflation Rate**: {average_inflation}%")
st.sidebar.write(f"**Interest Rate**: {average_interest}%")

# Prediction function
def predict_price(unemployment_rate, inflation_rate, interest_rate, year):
    previous_price = 400000  # Base price for 2024
    for target_year in range(2025, year + 1):
        input_data = {
            'Unemployment_Rate': [unemployment_rate],
            'Inflation_Rate': [inflation_rate],
            'Interest_Rate': [interest_rate],
            'Previous_Median_House_Price': [previous_price]
        }
        input_df = pd.DataFrame(input_data)
        predicted_price = round(model.predict(input_df)[0], 2)
        previous_price = predicted_price
    return previous_price

# Predict the price for the selected year
predicted_price = predict_price(unemployment_rate, inflation_rate, interest_rate, target_year)
st.subheader(f"Predicted Median House Price for {target_year}: ${predicted_price:,.2f}")

# Add spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Visualization 1: Predicted House Price Trend (2025–2030)
st.subheader("Predicted House Price Trend (2025–2030)")
future_years = list(range(2025, 2031))
future_prices = [predict_price(unemployment_rate, inflation_rate, interest_rate, year) for year in future_years]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(future_years, future_prices, marker='o', label='Predicted Prices')
ax.set_title("Predicted Median House Prices (2025–2030)")
ax.set_xlabel("Year")
ax.set_ylabel("Median House Price (USD)")
ax.grid()
ax.legend()
st.pyplot(fig)

# Add spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Visualization 2: Historical Median House Prices
st.subheader("Historical Median House Prices")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Year'], data['Median_House_Price'], marker='o', label='Historical Prices')
ax.set_title("Historical Median House Prices")
ax.set_xlabel("Year")
ax.set_ylabel("Median House Price (USD)")
ax.grid()
ax.legend()
st.pyplot(fig)

# Add spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Visualization 3: Economic Rates Over Time
st.subheader("Economic Rates Over Time")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Year'], data['Unemployment_Rate'], label='Unemployment Rate (%)', marker='o')
ax.plot(data['Year'], data['Inflation_Rate'], label='Inflation Rate (%)', marker='s')
ax.plot(data['Year'], data['Interest_Rate'], label='Interest Rate (%)', marker='^')
ax.set_title("Economic Rates Over Time")
ax.set_xlabel("Year")
ax.set_ylabel("Percentage (%)")
ax.grid()
ax.legend()
st.pyplot(fig)



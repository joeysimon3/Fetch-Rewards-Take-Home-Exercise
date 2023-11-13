import pandas as pd
import tkinter as tk
from tkinter import ttk
from pandas import to_datetime
# Sample DataFrame
df = pd.read_csv('Data/predicted_data.csv')
print(df)
# Function to calculate the sum
def calculate_sum():
	start_date = entry_start_date.get()
	start_mask = (df['month'] == int(start_date.split('-')[0])) & (df['day'] == int(start_date.split('-')[1]))
	start_index = df.index[start_mask][0]
	end_date = entry_end_date.get()
	end_mask = (df['month'] == int(end_date.split('-')[0])) & (df['day'] == int(end_date.split('-')[1]))
	end_index = df.index[end_mask][0]
	sum_value = int(round(df['predicted_values'][start_index:end_index + 1].sum()))
	label_result.config(text=f"Predicted Receipts for Date Range: {sum_value:,}")

# Create the main window
root = tk.Tk()
root.title("Date Range Sum Calculator")

# Create widgets
label_start_date = tk.Label(root, text="Start Date (MM-DD):")
label_end_date = tk.Label(root, text="End Date (MM-DD):")
entry_start_date = tk.Entry(root)
entry_end_date = tk.Entry(root)
button_calculate = tk.Button(root, text="Calculate", command=calculate_sum)
label_result = tk.Label(root, text="Sum: ")

# Layout widgets
label_start_date.grid(row=0, column=0)
entry_start_date.grid(row=0, column=1)
label_end_date.grid(row=1, column=0)
entry_end_date.grid(row=1, column=1)
button_calculate.grid(row=2, column=0, columnspan=2)
label_result.grid(row=3, column=0, columnspan=2)

# Run the application
root.mainloop()


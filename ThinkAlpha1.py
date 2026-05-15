import csv, os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "D:\\Python project\\sample_data.csv"


def display_table(df):
    print(df)

def duplications(df):
    duplicate_rows = df[df.duplicated()]
    print(f"Duplicated rows: {duplicate_rows}")
    total_duplicates = df.duplicated().sum()
    print(f"There are: {total_duplicates} duplicated rows")

def plot_columns(df, chart_type, column1, column2):
    match chart_type:
        case "1": 
            plt.plot(df[column1], df[column2], marker='o', linestyle='-')
            plt.xlabel(column1)
            plt.ylabel(column2)
            plt.title(f"Line Graph of {column1} vs {column2}")
            plt.show()

        case "2":
            plt.bar(df[column1], df[column2])
            plt.xlabel(column1)
            plt.ylabel(column2)
            plt.title(f"Bar Graph of {column1} vs {column2}")
            plt.show()

        case "3":
            plt.scatter(df[column1], df[column2])
            plt.xlabel(column1)
            plt.ylabel(column2)
            plt.title(f"Scatter plot of {column1} vs {column2}")
            plt.show()

def main():
    try:
        df = pd.read_csv(file_path)

        is_running = True
        while is_running:
            print("Available Functions:")
            print("1. Display table\n2. Check for duplications\n3. Plot\n4. Quit")
            prompt = input("Please choose your function: ")

            while not prompt == "1" and not prompt == "2" and not prompt == "3" and not prompt == "4":
                print("Your prompt is invalid!")
                prompt = input("Please choose your function: ")

            match prompt:
                case "1":
                    display_table(df)

                case "2":
                    duplications(df)

                case "3":
                    print("Chart types: ")
                    print("1. Line\n2. Bar chart\n3. Scatter")
                    chart_type = input("Choose your chart type: ")

                    while not chart_type == "1" and not chart_type == "2" and not chart_type == "3":
                        print("Your prompt is invalid!")
                        chart_type = input("Choose your chart type: ")

                    columns_names = df.columns
                    print(columns_names)

                    column1 = input("Choose a column as x axis: ")
                    column2 = input("Choose a column as y axis: ")
                    plot_columns(df, chart_type, column1, column2)

                case "4":
                    is_running = False


        print("Thank you for using the program")

    except FileNotFoundError:
        print("That file was not found")

    except PermissionError:
        print("You do not have permission to read that file")
    
if __name__ == "__main__":
    main()
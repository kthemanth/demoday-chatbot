# this package consolidates all the charting functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

def df_initial_preproc(df):
    town_mapping = {
    "TAMPINES": "Tampines",
    "YISHUN": "Yishun",
    "JURONG WEST": "Jurong West",
    "BEDOK": "Bedok",
    "WOODLANDS": "Woodlands",
    "ANG MO KIO": "Ang Mo Kio",
    "HOUGANG": "Hougang",
    "BUKIT BATOK": "Bukit Batok",
    "CHOA CHU KANG": "Choa Chu Kang",
    "BUKIT MERAH": "Bukit Merah",
    "SENGKANG": "Sengkang",
    "PASIR RIS": "Pasir Ris",
    "TOA PAYOH": "Toa Payoh",
    "QUEENSTOWN": "Queenstown",
    "GEYLANG": "Geylang",
    "CLEMENTI": "Clementi",
    "BUKIT PANJANG": "Bukit Panjang",
    "KALLANG/WHAMPOA": "Kallang_Whampoa",
    "JURONG EAST": "Jurong East",
    "SERANGOON": "Serangoon",
    "PUNGGOL": "Punggol",
    "BISHAN": "Bishan",
    "SEMBAWANG": "Sembawang",
    "MARINE PARADE": "Marine Parade",
    "CENTRAL AREA": "Central Area",
    "BUKIT TIMAH": "Bukit Timah",
    "LIM CHU KANG": "Lim Chu Kang",
    }

    room_mapping = {
    "1 ROOM": "1 room",
    "2 ROOM": "2 room",
    "3 ROOM": "3 room",
    "4 ROOM": "4 room",
    "5 ROOM": "5 room",
    "EXECUTIVE": "Executive",
    "MULTI-GENERATION": "Multi-Gen",
    }

    df.month = pd.to_datetime(df.month)
    df["year_of_sales"] = df["month"].dt.year
    df["month_of_sales"] = df["month"].dt.month
    # Clean up the 'MULTI-GENERATION' entries
    df["flat_type"] = df["flat_type"].str.replace(
        "MULTI GENERATION", "MULTI-GENERATION")
    # add price per sqm
    df["price_per_sqm"] = df.resale_price / df.floor_area_sqm

    # change town and flat_type name to lower cap
    df["town_map"] = df["town"].map(town_mapping)
    df["flat_type_map"] = df["flat_type"].map(room_mapping)

    df.drop(["town", "flat_type"], axis=1, inplace=True)
    df.rename(columns={"town_map": "town", "flat_type_map": "flat_type"}, inplace=True)

    return df

# price per sqm across different town and all flat type
def plot_sqm_all_town(df):

    #df_initial_preproc(df)
    #df_query = df.query("flat_type == @room")

    sns.set_style("whitegrid")
    sns.set_palette("RdBu")

    # Calculate the mean resale price for each town
    mean_prices = df.groupby("town")["price_per_sqm"].mean().sort_values()

    # Create a new categorical order based on the sorted mean prices
    town_order = mean_prices.index.tolist()

    g = sns.catplot(
        data=df,
        x="price_per_sqm",
        y="town",
        kind="bar",
        height=6.5,
        aspect=1,
        errorbar=None,
        order=town_order,
    )
    g.fig.suptitle(f"Price Per Square Meter across all town and all flat type", y=1.03, fontsize=11)
    g.set(xlabel="Price Per Square Meter", ylabel="Town")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=10
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=10
    )  # Access and set y-axis label font size

    plt.xticks(rotation=0)

    # Format the x-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.xaxis.set_major_formatter(formatter)
    return g

# price per sqm across different town and selected flat type
def plot_sqm_all_town_2(df, room):

    #df_initial_preproc(df)
    df_query = df.query("flat_type == @room")

    sns.set_style("whitegrid")
    sns.set_palette("RdBu")

    # Calculate the mean resale price for each town
    mean_prices = df_query.groupby("town")["price_per_sqm"].mean().sort_values()

    # Create a new categorical order based on the sorted mean prices
    town_order = mean_prices.index.tolist()

    g = sns.catplot(
        data=df_query,
        x="price_per_sqm",
        y="town",
        kind="bar",
        height=6.5,
        aspect=1,
        errorbar=None,
        order=town_order,
    )
    g.fig.suptitle(f"Price Per Square Meter across all town and flat type: {room}", y=1.03, fontsize=11)
    g.set(xlabel="Price Per Square Meter", ylabel="Town")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=10
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=10
    )  # Access and set y-axis label font size

    plt.xticks(rotation=0)

    # Format the x-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.xaxis.set_major_formatter(formatter)
    return g

# price per sqm across single town and flat type
def plot_sqm_single_twn_room(df, room, twn):
    #df_initial_preproc(df)
    df_query = df.query("flat_type == @room & town == @twn")
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")

    g = sns.relplot(
        data=df_query,
        x="year_of_sales",
        y="price_per_sqm",
        kind="line",
        height=5,
        aspect=2,
        errorbar=None,
    )
    g.fig.suptitle(
        f"Price Per Square Meter across {twn} and flat type: {room}",
        y=1.01,
        fontsize=17,
    )
    g.set(xlabel="Year of sales", ylabel="Price Per Square Meter")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=16
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=16
    )  # Access and set y-axis label font size
    plt.xticks(rotation=0)

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)
    return g

# mean resale price across all town and all flat type
def plot_resale_price_all(df):
    #df_initial_preproc(df)
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")

    # Calculate the mean resale price for each town
    mean_prices = df.groupby("town")["resale_price"].mean().sort_values()

    # Create a new categorical order based on the sorted mean prices
    town_order = mean_prices.index.tolist()

    g = sns.catplot(
        data=df,
        x="town",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=1.7,
        errorbar=None,
        order=town_order,
    )
    g.fig.suptitle("Mean Resale Price across all town and all flat type", y=1.03, fontsize=15)
    g.set(xlabel="Town", ylabel="Resale Price")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=14
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=14
    )  # Access and set y-axis label font size
    plt.xticks(rotation=90)

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    return g

# mean resale price across single town and all flat type
def plot_resale_price_single(df, town):       ##changes: added df
    #df_initial_preproc(df)
    df_query = df.query("town == @town")
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")
    hue_order = [
        "1 room",
        "2 room",
        "3 room",
        "4 room",
        "5 room",
        "Executive",
        "Multi-Gen",
    ]
    g = sns.catplot(
        data=df_query,
        x="town",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=1.5,
        errorbar=None,
        hue="flat_type",
        hue_order=hue_order,
        palette="bright",
    )

    g.set(xlabel="Town", ylabel="Resale Price")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=15
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=15
    )  # Access and set y-axis label font size

    plt.xticks(rotation=0)

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    return g

# mean resale price across all town and selected flat type
def plot_resale_price_all_2(df, room):       ##changes: added df
    #df_initial_preproc(df)
    df_query = df.query("flat_type == @room")
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")

    # Calculate the mean resale price for each town
    mean_prices = df_query.groupby("town")["resale_price"].mean().sort_values()

    # Create a new categorical order based on the sorted mean prices
    town_order = mean_prices.index.tolist()

    g = sns.catplot(
        data=df_query,
        x="town",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=1.7,
        errorbar=None,
        order=town_order,
    )
    g.fig.suptitle(
        f"Mean Resale Price across all town and flat type: {room}", y=1.03, fontsize=15
    )
    g.set(xlabel="Town", ylabel="Resale Price")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=14
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=14
    )  # Access and set y-axis label font size

    plt.xticks(rotation=90)

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    return g

# mean resale price of each month across all town
def plot_pricePerMonth_all(df):
    #df_initial_preproc(df)
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")

    g = sns.catplot(
        data=df,
        x="month_of_sales",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=1.5,
        errorbar=None,
    )
    g.fig.suptitle(
        "Mean Resale Price of each month across all town and flat type",
        y=1.02,
        fontsize=12,
    )
    g.set(xlabel="Month of Sales", ylabel="Resale Price")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=11
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=11
    )  # Access and set y-axis label font size

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    return g

# mean resale price of each month across single town
def plot_pricePerMonth_single(df, room, twn):
    #df_initial_preproc(df)
    df_query = df.query("flat_type == @room & town == @twn")
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")

    g = sns.catplot(
        data=df_query,
        x="month_of_sales",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=1.5,
        errorbar=None,
    )
    g.fig.suptitle(
        f"Mean Resale Price of each month across {twn} and flat type: {room}",
        y=1.03,
        fontsize=12,
    )
    g.set(xlabel="Month of Sales", ylabel="Resale Price")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=11
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=11
    )  # Access and set y-axis label font size

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    return g


# mean resale price of each month across all town and selected room
def plot_pricePerMonth_all_2(df, room):
    #df_initial_preproc(df)
    df_query = df.query("flat_type == @room")
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")

    g = sns.catplot(
        data=df_query,
        x="month_of_sales",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=1.5,
        errorbar=None,
    )
    g.fig.suptitle(
        f"Mean Resale Price of each month across all town and flat type: {room}",
        y=1.03,
        fontsize=12,
    )
    g.set(xlabel="Month of Sales", ylabel="Resale Price")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=11
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=11
    )  # Access and set y-axis label font size

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    return g


# resale price trend across all town
def plot_priceTrend_all(df):
    #df_initial_preproc(df)
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")
    hue_order = [
        "1 room",
        "2 room",
        "3 room",
        "4 room",
        "5 room",
        "Executive",
        "Multi-Gen",
    ]
    g = sns.relplot(
        data=df,
        x="month",
        y="resale_price",
        kind="line",
        height=5.5,
        aspect=1.5,
        palette="bright",
        errorbar=None,
        hue="flat_type",
        hue_order=hue_order,
    )
    g.fig.suptitle(f"Resale Price Trend across all town and all flat type", y=1.01, fontsize=18)
    g.set(xlabel="Year", ylabel="Resale Price")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=17
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=17
    )  # Access and set y-axis label font size

    plt.ticklabel_format(style="plain", axis="y")

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    return g

# resale price trend across single town and flat type
def plot_priceTrend_single(df, room, twn):
    #df_initial_preproc(df)
    df_query = df.query("flat_type == @room & town == @twn")
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")
    g = sns.relplot(
        data=df_query,
        x="month",
        y="resale_price",
        kind="line",
        height=5,
        aspect=2,
        errorbar=None,
    )
    g.fig.suptitle(
        f"Resale Price Trend across {twn} and flat type: {room}", y=1.01, fontsize=16
    )
    g.set(xlabel="Year", ylabel="Resale Price")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=14
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=14
    )  # Access and set y-axis label font size

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")

    g.ax.yaxis.set_major_formatter(formatter)

    return g

# resale price trend across single town and all flat type
def plot_priceTrend_allFlat(df, twn):
    #df_initial_preproc(df)
    df_query = df.query("town == @twn")
    sns.set_style("whitegrid")
    sns.set_palette("RdBu")

    hue_order = [
        "1 room",
        "2 room",
        "3 room",
        "4 room",
        "5 room",
        "Executive",
        "Multi-Gen",
    ]

    g = sns.relplot(
        data=df_query,
        x="month",
        y="resale_price",
        kind="line",
        height=5.5,
        aspect=1.5,
        errorbar=None,
        hue="flat_type",
        hue_order=hue_order,
        palette="bright",
    )
    g.fig.suptitle(
        f"Resale Price Trend across {twn} and all flat type", y=1.01, fontsize=17
    )
    g.set(xlabel="Year", ylabel="Resale Price")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=16
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=16
    )  # Access and set y-axis label font size

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")

    g.ax.yaxis.set_major_formatter(formatter)

    return g

# summary of mean resale price across single town and room
def data_resale_price_single(df, room, twn):
    #df_initial_preproc(df)
    df_query = df.query("flat_type == @room & town == @twn")
    price_summary_series = df_query["resale_price"].agg(["max", "min", "mean"])
    # Convert the Series to a DataFrame
    price_summary_df = (
        (
            price_summary_series.to_frame(
                name="Resale price (SGD)"
            )
        )
        .round()
        .astype(int)
    )

    # Apply the formatting to the column
    price_summary_df[price_summary_df.columns[0]] = price_summary_df[
        price_summary_df.columns[0]
    ].apply(lambda x: "${:,}".format(x))

    return price_summary_df

# summary of price per sqm across single town and room
def data_sqm_single_twn_room(df, room, twn):
    #df_initial_preproc(df)
    df_query = df.query("flat_type == @room & town == @twn")
    price_summary_series = df_query["price_per_sqm"].agg(["max", "min", "mean"])
    # Convert the Series to a DataFrame
    price_summary_df = (
        (
            price_summary_series.to_frame(
                name="Price per sqm (SGD)"
            )
        )
        .round()
        .astype(int)
    )

    # Apply the formatting to the column
    price_summary_df[price_summary_df.columns[0]] = price_summary_df[
        price_summary_df.columns[0]
    ].apply(lambda x: "${:,}".format(x))

    return price_summary_df

# last resale price of selected town and room
def data_last_resale_price(df, room, twn):

    df_query = df.query("flat_type == @room & town == @twn")

    df_last_resale_price = df_query.sort_index(ascending=False)
    last_resale_price = df_last_resale_price.iloc[0]["resale_price"]

    formatted_price = "${:,}".format(int(round(last_resale_price)))

    return formatted_price
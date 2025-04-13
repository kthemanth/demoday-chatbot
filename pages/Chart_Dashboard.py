
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from hdb_charts import df_initial_preproc
from hdb_charts import plot_sqm_all_town
from hdb_charts import plot_sqm_all_town_2
from hdb_charts import plot_sqm_single_twn_room
from hdb_charts import plot_resale_price_all
from hdb_charts import plot_resale_price_all_2
from hdb_charts import plot_resale_price_single
from hdb_charts import plot_pricePerMonth_all
from hdb_charts import plot_pricePerMonth_all_2
from hdb_charts import plot_pricePerMonth_single
from hdb_charts import plot_priceTrend_all
from hdb_charts import plot_priceTrend_single
from hdb_charts import plot_priceTrend_allFlat
from hdb_charts import data_resale_price_single
from hdb_charts import data_sqm_single_twn_room
from hdb_charts import data_last_resale_price


#import data
df = pd.read_csv("data_concat.csv", header=0)

# run intial preprocessing
df_initial_preproc(df)

# Get unique values for dropdowns
flat_types = sorted(df['flat_type'].unique())
towns = sorted(df['town'].unique())

#start of streamlit app
st.title("HDB buying & selling companion")
st.header("Chart Dashboard")

# Create select boxes for user input
st.markdown("\n\n\n\n\n\n\n")
#st.dataframe(df)

col1, col2= st.columns(2)
with col1:
    selected_room = st.selectbox("Select Flat Type:", options=flat_types, index=3)
    st.markdown("\n\n\n")

with col2:
    selected_town = st.selectbox("Select Town:", options=towns, index=0)
    st.markdown("\n\n\n")

# Call the plotting function and display the plot
if selected_room and selected_town:
    st.markdown("\n\n\n\n\n")
    st.subheader(f"Summary of {selected_town}, {selected_room} flat:")
    # Create a row layout
    col3, col4, col5= st.columns(3)

    with col3:
        price_summary_df = data_resale_price_single(df, selected_room, selected_town)
        st.dataframe(price_summary_df)

    with col4:
        price_summary_df_sqm = data_sqm_single_twn_room(df, selected_room, selected_town)
        st.dataframe(price_summary_df_sqm)

    with col5:
        last_resale_price = data_last_resale_price(df, selected_room, selected_town)
        st.metric(label="Last Resale Price (SGD)", value=f"{last_resale_price}")

    st.markdown("\n\n\n\n\n")
    tab1, tab2, tab3= st.tabs([f"{selected_town}", f"{selected_town}-All Flat", "All Town-All Flat"])
    with tab1:
        chart1 = plot_priceTrend_single(df, selected_room, selected_town)
        st.markdown("\n\n\n")
        st.pyplot(chart1)

    with tab2:
        chart2 = plot_priceTrend_allFlat(df, selected_town)
        st.markdown("\n\n\n")
        st.pyplot(chart2)

    with tab3:
        chart3 = plot_priceTrend_all(df)
        st.markdown("\n\n\n")
        st.pyplot(chart3)


    st.markdown("\n\n\n\n\n")
    tab3, tab3a, tab4= st.tabs([f"{selected_town}", f"All Town-{selected_room}", "All Town-All Flat"])
    with tab3:
        chart3 = plot_resale_price_single(df, selected_town)
        st.markdown("\n\n\n")
        st.pyplot(chart3)

    with tab3a:
        chart3a = plot_resale_price_all_2(df, selected_room)
        st.markdown("\n\n\n")
        st.pyplot(chart3a)

    with tab4:
        chart4 = plot_resale_price_all(df)
        st.markdown("\n\n\n")
        st.pyplot(chart4)


    st.markdown("\n\n\n\n\n")
    tab5, tab6, tab6a = st.tabs([f"{selected_town}", f"All Town-{selected_room}", f"All Town-All Flat"])
    with tab5:
        chart5 = plot_sqm_single_twn_room(df, selected_room, selected_town)
        st.markdown("\n\n\n")
        st.pyplot(chart5)

    with tab6:
        chart6 = plot_sqm_all_town_2(df, selected_room)
        st.markdown("\n\n\n")
        st.pyplot(chart6)

    with tab6a:
        chart6a = plot_sqm_all_town(df)
        st.markdown("\n\n\n")
        st.pyplot(chart6a)

    #st.markdown("\n\n\n\n\n")
    #tab7, tab7a, tab8= st.tabs([f"{selected_town}", f"All Town-{selected_room}", "All Town-All Flat"])
    #with tab7:
        #chart7 = plot_pricePerMonth_single(df, selected_room, selected_town)
        #st.markdown("\n\n\n")
        #st.pyplot(chart7)

    #with tab7a:
        #chart7a = plot_pricePerMonth_all_2(df, selected_room)
        #st.markdown("\n\n\n")
        #st.pyplot(chart7a)


    #with tab8:
        #chart8 = plot_pricePerMonth_all(df)
        #st.markdown("\n\n\n")
        #st.pyplot(chart8)


else:
    st.info("Please select a Flat Type and Town to view the trend.")

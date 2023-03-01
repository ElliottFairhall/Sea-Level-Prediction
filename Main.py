import streamlit as st
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import mpld3

PAGE_ICON = ":chart_with_upwards_trend:"
PAGE_TITLE = "Data Engineer, Educator Analyst and Technology Enthusiast"

# Set the title and icon of the application
st.set_page_config(page_title = PAGE_TITLE, page_icon = PAGE_ICON, layout="centered")

# Get the current directory and open the css file
current_dir = Path(__file__).parent if "_file_" in locals() else Path.cwd()
home_page = current_dir / "Home_Page.py"
sea_level_image = current_dir / "assets"/ "images" / "Sea Level.jpg"
csv_file = current_dir / "assets" / "data" / "epa-sea-level.csv" 
css_file = current_dir / "styles" / "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# provide title for page
st.markdown("<h1>Sea Level Prediction</h1>", unsafe_allow_html=True)

st.markdown("---")

#open sea_level_image for page
image = Image.open(sea_level_image)
st.image(image)

# create information related to project outline
st.markdown("""
<p>In this project, I shall provide an analysis of sea level changes looking at data since 1980 considering their
implications. Rising sea levels have become a well documented topic in recent times, with significant impacts on
coastal communities, ecosystems, and infrastructure being impacted on a day to day basis. The study of sea level
changes can provide valuable insights into the broader climate system and help us understand the causes and
mechanisms of sea level rise. Information like this could be critical for governments, organisations, and
coastal communities to take action and implement solutions to support adaptation to rising sea level changes
 and minimise negative effects locally where most affected.</p>
""", unsafe_allow_html=True)

st.write("---")

# create information related to input parameters
st.markdown("""
<h2>Input Parameters</h2>
<p>By default, the sample data is loaded. If you want to upload your own data, press the 'Upload Data' button.</p>
""", unsafe_allow_html=True)

# Create a button to allow user to upload their own data
upload_data = st.button("Upload Data")
# Initialize a variable to store the uploaded file
file_data = None
# Check if the button was clicked
if upload_data:
    # Create a file uploader widget
    file_data = st.file_uploader("Upload a sea level data CSV file", type=["csv"])

# read_csv or read information provided via upload
@st.cache_data
def get_data():
    # Check if the user uploaded a file
    if file_data is not None:
        # Read the uploaded data
        data = pd.read_csv(file_data)
        return data
    else:
        # If no file was uploaded, use the sample data
        data = pd.read_csv(csv_file)
        return data

# Get the data to be plotted
data = get_data()

# Create a figure and an axis
fig, ax = plt.subplots()

# Create a scatter plot of the data
fig, ax = plt.subplots()
scatter = plt.scatter(data["Year"], data["CSIRO Adjusted Sea Level"])
plt.xlabel("Year")
plt.ylabel("Sea Level (inches)")
plt.title("Rise in Sea Level")

# Add the slider to adjust the data range of the plot
start_year = st.slider('Start Year', 1880, 2020, 1980)
end_year = st.slider('End Year', 1980, 2051, 2020)
plt.xlim(start_year, end_year)

# Perform linear regression on the data
res = linregress(data["Year"], data["CSIRO Adjusted Sea Level"])
x_pred = pd.Series([float(i) for i in range(start_year, end_year+1)])
y_pred = res.slope*x_pred + res.intercept
plt.plot(x_pred, y_pred, 'r')

# Add hover tooltip to the data points
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=data["Year"])
mpld3.plugins.connect(fig, tooltip)

# Add a label to the x-axis indicating the units of measurement
plt.xlabel("Year")

# Add a label to the y-axis indicating the units of measurement
plt.ylabel("Sea Level (inches)")

# Set the title of the plot
plt.title("Rise in Sea Level") 

 # Display plot chart
st.pyplot(fig) 

st.write("---")

# provide a impact summary on the analysis
st.markdown("""
<h2>Impact on Coastal Communities, Ecosystems, and Infrastructure</h2>
<p>As you can see from the  line chart provided above, over the course of 40 years we have seen changes to sea levels
by up to 2 inches or 5.08cm. Looking at the red median line which shows the average over this time that sea levels are 
continuing to increase. 
<p> For example the impact to communities in the east coast US as well as south east asia like Bangkok, on a deeper
community-level the increase in rising sea levels impacts local economies, the production of crops and goods as well 
as educational prosperity for young people. In ecosystems the impact of rising sea levels can cause erosion of beaches, 
wetlands and increase saltwater intrusion impacting plants and local wildlife.<p>   
""", unsafe_allow_html=True)

# provide a conclusion on the analysis
st.markdown("""
<h2>Conclusion</h2>
<p>In this project, we aimed to analyse sea level changes and their impact on coastal communities, ecosystems, 
and infrastructure. To accomplish this, we used a scatter plot to visualise the data. The scatter plot was chosen 
as it is an appropriate choice for visualising the relationship between two continuous variables, in this case, 
the year and the sea level.</p><p>To create the scatter plot, we used the Matplotlib library/ module, which is a powerful 
data visualisation library in Python. We also used the Pandas library to read and manipulate the data, as it is a 
widely-used library for data manipulation and analysis. Additionally, we used the Scipy library's linregress function 
to perform linear regression on the data, which helped us to understand the trend of sea level changes.</p>
<p>In summary, this project used a scatter plot to visualise sea level changes over time and used the 
Matplotlib, Pandas and Scipy library to read, manipulate and analyse the data.</p>
""", unsafe_allow_html=True)
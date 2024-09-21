
# Project
A machine learning model is built to Use biological and physical measurements to predict the penguin species.  
Then, a streamlit app is developed for user interaction: users can input the values they observed to this model, and then get the model's prediction immediately.
  
# Tech Stack
Pandas, Sklearn, Docker, Streamlit
  
# Who works on it  
Yuding Duan

# Data
The original dataset is from [palmerpenguins](https://github.com/allisonhorst/palmerpenguins), and i use a cleaned version in this project.

# usage
1. clone this repo 
2. I'd suggest you to run the `streamlit_app.py` on your local machine first to adjust the visual display to the one you prefer the most. refer [streamlit documentation](https://docs.streamlit.io/)
3. build the image `docker build -t <image_name> .`  
4. run the container `docker run -p 8501:8501  <image_name>`  
And the you will get it.  



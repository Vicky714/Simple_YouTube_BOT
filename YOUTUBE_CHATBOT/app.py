import streamlit as st    
from Backend_botcode import chatbot

st.header("YOUTUBE_CHATBOT")

video_input=st.text_input('Enter the youtube video URL below to get started:')

user_input=st.text_input('Ask your Question and click start button to get your Response')

video_id = video_input.split('v=')[-1].split('&')[0] if video_input else None
# video_id ='https://www.youtube.com/watch?v=Ebyesd3mPAA'

if st.button('start'):
    result= st.write(chatbot(user_input, video_id))
    # st.write(video_id)
import numpy as np
from PIL import Image
from utility.face_utils import *
from streamlit_option_menu import option_menu
from pinecone import Pinecone
import base64
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

st.set_page_config(
    page_title="criminal detection", 
    page_icon="👨‍🎓", 
)

with open("images/back.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
f"""
<style>
.stApp {{
    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover
}}
</style>
""",
unsafe_allow_html=True
)
    
# ---- user authentication ----
names = ['yousef','youssef']
usernames = ['yhamdan','yibrahim']

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
                                    "user_dashboard","abcdef",cookie_expiry_days=30)

name,authentication_status,username = authenticator.login("login","main")

if authentication_status == False:
    st.error("username/password is incorrect")

if authentication_status == None:
    st.warning("please enter your username and password")

if authentication_status:
    # ---- HEADER SECTION ----

    #LOGOUT
    authenticator.logout('Logout','main')

    # Initializing Pinecone index
    # pinecone.init(api_key= "pcsk_2Kb6sm_KqYLEeBCeeSbLV2ZngQEfJMpgiPMnjKqbBMczcPG66QeC4X8NrbngmeY2vjMndk", environment="us-east-1")
    # index = pinecone.Index("criminals-index")
    pc = Pinecone(api_key="pcsk_2Kb6sm_KqYLEeBCeeSbLV2ZngQEfJMpgiPMnjKqbBMczcPG66QeC4X8NrbngmeY2vjMndk")
    index = pc.Index("criminals-index2")

    st.session_state.more_stuff = False

    #main menu
    # selected = option_menu(
    #     menu_title=None,  # None means no title
    #     options=["How to use", "Upload Image", "Upload Video", "Webcam"],
    #     icons=['info-circle', 'card-image', 'camera-video', 'camera'],  # same order as options
    #     menu_icon="cast",
    #     default_index=0,
    #     orientation="horizontal"
    # )

    selected = st.radio(    
        "",
        ["How to use", "Upload Image", "Upload Video", "Webcam"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
        )

    # Example usage
    # st.write(f"You selected: {selected}")

    st.header(selected)

    # How to use section
    if selected=="How to use":
        
        st.subheader("Upload an Image or Video")
        st.markdown("Simply :red[drag] and :blue[drop] your desired image or video")
        file_ = open("images/upload_gif_dark.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="upload gif">',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown("Once the file is done being processed a list of detected faces will appear.")
        st.subheader("Webcam")
        st.markdown("Click on the :red[RED START BUTTON] to start your camera then point it at the area where the detection will happen. Once you are done click on the 'detect faces' button and a list of detected faces will appear.")

    # Upload Image section
    if selected=="Upload Image":
        empty_files()
        uploaded_files = st.file_uploader("Choose an image",accept_multiple_files=True,type=["png","jpg","jpeg","pgm"])

        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                empty_files()
                st.image(image_face_detector(np.array(Image.open(uploaded_file).convert("RGB")),n=1,conf_thresh=0))
                
                for i, person in enumerate(glob("detected_faces/*.jpg")):
                    cn=0
                    
                    # Create a set to store the displayed labels
                    displayed_labels = set()
                    
                    query_response = index.query(
                        top_k=400,
                        include_values=False,
                        include_metadata=True,
                        vector=get_image_embedding(person).tolist(),
                    )

                    with st.expander(f"Person {i+1}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            img = Image.open(person).resize((224, 224))
                            st.subheader("Detected person image")
                            st.image(img)

                        with col2:
                            st.subheader("Criminal profile")
                            cols = st.columns(3)
                            
                            for col in cols:
                                label = query_response['matches'][cn]["metadata"]["label"]
                                # Check if the label has already been displayed
                                while label in displayed_labels:
                                    # Find the next available label
                                    if cn == len(query_response['matches'])-1:
                                        break  # Break the loop if no more labels available
                                    cn += 1
                                    label = query_response['matches'][cn]["metadata"]["label"]

                                displayed_labels.add(label)
                                with col:
                                    st.image(Image.open(query_response['matches'][cn]["metadata"]["path"]).resize((224, 224)))
                                    st.divider()
                                    st.markdown("Name: " + label)
                                    st.markdown("Age: " + str(query_response['matches'][cn]["metadata"]["age"]))
                                    st.markdown("Gender: " + str(query_response['matches'][cn]["metadata"]["gender"]))
                                    st.markdown("Felony: " + str(query_response['matches'][cn]["metadata"]["felony"]))


    # Upload Video section
    if selected == "Upload Video":
        empty_files()
        uploaded_files = st.file_uploader("Choose videos", accept_multiple_files=True, type=["mp4", "avi", "mkv"])

        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                empty_files()
                video_face_detector(uploaded_file)
                best_images("detected_faces", get_conf())

                # Display results for each uploaded file
                for i, person in enumerate(glob("detected_faces/highest_score_images/*.jpg")):
                    cn=0
                    
                    # Create a set to store the displayed labels
                    displayed_labels = set()
                    
                    query_response = index.query(
                        top_k=30,
                        include_values=False,
                        include_metadata=True,
                        vector=get_image_embedding(person).tolist(),
                    )

                    with st.expander(f"Person {i+1}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            img = Image.open(person).resize((224, 224))
                            st.subheader("Detected person image")
                            st.image(img)

                        with col2:
                            st.subheader("Criminal profile")
                            cols = st.columns(3)
                            
                            for col in cols:
                                label = query_response['matches'][cn]["metadata"]["label"]
                                # Check if the label has already been displayed
                                while label in displayed_labels:
                                    # Find the next available label
                                    if cn == len(query_response['matches'])-1:
                                        break  # Break the loop if no more labels available
                                    cn += 1
                                    label = query_response['matches'][cn]["metadata"]["label"]

                                displayed_labels.add(label)
                                with col:
                                    st.image(Image.open(query_response['matches'][cn]["metadata"]["path"]).resize((224, 224)))
                                    st.divider()
                                    st.markdown("Name: " + label)
                                    st.markdown("Age: " + str(query_response['matches'][cn]["metadata"]["age"]))
                                    st.markdown("Gender: " + str(query_response['matches'][cn]["metadata"]["gender"]))
                                    st.markdown("Felony: " + str(query_response['matches'][cn]["metadata"]["felony"]))


        
    # Webcam section
    if selected=="Webcam":

        webrtc_streamer(key="example",video_frame_callback=video_frame_callback,media_stream_constraints={"video": True, "audio": False})
        
        click = st.button("Detect Faces")
        if click:
            st.session_state.more_stuff = True

        if st.session_state.more_stuff:
            best_images("detected_faces", get_conf())

            # Display results for each uploaded file
            for i, person in enumerate(glob("detected_faces/highest_score_images/*.jpg")):
                cn=0
                
                # Create a set to store the displayed labels
                displayed_labels = set()
                
                query_response = index.query(
                    top_k=30,
                    include_values=False,
                    include_metadata=True,
                    vector=get_image_embedding(person).tolist(),
                )

                with st.expander(f"Person {i+1}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        img = Image.open(person).resize((224, 224))
                        st.subheader("Detected person image")
                        st.image(img)

                    with col2:
                        st.subheader("Criminal profile")
                        cols = st.columns(3)
                        
                        for col in cols:
                            label = query_response['matches'][cn]["metadata"]["label"]
                            # Check if the label has already been displayed
                            while label in displayed_labels:
                                # Find the next available label
                                if cn == len(query_response['matches'])-1:
                                    break  # Break the loop if no more labels available
                                cn += 1
                                label = query_response['matches'][cn]["metadata"]["label"]

                            displayed_labels.add(label)
                            with col:
                                st.image(Image.open(query_response['matches'][cn]["metadata"]["path"]).resize((224, 224)))
                                st.divider()
                                st.markdown("Name: " + label)
                                st.markdown("Age: " + str(query_response['matches'][cn]["metadata"]["age"]))
                                st.markdown("Gender: " + str(query_response['matches'][cn]["metadata"]["gender"]))
                                st.markdown("Felony: " + str(query_response['matches'][cn]["metadata"]["felony"]))

    empty_files()

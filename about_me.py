import streamlit as st
from PIL import Image
def app(df):
    
    st.balloons()
    st.title("Contributers of This Project")


    img = Image.open("rohan.jpg")
    st.image(img,width=200)
    st.markdown("### Name: ")
    st.text("Rohan Goyal")

    st.markdown("""### Work:""")
    st.text("Data Anayst Intern")

    st.markdown("""### Company :""")
    st.markdown("[ShapeAI](https://www.shapeai.tech/)")
                
    st.markdown("""### Github:""")
    st.markdown("[Rohan Goyal](https://github.com/37rohan)")

    st.markdown("""### Linkedin:""")
    st.markdown("#### [Rohan Goyal](https://www.linkedin.com/in/rohan-goyal-3a1744200/)")
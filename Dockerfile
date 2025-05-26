FROM python:3.9-slim 


# Install dependencies
COPY requirements/requirements.in requirements.in
RUN pip install pip-tools && pip-compile -r requirements.in && pip install -r requirements.txt

COPY .env .env
COPY streamlit_app.py vector_loader.py input_pdf.pdf ./
# COPY vector_loader.py vector_loader.py
EXPOSE 8501
# Set environment variable for streamlit
ENV STREAMLIT_PORT=8501

RUN python vector_loader.py
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 